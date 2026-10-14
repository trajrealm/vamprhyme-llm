"""Clean the raw Gutenberg corpus into tagged prose and verse units."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

STYLES = ("rhymes", "horror", "bridge")
VERSE_ONLY = {"rhymes"}
PROSE_LINE_LEN = 55
LIST_MIN_LINES = 8

# Per-file cut points for leftover front/back matter.
ANCHORS: dict[str, dict[str, str]] = {
    # "real-mother-goose.txt": {"start_at": "...", "end_at": "..."},
}

CHAR_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": "--", "\u2013": "-", "\u2026": "...", "\u00a0": " ",
    "\u2032": "'", "\u2044": "/", "\t": " ", "\u2020": "", "\u2021": "",
    "\u0153": "oe", "\u0152": "OE", "\u00e6": "ae", "\u00c6": "AE",
}

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
EDITORIAL = re.compile(r"^(transcriber|produced by|preparer|e-text prepared|\[?note)", re.I)
CHAPTER = re.compile(r"^(chapter|book|part|section)\s+[ivxlcdm\d]+\b|^[ivxlcdm]+\.?$", re.I)
PAGE_REF = re.compile(r"\d+\.?$")


def strip_gutenberg(text: str) -> str:
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def apply_anchors(text: str, name: str) -> str:
    anchors = ANCHORS.get(name, {})
    if start := anchors.get("start_at"):
        text = text[text.rfind("\n", 0, text.index(start)) + 1 :]
    if end := anchors.get("end_at"):
        text = text[: text.index(end)]
    return text


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    text = fold_accents(text.replace("\r\n", "\n"))
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r"\[(Illustration|Footnote)[^\]]*\]", "", text, flags=re.I | re.S)
    text = re.sub(r"\[(\d+|[A-Z])\]", "", text)
    text = re.sub(r"\{\d+\}", "", text)
    text = re.sub(r"_([^_]+?)_", r"\1", text, flags=re.S)
    return text.replace("_", "")


def is_noise(lines: list[str]) -> bool:
    letters = [c for c in " ".join(lines) if c.isalpha()]
    return (
        not letters
        or all(c.isupper() for c in letters)
        or (len(lines) == 1 and bool(CHAPTER.match(lines[0])))
        or bool(EDITORIAL.match(lines[0]))
        or sum(bool(PAGE_REF.search(line)) for line in lines) > len(lines) / 2
        or is_title_list(lines)
    )


def is_title_list(lines: list[str]) -> bool:
    # Contents pages and first-line indexes without page numbers: many lines, few ending in punctuation.
    return len(lines) >= LIST_MIN_LINES and sum(line[-1].isalpha() for line in lines) > 0.8 * len(lines)


def is_prose(lines: list[str], style: str) -> bool:
    if len(lines) == 1:
        return len(lines[0]) > PROSE_LINE_LEN or style not in VERSE_ONLY
    return any(len(line) > PROSE_LINE_LEN for line in lines[:-1])


def extract_units(text: str, style: str) -> list[dict]:
    units = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or is_noise(lines):
            continue
        prose = is_prose(lines, style)
        if prose and style in VERSE_ONLY:
            continue
        body = re.sub(r"\s+", " ", " ".join(lines)) if prose else "\n".join(lines)
        if not body.isascii():
            continue
        units.append({"kind": "prose" if prose else "verse", "text": body})
    return units


def describe(style: str, name: str, raw: str, units: list[dict]) -> list[str]:
    kinds = Counter(u["kind"] for u in units)
    clean_len = sum(len(u["text"]) for u in units)
    lines = [
        f"\n{'=' * 70}\n{style}/{name}",
        f"raw {len(raw):,} -> clean {clean_len:,} chars | "
        f"{len(units)} units ({kinds['verse']} verse, {kinds['prose']} prose)",
    ]
    samples = units[:3] + [{"text": "..."}] + units[-2:] if len(units) > 5 else units
    lines += ["  | " + u["text"][:300].replace("\n", "\n  | ") for u in samples]
    return lines


def main(raw_dir: Path, out_dir: Path) -> None:
    records, details, chars = [], [], Counter()
    totals = {style: Counter() for style in STYLES}

    for style in STYLES:
        (out_dir / style).mkdir(parents=True, exist_ok=True)
        for path in sorted((raw_dir / style).glob("*.txt")):
            raw = path.read_text(encoding="utf-8")
            units = extract_units(normalize(apply_anchors(strip_gutenberg(raw), path.name)), style)
            cleaned = "\n\n".join(u["text"] for u in units)
            (out_dir / style / path.name).write_text(cleaned + "\n", encoding="utf-8")

            records += [{"source": path.name, "style": style, **u} for u in units]
            totals[style].update(raw=len(raw), clean=len(cleaned), units=len(units))
            chars.update(cleaned)
            details += describe(style, path.name, raw, units)

    with open(out_dir / "units.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

    rare = ", ".join(f"{c!r}x{n}" for c, n in chars.most_common() if n < 20) or "none"
    summary = ["SUMMARY BY STYLE"]
    summary += [
        f"  {s:7s} raw {t['raw']:>10,}  clean {t['clean']:>10,}  units {t['units']:>6,}"
        for s, t in totals.items()
    ]
    summary += [f"\nVOCABULARY: {len(chars)} distinct characters", f"  Rare (<20 uses): {rare}"]

    (out_dir / "report.txt").write_text("\n".join(summary + details) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\nFull report: {out_dir / 'report.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/clean"))
    args = parser.parse_args()
    main(args.raw, args.out)
