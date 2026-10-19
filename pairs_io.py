"""Read and write rhyme pairs in a plain-text format that is easy to review and edit."""
import re
from pathlib import Path

HEADER = re.compile(r"^=== (\S+)[ \t]*$", re.MULTILINE)
FENCE = re.compile(r"^```.*$", re.MULTILINE)


def format_pairs(pairs: list[dict]) -> str:
    return "\n".join(
        f"=== {p['id']}\n--- original\n{p['original']}\n--- dark\n{p.get('dark', '')}\n" for p in pairs
    )


def parse_pairs(text: str) -> list[dict]:
    parts = HEADER.split(FENCE.sub("", text))
    pairs = []
    for pair_id, body in zip(parts[1::2], parts[2::2]):
        original, _, dark = body.partition("--- dark")
        pairs.append({
            "id": pair_id,
            "original": original.replace("--- original", "", 1).strip(),
            "dark": dark.strip(),
        })
    return pairs


def read_pairs(path: Path) -> list[dict]:
    return parse_pairs(path.read_text(encoding="utf-8"))


def write_pairs(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_pairs(pairs), encoding="utf-8")
