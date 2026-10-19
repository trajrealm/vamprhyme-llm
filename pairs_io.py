"""Read and write rhyme pairs in a plain-text format that is easy to review and edit."""
import re
from pathlib import Path

HEADER = re.compile(r"^=== ?([\w-]+):?$")
SECTION = re.compile(r"^--- ?(original|dark):?$", re.IGNORECASE)
DECORATION = "*#>_` \t"


def format_pairs(pairs: list[dict]) -> str:
    return "\n".join(
        f"=== {p['id']}\n--- original\n{p['original']}\n--- dark\n{p.get('dark', '')}\n" for p in pairs
    )


def parse_pairs(text: str) -> list[dict]:
    """Parse blocks, tolerating the markdown decoration and code fences chat models often add."""
    pairs, current, section = [], None, None
    for line in text.splitlines():
        bare = line.strip(DECORATION)
        if line.strip().startswith("```"):
            continue
        if header := HEADER.match(bare):
            current = {"id": header.group(1), "original": [], "dark": []}
            pairs.append(current)
            section = None
        elif current and (marker := SECTION.match(bare)):
            section = marker.group(1).lower()
        elif current and section:
            current[section].append(line)
    return [{"id": p["id"], "original": "\n".join(p["original"]).strip(), "dark": "\n".join(p["dark"]).strip()}
            for p in pairs]


def read_pairs(path: Path) -> list[dict]:
    return parse_pairs(path.read_text(encoding="utf-8"))


def write_pairs(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_pairs(pairs), encoding="utf-8")
