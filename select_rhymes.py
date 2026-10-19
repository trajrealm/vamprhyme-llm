"""Pick nursery rhymes to pair with dark versions, and hold some back for testing."""
import argparse
import json
import random
from pathlib import Path

from pairs_io import read_pairs, write_pairs


def first_line_key(text: str) -> str:
    return "".join(c for c in text.split("\n", 1)[0].lower() if c.isalpha())


def load_rhymes(units_path: Path, args: argparse.Namespace) -> list[str]:
    rhymes = []
    with units_path.open(encoding="utf-8") as f:
        for line in f:
            unit = json.loads(line)
            lines = unit["text"].split("\n")
            if (unit["style"] == "rhymes" and unit["kind"] == "verse"
                    and args.min_lines <= len(lines) <= args.max_lines
                    and len(unit["text"]) <= args.max_chars):
                rhymes.append(unit["text"])
    return rhymes


def main(args: argparse.Namespace) -> None:
    excluded = {first_line_key(p["original"]) for p in read_pairs(args.seeds)}
    excluded |= {"".join(c for c in word.lower() if c.isalpha()) for word in args.exclude}

    unique: dict[str, str] = {}
    for rhyme in load_rhymes(args.units, args):
        key = first_line_key(rhyme)
        if key and key not in unique and not any(e and key.startswith(e) for e in excluded):
            unique[key] = rhyme
    rhymes = list(unique.values())
    random.Random(args.seed).shuffle(rhymes)

    heldout, candidates = rhymes[: args.heldout], rhymes[args.heldout : args.heldout + args.count]
    write_pairs(args.out / "candidates.txt", [{"id": f"c{i:03d}", "original": r} for i, r in enumerate(candidates, 1)])
    write_pairs(args.out / "heldout.txt", [{"id": f"h{i:03d}", "original": r} for i, r in enumerate(heldout, 1)])
    print(f"{len(rhymes):,} eligible rhymes -> {len(candidates)} candidates, {len(heldout)} held out, in {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=Path("data/clean/units.jsonl"))
    parser.add_argument("--seeds", type=Path, default=Path("pairs/seed_pairs.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/pairs"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--heldout", type=int, default=30)
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--max-lines", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--exclude", nargs="*", default=["twinkle twinkle"],
                        help="first-line beginnings to leave out entirely, kept for testing")
    parser.add_argument("--seed", type=int, default=1337)
    main(parser.parse_args())
