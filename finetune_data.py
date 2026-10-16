"""Build chat-format fine-tuning examples from the cleaned corpus."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a creative writing assistant. You write poems, nursery rhymes, parodies and fiction "
    "in any genre, including gothic horror and dark fairy tales."
)
PROMPTS = {
    "rhymes": [
        "Write a nursery rhyme.",
        "Write a traditional English nursery rhyme.",
        "Recite a short nursery rhyme.",
    ],
    "horror": [
        "Write a passage of Victorian gothic horror.",
        "Write a paragraph in the style of a nineteenth-century horror story.",
        "Continue a gothic horror tale with one paragraph.",
    ],
    "bridge": [
        "Write a darkly comic passage for children.",
        "Write a passage from a cautionary tale or a grim fairy tale.",
        "Write a short piece of grim children's verse or story.",
    ],
}
MIN_PROSE_CHARS = 200


def load_texts(units_path: Path) -> dict[str, list[str]]:
    by_style = defaultdict(list)
    with units_path.open(encoding="utf-8") as f:
        for line in f:
            unit = json.loads(line)
            if unit["kind"] == "verse" or len(unit["text"]) >= MIN_PROSE_CHARS:
                by_style[unit["style"]].append(unit["text"])
    return by_style


def to_example(style: str, text: str, rng: random.Random) -> dict:
    return {
        "style": style,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rng.choice(PROMPTS[style])},
            {"role": "assistant", "content": text},
        ],
    }


def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    splits = {"train": [], "val": []}
    for style, texts in sorted(load_texts(args.units).items()):
        rng.shuffle(texts)
        n_val = min(int(len(texts) * args.val_frac), args.max_val)
        val, train = texts[:n_val], texts[n_val : n_val + args.per_style]
        splits["val"] += [to_example(style, t, rng) for t in val]
        splits["train"] += [to_example(style, t, rng) for t in train]
        print(f"  {style:7s} {len(texts):>5,} usable  {len(train):>5,} train  {len(val):>4,} val")

    args.out.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        rng.shuffle(examples)
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
        print(f"{split}: {len(examples):,} examples -> {args.out / f'{split}.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=Path("data/clean/units.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/sft"))
    parser.add_argument("--per-style", type=int, default=1500, help="maximum training examples per style")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-val", type=int, default=100, help="maximum validation examples per style")
    parser.add_argument("--seed", type=int, default=1337)
    main(parser.parse_args())
