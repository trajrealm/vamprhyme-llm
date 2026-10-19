"""Combine reviewed rhyme pairs with style examples into a fine-tuning set."""
import argparse
import json
import random
from pathlib import Path

from finetune_data import SYSTEM_PROMPT
from pairs_io import read_pairs

FULL_TEXT_PROMPTS = [
    "Write a Dracula version of this nursery rhyme:\n\n{rhyme}",
    "Rewrite this nursery rhyme as Victorian gothic horror:\n\n{rhyme}",
    "Make this nursery rhyme sinister, like something from a vampire tale:\n\n{rhyme}",
    "Turn this rhyme into a gothic horror verse:\n\n{rhyme}",
]
BY_NAME_PROMPTS = [
    'Write a Dracula version of the nursery rhyme "{name}".',
    'Write a gothic horror parody of "{name}".',
    'Rewrite "{name}" as a vampire rhyme.',
]
TEST_PROMPTS = [
    "Write a Dracula version of Twinkle, Twinkle, Little Star.",
    "Write a gothic horror parody of the nursery rhyme \"Twinkle, twinkle, little star\".",
]


def rhyme_name(rhyme: str) -> str:
    return rhyme.split("\n", 1)[0].rstrip(",;:.!?")


def pair_prompt(rhyme: str, rng: random.Random, by_name_share: float) -> str:
    if rng.random() < by_name_share:
        return rng.choice(BY_NAME_PROMPTS).format(name=rhyme_name(rhyme))
    return rng.choice(FULL_TEXT_PROMPTS).format(rhyme=rhyme)


def conversation(style: str, prompt: str, reply: str) -> dict:
    return {
        "style": style,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ],
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main(args: argparse.Namespace) -> None:
    if not args.pairs.exists():
        raise SystemExit(f"{args.pairs} not found. Draft the pairs first with draft_pairs.py (see pairs.md).")
    rng = random.Random(args.seed)
    seeds = read_pairs(args.seeds)
    reviewed = [p for p in read_pairs(args.pairs) if p["dark"]]
    rng.shuffle(reviewed)
    n_val = int(len(reviewed) * args.val_frac)
    pairs_train, pairs_val = seeds + reviewed[n_val:], reviewed[:n_val]

    train = [conversation("pairs", pair_prompt(p["original"], rng, args.by_name_share), p["dark"])
             for p in pairs_train for _ in range(args.pair_repeat)]
    val = [conversation("pairs", pair_prompt(p["original"], rng, args.by_name_share), p["dark"]) for p in pairs_val]

    style_train = read_jsonl(args.styles / "train.jsonl")
    for style in sorted({e["style"] for e in style_train}):
        examples = [e for e in style_train if e["style"] == style]
        train += rng.sample(examples, min(args.style_per, len(examples)))
    val += read_jsonl(args.styles / "val.jsonl")
    rng.shuffle(train)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)

    heldout = read_pairs(args.heldout)[: args.test_rhymes]
    tests = [{"prompt": p} for p in TEST_PROMPTS]
    for rhyme in heldout:
        tests.append({"prompt": BY_NAME_PROMPTS[0].format(name=rhyme_name(rhyme["original"]))})
        tests.append({"prompt": FULL_TEXT_PROMPTS[0].format(rhyme=rhyme["original"])})
    write_jsonl(args.out / "test_prompts.jsonl", tests)

    print(f"pairs: {len(seeds)} seeds + {len(reviewed)} reviewed -> {len(pairs_train)} train "
          f"(x{args.pair_repeat}), {len(pairs_val)} val")
    print(f"train: {len(train):,} examples  val: {len(val):,} examples  test prompts: {len(tests)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs/drafts.txt"))
    parser.add_argument("--seeds", type=Path, default=Path("pairs/seed_pairs.txt"))
    parser.add_argument("--heldout", type=Path, default=Path("data/pairs/heldout.txt"))
    parser.add_argument("--styles", type=Path, default=Path("data/sft"), help="output of finetune_data.py")
    parser.add_argument("--out", type=Path, default=Path("data/sft-pairs"))
    parser.add_argument("--style-per", type=int, default=300, help="style examples kept per style")
    parser.add_argument("--pair-repeat", type=int, default=2, help="times each training pair appears")
    parser.add_argument("--by-name-share", type=float, default=0.3,
                        help="share of pair prompts that name the rhyme instead of quoting it")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-rhymes", type=int, default=8, help="held-out rhymes used as test prompts")
    parser.add_argument("--seed", type=int, default=1337)
    main(parser.parse_args())
