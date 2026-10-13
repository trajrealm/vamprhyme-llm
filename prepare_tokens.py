"""Build the tokenizer and encode the clean corpus into train/val token arrays."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from tokenizer import CharTokenizer

STYLES = ("rhymes", "horror", "bridge")
END = "<|end|>"
SPECIALS = [END] + [f"<|{style}|>" for style in STYLES]


def load_units(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_units(tok: CharTokenizer, style: str, texts: list[str]) -> np.ndarray:
    dtype = np.uint8 if tok.vocab_size <= 256 else np.uint16
    ids = []
    for text in texts:
        ids += tok.encode(f"<|{style}|>{text}{END}")
    return np.array(ids, dtype=dtype)


def main(units_path: Path, out_dir: Path, val_frac: float, seed: int) -> None:
    units = load_units(units_path)
    tok = CharTokenizer.from_texts((u["text"] for u in units), SPECIALS)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(out_dir / "tokenizer.json")
    print(f"Vocabulary: {tok.vocab_size} tokens ({len(SPECIALS)} special)\n")

    by_style = defaultdict(list)
    for u in units:
        by_style[u["style"]].append(u["text"])

    rng = random.Random(seed)
    for style in STYLES:
        texts = by_style[style]
        rng.shuffle(texts)
        n_val = int(len(texts) * val_frac)
        for split, subset in (("train", texts[n_val:]), ("val", texts[:n_val])):
            ids = encode_units(tok, style, subset)
            np.save(out_dir / f"{style}_{split}.npy", ids)
            print(f"  {style:7s} {split:5s} {len(subset):>5,} units  {len(ids):>10,} tokens")

    sample = f"<|rhymes|>Twinkle, twinkle, little bat!{END}"
    ids = tok.encode(sample)
    print(f"\nExample: {sample!r}\n  -> {ids}\n  -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=Path("data/clean/units.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/tokens"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    main(args.units, args.out, args.val_frac, args.seed)
