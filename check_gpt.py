"""Sanity checks for the GPT model before real training."""
import argparse
import math
import time
from pathlib import Path

import torch

from data import TokenStream
from models_gpt import GPT
from tokenizer import CharTokenizer
from train_gpt import add_gpt_args, gpt_config


def report(name: str, passed: bool, detail: str) -> None:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    tok = CharTokenizer.load(args.data / "tokenizer.json")
    config = {**gpt_config(args, tok.vocab_size), "dropout": 0.0}
    model = GPT(**config)
    stream = TokenStream(args.data, ["rhymes"], "train")
    x, y = stream.batch(args.batch_size, args.block_size, "cpu")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        print(f"  {name:10s} {count:>10,}")
    print("  (lm_head shares its weights with token_emb)\n")

    logits, loss = model(x, y)
    expected_shape = (args.batch_size, args.block_size, tok.vocab_size)
    report("output shape", tuple(logits.shape) == expected_shape, f"{tuple(logits.shape)}")

    expected_loss = math.log(tok.vocab_size)
    report("initial loss", abs(loss.item() - expected_loss) < 0.3,
           f"{loss.item():.3f} (uniform guessing would be {expected_loss:.3f})")

    cut = args.block_size // 2
    changed = x[:1].clone()
    changed[0, cut:] = torch.randint(tok.vocab_size, (args.block_size - cut,))
    before, after = model(x[:1])[0], model(changed)[0]
    past_same = torch.allclose(before[:, :cut], after[:, :cut])
    future_differs = not torch.allclose(before[:, cut:], after[:, cut:])
    report("causal mask", past_same and future_differs,
           f"positions before {cut} unchanged when later tokens change")

    tiny_x, tiny_y = x[:4], y[:4]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(args.overfit_steps + 1):
        _, tiny_loss = model(tiny_x, tiny_y)
        if step % 50 == 0:
            print(f"    overfit step {step:>3}  loss {tiny_loss.item():.4f}")
        optimizer.zero_grad(set_to_none=True)
        tiny_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    report("overfit one batch", tiny_loss.item() < 0.2, f"final loss {tiny_loss.item():.4f}")

    model = GPT(**{**config, "dropout": args.dropout})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    timed_steps = 20
    start = time.time()
    for _ in range(timed_steps):
        bx, by = stream.batch(args.batch_size, args.block_size, "cpu")
        _, step_loss = model(bx, by)
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        optimizer.step()
    per_step = (time.time() - start) / timed_steps
    print(f"\nSpeed: {per_step * 1000:.0f} ms per training step "
          f"(about {per_step * 5000 / 60:.0f} min for 5,000 steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tokens"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--overfit-steps", type=int, default=200)
    add_gpt_args(parser)
    main(parser.parse_args())
