"""Training loop, evaluation and checkpointing shared by all models."""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import TokenStream
from sample import generate
from tokenizer import CharTokenizer


def add_training_args(parser: argparse.ArgumentParser, *, lr: float, block_size: int, out: str) -> None:
    parser.add_argument("--styles", nargs="+", default=["rhymes"])
    parser.add_argument("--data", type=Path, default=Path("data/tokens"))
    parser.add_argument("--out", type=Path, default=Path(out))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=block_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")


def prepare(args: argparse.Namespace) -> tuple[CharTokenizer, dict[str, TokenStream]]:
    torch.manual_seed(args.seed)
    tok = CharTokenizer.load(args.data / "tokenizer.json")
    streams = {split: TokenStream(args.data, args.styles, split) for split in ("train", "val")}
    print(f"Train tokens: {len(streams['train']):,}  Val tokens: {len(streams['val']):,}")
    return tok, streams


@torch.no_grad()
def estimate_loss(model: nn.Module, streams: dict[str, TokenStream], args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, stream in streams.items():
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = stream.batch(args.batch_size, args.block_size, args.device)
            total += model(x, y)[1].item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def plot_losses(history: list[dict], path: Path) -> None:
    steps = [h["step"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [h["train"] for h in history], label="train")
    plt.plot(steps, [h["val"] for h in history], label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def train(model: nn.Module, model_name: str, model_config: dict, tok: CharTokenizer,
          streams: dict[str, TokenStream], args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model: {model_name}, {sum(p.numel() for p in model.parameters()):,} parameters\n")

    prompt = f"<|{args.styles[0]}|>"
    print("Sample before training:\n" + generate(model, tok, prompt, 200, 1.0, args.device) + "\n")

    history = []
    start = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, streams, args)
            history.append({"step": step, **losses})
            print(f"step {step:>6}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"({time.time() - start:.0f}s)")
        if step == args.steps:
            break

        x, y = streams["train"].batch(args.batch_size, args.block_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    torch.save({
        "model": model.state_dict(),
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer_path": str(args.data / "tokenizer.json"),
    }, args.out / "model.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    plot_losses(history, args.out / "loss.png")

    print("\nSample after training:\n" + generate(model, tok, prompt, 300, 1.0, args.device))
    print(f"\nSaved checkpoint, history and loss plot to {args.out}/")
