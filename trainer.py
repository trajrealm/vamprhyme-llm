"""Training loop, evaluation and checkpointing shared by all models."""
import argparse
import json
import math
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
    parser.add_argument("--style-weights", nargs="+", type=float,
                        help="sampling weight per style, in --styles order; defaults to dataset sizes")
    parser.add_argument("--data", type=Path, default=Path("data/tokens"))
    parser.add_argument("--out", type=Path, default=Path(out))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=block_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")


def prepare(args: argparse.Namespace) -> tuple[CharTokenizer, dict[str, TokenStream]]:
    torch.manual_seed(args.seed)
    tok = CharTokenizer.load(args.data / "tokenizer.json")
    streams = {split: TokenStream(args.data, args.styles, split, args.style_weights)
               for split in ("train", "val")}
    if len(args.styles) > 1:
        streams |= {f"val_{style}": TokenStream(args.data, [style], "val") for style in args.styles}

    print(f"Train tokens: {len(streams['train']):,}  Val tokens: {len(streams['val']):,}")
    mix = ", ".join(f"{s} {w:.0%}" for s, w in zip(args.styles, streams["train"].mix))
    print(f"Batch mix: {mix}")
    return tok, streams


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


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
    if len(args.styles) == 1:
        losses[f"val_{args.styles[0]}"] = losses["val"]
    return losses


def plot_losses(history: list[dict], best_step: int, path: Path) -> None:
    steps = [h["step"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [h["train"] for h in history], label="train")
    plt.plot(steps, [h["val"] for h in history], label="val")
    style_keys = [k for k in history[0] if k.startswith("val_")]
    if len(style_keys) > 1:
        for key in style_keys:
            plt.plot(steps, [h[key] for h in history], linestyle=":", label=key)
    plt.axvline(best_step, color="gray", linestyle="--", label=f"best val (step {best_step})")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_checkpoint(model: nn.Module, model_name: str, model_config: dict,
                    args: argparse.Namespace, step: int, val_loss: float) -> None:
    torch.save({
        "model": model.state_dict(),
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer_path": str(args.data / "tokenizer.json"),
        "step": step,
        "val_loss": val_loss,
    }, args.out / "model.pt")


def train(model: nn.Module, model_name: str, model_config: dict, tok: CharTokenizer,
          streams: dict[str, TokenStream], args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model: {model_name}, {sum(p.numel() for p in model.parameters()):,} parameters\n")

    untrained = generate(model, tok, f"<|{args.styles[0]}|>", 200, 1.0, args.device)
    print(f"Sample before training:\n{untrained}\n")

    history = []
    best_val, best_step = float("inf"), 0
    start = time.time()
    for step in range(args.steps + 1):
        lr = lr_at(step, args)
        if step % args.eval_every == 0:
            losses = estimate_loss(model, streams, args)
            history.append({"step": step, "lr": lr, **losses})
            (args.out / "history.json").write_text(json.dumps(history, indent=2))
            improved = losses["val"] < best_val
            if improved:
                best_val, best_step = losses["val"], step
                save_checkpoint(model, model_name, model_config, args, step, best_val)
            per_style = "  ".join(f"{k[4:]} {v:.3f}" for k, v in losses.items() if k.startswith("val_"))
            print(f"step {step:>6}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"[{per_style}]  lr {lr:.2e}  ({time.time() - start:.0f}s){'  *' if improved else ''}")
        if step == args.steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = streams["train"].batch(args.batch_size, args.block_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    plot_losses(history, best_step, args.out / "loss.png")

    model.load_state_dict(torch.load(args.out / "model.pt", weights_only=True)["model"])
    print(f"\nBest val loss {best_val:.4f} at step {best_step} (* marks each new best)")
    for style in args.styles:
        print(f"\nSample from best checkpoint ({style}):\n"
              + generate(model, tok, f"<|{style}|>", 300, 1.0, args.device))
    print(f"\nSaved checkpoint, history and loss plot to {args.out}/")
