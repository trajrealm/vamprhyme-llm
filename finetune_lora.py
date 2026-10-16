"""LoRA fine-tuning of a pretrained chat model on the style examples."""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import DEFAULT_MODEL, reply
from finetune_data import PROMPTS
from trainer import lr_at, plot_losses

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE = -100


def encode(tokenizer, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    # Loss only on the assistant's reply, not on the prompt.
    labels = [IGNORE] * n_prompt + ids[n_prompt:]
    return ids, labels[: len(ids)]


def load_examples(path: Path, tokenizer, max_len: int) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ids, labels = encode(tokenizer, record["messages"], max_len)
            examples.append({"style": record["style"], "ids": ids, "labels": labels})
    return examples


def collate(batch: list[dict], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    width = max(len(e["ids"]) for e in batch)
    ids = [e["ids"] + [pad_id] * (width - len(e["ids"])) for e in batch]
    labels = [e["labels"] + [IGNORE] * (width - len(e["labels"])) for e in batch]
    mask = [[1] * len(e["ids"]) + [0] * (width - len(e["ids"])) for e in batch]
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": torch.tensor(mask, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def train_batches(examples: list[dict], batch_size: int, steps: int, rng: random.Random):
    order: list[dict] = []
    for _ in range(steps):
        if len(order) < batch_size:
            order = examples[:]
            rng.shuffle(order)
        yield [order.pop() for _ in range(batch_size)]


@torch.no_grad()
def evaluate(model, val_by_style: dict[str, list[dict]], args: argparse.Namespace, pad_id: int) -> dict[str, float]:
    model.eval()
    losses = {}
    for style, examples in val_by_style.items():
        total = 0.0
        chunks = [examples[i : i + args.batch_size] for i in range(0, len(examples), args.batch_size)]
        for chunk in chunks:
            with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                total += model(**collate(chunk, pad_id, args.device)).loss.item()
        losses[f"val_{style}"] = total / len(chunks)
    losses["val"] = sum(losses.values()) / len(losses)
    model.train()
    return losses


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train = load_examples(args.data / "train.jsonl", tokenizer, args.max_len)
    val_by_style = defaultdict(list)
    for example in load_examples(args.data / "val.jsonl", tokenizer, args.max_len):
        val_by_style[example["style"]].append(example)

    args.steps = args.steps or math.ceil(args.epochs * len(train) / args.batch_size)
    print(f"{len(train):,} training examples, {sum(map(len, val_by_style.values())):,} validation, "
          f"{args.steps:,} steps")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(args.device, enabled=args.device == "cuda")

    history, running = [], []
    best_val, best_step = float("inf"), 0
    start = time.time()
    batches = train_batches(train, args.batch_size, args.steps, rng)
    for step in range(args.steps + 1):
        lr = lr_at(step, args)
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate(model, val_by_style, args, pad_id)
            train_loss = sum(running) / len(running) if running else losses["val"]
            running = []
            history.append({"step": step, "lr": lr, "train": train_loss, **losses})
            (args.out / "history.json").write_text(json.dumps(history, indent=2))
            improved = losses["val"] < best_val
            if improved:
                best_val, best_step = losses["val"], step
                model.save_pretrained(args.out / "adapter")
            per_style = "  ".join(f"{k[4:]} {v:.3f}" for k, v in losses.items() if k.startswith("val_"))
            print(f"step {step:>5}  train {train_loss:.4f}  val {losses['val']:.4f}  [{per_style}]  "
                  f"lr {lr:.2e}  ({time.time() - start:.0f}s){'  *' if improved else ''}")
        if step == args.steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            loss = model(**collate(next(batches), pad_id, args.device)).loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running.append(loss.item())

    plot_losses(history, best_step, args.out / "loss.png")
    print(f"\nBest val loss {best_val:.4f} at step {best_step}; adapter saved to {args.out / 'adapter'}")

    model.load_adapter(args.out / "adapter", adapter_name="best")
    model.set_adapter("best")
    model.eval()
    for style, prompts in PROMPTS.items():
        torch.manual_seed(args.seed)
        print(f"\n=== {prompts[0]} ===\n{reply(model, tokenizer, prompts[0], 200, 0.8, args.device)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data/sft"))
    parser.add_argument("--out", type=Path, default=Path("runs/lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, help="overrides --epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
