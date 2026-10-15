"""Interactive prompt loop for comparing trained checkpoints side by side."""
import argparse
import readline  # noqa: F401  enables line editing and history in input()
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from sample import MODELS, generate, generate_blend
from tokenizer import CharTokenizer

STYLES = ("rhymes", "horror", "bridge")
HELP = """Type a prompt to generate from every active model. Use \\n for a line break.
Commands:
  /style rhymes|horror|bridge|none   tag added to prompts that don't start with one
  /temp 0.8                          sampling temperature
  /len 300                           maximum new characters
  /blend horror 0.6 | /blend off     blend the prompt's style with another
  /seed 1                            random seed (same seed for every model)
  /models                            list loaded models
  /use 1 3 | /use all                choose which models answer
  /settings                          show current settings
  /help  /quit"""


@dataclass
class Run:
    name: str
    model: nn.Module
    tok: CharTokenizer
    note: str


@dataclass
class Settings:
    style: str | None = "rhymes"
    temperature: float = 0.8
    max_new_tokens: int = 300
    blend_style: str | None = None
    blend_weight: float = 0.5
    seed: int = 0


def load_run(path: Path, device: str) -> Run:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MODELS[ckpt["model_name"]](**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.load(Path(ckpt["tokenizer_path"]))
    if ckpt["model_config"]["vocab_size"] != tok.vocab_size:
        print(f"warning: {path} was trained with {ckpt['model_config']['vocab_size']} tokens, "
              f"tokenizer now has {tok.vocab_size}; its output will be wrong")

    params = sum(p.numel() for p in model.parameters())
    note = f"{ckpt['model_name']}, {params:,} params"
    if "val_loss" in ckpt:
        note += f", val {ckpt['val_loss']:.4f} at step {ckpt['step']}"
    return Run(path.parent.name, model, tok, note)


def build_prompt(text: str, settings: Settings) -> str:
    text = text.replace("\\n", "\n")
    if settings.style and not text.startswith("<|"):
        return f"<|{settings.style}|>{text}"
    return text


def respond(runs: list[Run], text: str, settings: Settings, device: str) -> None:
    prompt = build_prompt(text, settings)
    if settings.blend_style and not prompt.startswith("<|"):
        print("Blending needs a style tag: set /style or start the prompt with one.")
        return
    for run in runs:
        torch.manual_seed(settings.seed)
        start = time.time()
        try:
            if settings.blend_style:
                out = generate_blend(run.model, run.tok, prompt, settings.blend_style, settings.blend_weight,
                                     settings.max_new_tokens, settings.temperature, device)
            else:
                out = generate(run.model, run.tok, prompt, settings.max_new_tokens, settings.temperature, device)
        except KeyError as e:
            print(f"The character {e} isn't in the vocabulary; remove it and try again.")
            return
        print(f"\n=== {run.name} ({time.time() - start:.1f}s) ===\n{out}")
    print()


def handle_command(line: str, runs: list[Run], active: list[Run], settings: Settings) -> list[Run]:
    cmd, *rest = line.split()
    if cmd == "/style":
        settings.style = None if rest[0] == "none" else rest[0]
    elif cmd == "/temp":
        settings.temperature = float(rest[0])
    elif cmd == "/len":
        settings.max_new_tokens = int(rest[0])
    elif cmd == "/seed":
        settings.seed = int(rest[0])
    elif cmd == "/blend":
        if rest[0] == "off":
            settings.blend_style = None
        else:
            settings.blend_style = rest[0]
            settings.blend_weight = float(rest[1]) if len(rest) > 1 else settings.blend_weight
    elif cmd == "/models":
        for i, run in enumerate(runs, 1):
            print(f"  {i}. {run.name:20s} {run.note}{'  (active)' if run in active else ''}")
    elif cmd == "/use":
        active = runs if rest == ["all"] else [runs[int(i) - 1] for i in rest]
        print("Active: " + ", ".join(r.name for r in active))
    elif cmd == "/settings":
        print(f"  {settings}")
    elif cmd == "/help":
        print(HELP)
    else:
        print(f"Unknown command {cmd}; type /help")
    return active


def main(args: argparse.Namespace) -> None:
    paths = args.checkpoints or sorted(Path("runs").glob("*/model.pt"))
    runs = [load_run(p, args.device) for p in paths]
    active = runs
    settings = Settings()

    print(f"Loaded {len(runs)} models:")
    handle_command("/models", runs, active, settings)
    print("\nType /help for commands.\n")

    while True:
        try:
            line = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        command = line.strip()
        if not command:
            continue
        if command in ("/quit", "/exit"):
            break
        if command.startswith("/"):
            try:
                active = handle_command(command, runs, active, settings)
            except (IndexError, ValueError):
                print("Couldn't read that command; type /help")
        else:
            respond(active, line, settings, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="defaults to every runs/*/model.pt")
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
