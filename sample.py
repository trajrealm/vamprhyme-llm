"""Generate text from a trained checkpoint, optionally blending two styles."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_bigram import BigramLanguageModel
from models_gpt import GPT
from tokenizer import CharTokenizer

END = "<|end|>"
MODELS = {"bigram": BigramLanguageModel, "gpt": GPT}


def load_checkpoint(path: Path, device: str) -> tuple[nn.Module, CharTokenizer]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MODELS[ckpt["model_name"]](**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer.load(Path(ckpt["tokenizer_path"]))


@torch.no_grad()
def generate(model: nn.Module, tok: CharTokenizer, prompt: str,
             max_new_tokens: int, temperature: float, device: str) -> str:
    end_id = tok.stoi[END]
    idx = torch.tensor([tok.encode(prompt)], device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.context_size :])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if next_id.item() == end_id:
            break
        idx = torch.cat([idx, next_id], dim=1)
    return tok.decode(idx[0].tolist())


@torch.no_grad()
def generate_blend(model: nn.Module, tok: CharTokenizer, prompt: str, blend_style: str, weight: float,
                   max_new_tokens: int, temperature: float, device: str) -> str:
    """Mix next-token predictions from two style tags. The prompt must start with a style tag."""
    end_id = tok.stoi[END]
    prompt_ids = tok.encode(prompt)
    tags = torch.tensor([[prompt_ids[0]], [tok.stoi[f"<|{blend_style}|>"]]], device=device)
    text = torch.tensor([prompt_ids[1:]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Both contexts share the text; the tag stays pinned at the front so it never scrolls out.
        recent = text[:, -(model.context_size - 1) :].expand(2, -1)
        logits, _ = model(torch.cat([tags, recent], dim=1))
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        mixed = (1 - weight) * log_probs[0] + weight * log_probs[1]
        next_id = torch.multinomial(F.softmax(mixed / temperature, dim=-1), num_samples=1)
        if next_id.item() == end_id:
            break
        text = torch.cat([text, next_id.view(1, 1)], dim=1)
    return tok.decode(text[0].tolist())


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model, tok = load_checkpoint(args.checkpoint, args.device)
    for i in range(args.num_samples):
        if args.blend_with:
            text = generate_blend(model, tok, args.prompt, args.blend_with, args.blend_weight,
                                  args.max_new_tokens, args.temperature, args.device)
        else:
            text = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.device)
        print(f"--- sample {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="<|rhymes|>")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--blend-with", choices=["rhymes", "horror", "bridge"])
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
