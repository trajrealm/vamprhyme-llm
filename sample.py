"""Generate text from a trained checkpoint."""
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


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model, tok = load_checkpoint(args.checkpoint, args.device)
    for i in range(args.num_samples):
        text = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.device)
        print(f"--- sample {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="<|rhymes|>")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
