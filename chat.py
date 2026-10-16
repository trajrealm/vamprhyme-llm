"""Chat with a pretrained model, optionally with a LoRA adapter, and compare the two."""
import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_data import SYSTEM_PROMPT

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load(base: str | None, adapter: Path | None, device: str):
    base = base or (PeftConfig.from_pretrained(adapter).base_model_name_or_path if adapter else DEFAULT_MODEL)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def reply(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str,
          system: str | None = SYSTEM_PROMPT) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=temperature, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def respond(model, tokenizer, prompt: str, args: argparse.Namespace) -> None:
    def run(label: str) -> None:
        torch.manual_seed(args.seed)
        text = reply(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.device, args.system)
        print(f"\n=== {label} ===\n{text}")

    if args.adapter and args.compare:
        with model.disable_adapter():
            run("base model")
    run("fine-tuned" if args.adapter else "base model")
    print()


def main(args: argparse.Namespace) -> None:
    model, tokenizer = load(args.base, args.adapter, args.device)
    for prompt in args.prompts:
        print(f"> {prompt}")
        respond(model, tokenizer, prompt, args)
    if args.prompts:
        return
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in ("/quit", "/exit"):
            break
        if prompt:
            respond(model, tokenizer, prompt, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", help="prompts to answer; omit for an interactive session")
    parser.add_argument("--base", help=f"base model; defaults to the adapter's base or {DEFAULT_MODEL}")
    parser.add_argument("--adapter", type=Path, help="LoRA adapter folder from finetune_lora.py")
    parser.add_argument("--compare", action="store_true", help="also answer with the adapter switched off")
    parser.add_argument("--system", default=SYSTEM_PROMPT,
                        help="system prompt; pass an empty string for the model's own default")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
