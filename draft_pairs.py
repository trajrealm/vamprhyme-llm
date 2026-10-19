"""Draft dark versions of candidate rhymes with an LLM API, or write prompts to paste into a chat."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from pairs_io import format_pairs, parse_pairs, read_pairs, write_pairs

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_MODEL = "claude-sonnet-5-5"
RETRYABLE = {429, 500, 502, 503, 504}
STYLE_GUIDE = """You write dark parodies of traditional English nursery rhymes in the voice of Victorian \
gothic horror: Bram Stoker's Dracula, Edgar Allan Poe, and Sheridan Le Fanu's Carmilla.

Rules for each parody:
- Keep the original's number of lines, rhythm, and rhyme scheme. Every line that rhymes in the original must rhyme in the parody.
- Keep the opening words or a recognisable phrase, so the original is obvious.
- Replace the subject with gothic imagery: vampires, bats, crypts, graves, mist, moonlight, candles, castles, pale strangers.
- Create dread through atmosphere and implication, not gore.
- Use Victorian diction where it fits, never modern slang.

Here are examples of the style and format:

{examples}"""

TASK = """Write a dark parody for each rhyme below. Reply only with the blocks in exactly this format, \
keeping each id and original unchanged and writing your parody under "--- dark":

{blocks}"""


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def openrouter(path: str, payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json", "X-Title": "vamprhyme-llm"}
    if key := os.environ.get("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload else None
    request = urllib.request.Request(OPENROUTER_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def list_free_models() -> None:
    models = [m for m in openrouter("/models")["data"]
              if float(m["pricing"]["prompt"]) == 0 and float(m["pricing"]["completion"]) == 0]
    for m in sorted(models, key=lambda m: m["id"]):
        print(f"  {m['id']:60s} context {m.get('context_length') or 0:>9,}")
    print(f"{len(models)} free models")


def complete_openrouter(model: str, system: str, prompt: str, retries: int = 5) -> str:
    payload = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.9,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    for attempt in range(retries):
        try:
            data = openrouter("/chat/completions", payload)
            if "choices" in data:
                return data["choices"][0]["message"]["content"] or ""
            error = data.get("error", data)
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE:
                raise SystemExit(f"OpenRouter error {e.code}: {e.read().decode()[:500]}")
            error = f"HTTP {e.code}"
        wait = 10 * 2**attempt
        print(f"  {error}; retrying in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"OpenRouter kept failing after {retries} attempts")


def complete_anthropic(model: str, system: str, prompt: str) -> str:
    from anthropic import Anthropic

    response = Anthropic().messages.create(
        model=model, max_tokens=4000, system=system, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(block.text for block in response.content if block.type == "text")


def write_prompts(system: str, todo: list[dict], args: argparse.Namespace) -> None:
    args.prompts_dir.mkdir(parents=True, exist_ok=True)
    (args.prompts_dir / "00_style_guide.txt").write_text(system, encoding="utf-8")
    for n, batch in enumerate(batched(todo, args.batch), 1):
        (args.prompts_dir / f"batch_{n:03d}.txt").write_text(TASK.format(blocks=format_pairs(batch)), encoding="utf-8")
    args.out.touch()
    print(f"Wrote the style guide and {n} batch prompts to {args.prompts_dir}/")
    print(f"Paste the style guide, then each batch, into a chat, and append every reply to {args.out}.")


def main(args: argparse.Namespace) -> None:
    if args.list_free:
        list_free_models()
        return

    system = STYLE_GUIDE.format(examples=format_pairs(read_pairs(args.seeds)))
    drafts = {p["id"]: p for p in read_pairs(args.out)} if args.out.exists() else {}
    todo = [p for p in read_pairs(args.candidates) if not drafts.get(p["id"], {}).get("dark")]
    print(f"{sum(bool(p['dark']) for p in drafts.values())} already drafted, {len(todo)} to go")
    if not todo:
        return
    if args.prompts_only:
        write_prompts(system, todo, args)
        return

    if args.provider == "openrouter":
        if not args.model:
            raise SystemExit("Choose a model with --model; run with --list-free to see the free ones.")
        if "OPENROUTER_API_KEY" not in os.environ:
            raise SystemExit("Set OPENROUTER_API_KEY first.")
        complete = complete_openrouter
    else:
        complete = complete_anthropic
    model = args.model or ANTHROPIC_MODEL

    for n, batch in enumerate(batched(todo, args.batch), 1):
        text = complete(model, system, TASK.format(blocks=format_pairs(batch)))
        written = {p["id"]: p["dark"] for p in parse_pairs(text) if p["dark"]}
        for pair in batch:
            drafts[pair["id"]] = {**pair, "dark": written.get(pair["id"], "")}
        write_pairs(args.out, sorted(drafts.values(), key=lambda p: p["id"]))
        print(f"batch {n}: {len(written)}/{len(batch)} drafted")
        time.sleep(args.delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/pairs/candidates.txt"))
    parser.add_argument("--seeds", type=Path, default=Path("pairs/seed_pairs.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/pairs/drafts.txt"))
    parser.add_argument("--batch", type=int, default=8, help="rhymes per request or prompt file")
    parser.add_argument("--provider", choices=["openrouter", "anthropic"], default="openrouter")
    parser.add_argument("--model", help=f"model id; for anthropic defaults to {ANTHROPIC_MODEL}")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument("--list-free", action="store_true", help="list free OpenRouter models and exit")
    parser.add_argument("--prompts-only", action="store_true", help="write prompts to paste into a chat instead")
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/pairs/prompts"))
    main(parser.parse_args())
