"""Gradio demo: dark parodies of nursery rhymes from a LoRA-tuned Qwen2.5 model."""
import os
from pathlib import Path
from threading import Thread

import gradio as gr
import spaces
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ADAPTER = os.environ.get("ADAPTER_REPO", "trajrealm/vamprhyme-qwen2.5-1.5b-lora")
# Must match the system prompt used in training (finetune_data.py).
SYSTEM_PROMPT = (
    "You are a creative writing assistant. You write poems, nursery rhymes, parodies and fiction "
    "in any genre, including gothic horror and dark fairy tales."
)
EXAMPLES = [
    "Write a Dracula version of this nursery rhyme:\n\n"
    "There was an old woman tossed in a basket,\nSeventeen times as high as the moon;\n"
    "But where she was going no mortal could tell,\nFor under her arm she carried a broom.",
    "Rewrite this nursery rhyme as Victorian gothic horror:\n\n"
    "Mary, Mary, quite contrary,\nHow does your garden grow?\n"
    "With silver bells and cockle shells,\nAnd pretty maids all in a row.",
    "Write a gothic horror parody of the nursery rhyme \"Twinkle, twinkle, little star\".",
    "Make this nursery rhyme sinister, like something from a vampire tale:\n\n"
    "Hush-a-bye, baby, on the tree top,\nWhen the wind blows the cradle will rock.",
    "Write a passage of Victorian gothic horror.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

base_name = PeftConfig.from_pretrained(ADAPTER).base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_name)
# Load everything on the CPU first: on ZeroGPU, no GPU exists at startup, only during @spaces.GPU calls.
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_name, dtype=DTYPE), ADAPTER,
                                  torch_device="cpu")
model.to(DEVICE).eval()


def stream_reply(prompt: str, temperature: float, max_new_tokens: int, use_adapter: bool):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)

    def run() -> None:
        torch.manual_seed(0)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens), do_sample=True,
                      temperature=temperature, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad():
            if use_adapter:
                model.generate(**kwargs)
            else:
                with model.disable_adapter():
                    model.generate(**kwargs)

    Thread(target=run).start()
    reply = ""
    for chunk in streamer:
        reply += chunk
        yield reply


# Attaches a GPU on ZeroGPU hardware for the duration of the call; has no effect on CPU hardware.
@spaces.GPU(duration=90)
def generate(prompt: str, temperature: float, max_new_tokens: int, compare: bool):
    tuned = ""
    for tuned in stream_reply(prompt, temperature, max_new_tokens, use_adapter=True):
        yield tuned, ""
    if compare:
        for base in stream_reply(prompt, temperature, max_new_tokens, use_adapter=False):
            yield tuned, base


with gr.Blocks(title="vamprhyme") as demo:
    gr.Markdown(
        "# vamprhyme\n"
        "Nursery rhymes meet Victorian gothic horror. A LoRA fine-tune of Qwen2.5-1.5B-Instruct, "
        "trained on public-domain nursery rhymes, gothic novels, and rhyme-to-parody pairs.\n\n"
        "Replies appear word by word; on CPU hardware a reply takes about a minute. "
        "The **Gallery** tab shows finished examples instantly."
    )
    with gr.Tab("Try it"):
        prompt = gr.Textbox(label="Prompt", lines=7, value=EXAMPLES[0])
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            max_new_tokens = gr.Slider(40, 300, value=120, step=10, label="Maximum new tokens")
            compare = gr.Checkbox(value=False, label="Also show the base model (twice as slow)")
        button = gr.Button("Generate", variant="primary")
        with gr.Row():
            tuned_out = gr.Textbox(label="Fine-tuned model", lines=10)
            base_out = gr.Textbox(label="Base model (Qwen2.5-1.5B-Instruct)", lines=10)
        gr.Examples([[e] for e in EXAMPLES], inputs=prompt)
        button.click(generate, [prompt, temperature, max_new_tokens, compare], [tuned_out, base_out])
    with gr.Tab("Gallery"):
        gr.Markdown(Path(__file__).with_name("examples.md").read_text(encoding="utf-8"))

demo.queue(default_concurrency_limit=1).launch()
