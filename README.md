# vamprhyme-llm

A hands-on project for learning how large language models work, built around one creative goal: rewriting traditional nursery rhymes as Victorian gothic horror. It covers the full path from a character-level GPT written from scratch in PyTorch to fine-tuning a pretrained 1.5-billion-parameter model with LoRA, and ends with a public demo on Hugging Face.

```
There was an old woman tossed in a basket,      There was an old woman tossed in a coffin,
Seventeen times as high as the moon;      ->    Seventeen times as low as the grave;
But where she was going no mortal could tell,   But where she was going no mortal could tell,
For under her arm she carried a broom.          For under her arm she carried a broom.
```

*Output of the fine-tuned model on a rhyme held out of training.*

## What it is for

- Learning LLM fundamentals by running, reading and modifying a compact, complete implementation: data cleaning, tokenization, attention, training loops, evaluation and sampling.
- Seeing what scale and pretraining change, by comparing models trained from scratch with a fine-tuned pretrained model on the same task.
- Learning practical fine-tuning: LoRA, chat templates, loss masking, mixed precision, building a supervised dataset with a larger model's help, and publishing the result.
- Training small models on your own text: the pipeline works with any plain-text corpus.

## Try it on Hugging Face

- **Demo:** [huggingface.co/spaces/trajrealm/vamprhyme-demo](https://huggingface.co/spaces/trajrealm/vamprhyme-demo). A Gradio app: enter or pick a rhyme, compare the fine-tuned model with the base model side by side, or browse a gallery of results.
- **Model:** [huggingface.co/trajrealm/vamprhyme-qwen2.5-1.5b-lora](https://huggingface.co/trajrealm/vamprhyme-qwen2.5-1.5b-lora). The LoRA adapter for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). The model card has loading code, training details and limitations.

With this repository, you can also chat with the published adapter directly. It downloads the base model (about 3 GB) on first use and runs on a CPU, slowly, or a GPU:

```bash
pip install transformers peft
python3 chat.py --adapter trajrealm/vamprhyme-qwen2.5-1.5b-lora --compare
```

Quoting the full rhyme in the request gives the best results, for example: `Write a Dracula version of this nursery rhyme:` followed by the rhyme.

## How it was built

**Phase 1: Data.** Public-domain books from Project Gutenberg in three styles: nursery rhymes, gothic horror, and darkly comic children's writing that mixes the two. They're cleaned into tagged prose and verse units.

**Phase 2: A GPT from scratch.** A character-level tokenizer, a bigram baseline, and a GPT (causal self-attention, multi-head attention, transformer blocks), trained with warmup, cosine decay, gradient clipping and best-checkpoint selection, first on a laptop CPU and then on a Colab GPU. Style tags and weighted data mixing let one model write all three styles, and a log-probability blend mixes two styles at generation time. These models learn spelling, form and style vocabulary, but not meaning, which needs far more text than a few megabytes.

**Phase 3: Fine-tuning a pretrained model.** LoRA on Qwen2.5-1.5B-Instruct. Trained on the corpus alone, the model gained an authentic period voice but recited rhymes instead of transforming them. Trained on rhyme-to-parody pairs (12 hand-written seeds, about 200 drafts by a larger model, all reviewed) mixed with style examples, it transforms rhymes it never saw paired, while keeping their shape.

## Project structure

| File | Purpose |
|---|---|
| `clean_data.py` | Cleans raw text into tagged prose and verse units |
| `tokenizer.py`, `prepare_tokens.py` | Character tokenizer; builds the vocabulary and train/validation token arrays |
| `data.py` | Training batches, with weighted mixing across styles |
| `models_bigram.py`, `models_gpt.py` | Bigram baseline and GPT model |
| `trainer.py` | Shared training loop, evaluation, learning rate schedule, checkpointing |
| `train_bigram.py`, `train_gpt.py`, `check_gpt.py` | Training entry points and pre-training sanity checks |
| `sample.py`, `play.py`, `compare_runs.py` | Generation and style blending, interactive model comparison, loss comparison |
| `finetune_data.py` | Turns the corpus into chat-format examples for fine-tuning |
| `select_rhymes.py`, `draft_pairs.py`, `pairs_io.py`, `build_pair_data.py` | Rhyme-to-parody pair dataset: selection, drafting, file format, assembly |
| `pairs/seed_pairs.txt` | The 12 hand-written pairs that define the parody style |
| `finetune_lora.py`, `chat.py` | LoRA fine-tuning; chat with a base model and adapter |
| `vamprhyme_*.ipynb` | Colab notebooks for the GPU steps |
| `huggingface/` | The Hugging Face Space app and model card |

## Getting started

### Requirements

Python 3.10 or later. Phases 1 and 2 need `torch numpy matplotlib`; Phase 3 adds `transformers peft`. Phase 3 training needs a GPU; the notebooks run on Colab's free T4. On an Intel Mac, the newest available PyTorch is 2.2.2, which needs `"numpy<2"`.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/vamprhyme-llm.git
cd vamprhyme-llm
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

### 1. Add your text

Corpus files and training runs aren't in the repository. Place UTF-8 `.txt` files in three style folders; each folder name becomes a style tag the model learns:

```
data/raw/rhymes/   short verse (prose in this folder is dropped)
data/raw/horror/   prose
data/raw/bridge/   material that mixes the two
```

The reference corpus uses Project Gutenberg's plain-text editions of *The Real Mother Goose*, Halliwell's *Nursery Rhymes of England* and Edward Lear (rhymes); *Dracula*, *Carmilla*, *The Vampyre*, Poe, M.R. James and W.W. Jacobs (horror); and *Alice in Wonderland*, *Struwwelpeter*, Belloc's *Cautionary Tales* and Grimms' Fairy Tales (bridge). Gutenberg headers and footers are removed automatically. To use other style names, update `STYLES` in `clean_data.py`, `prepare_tokens.py` and `play.py`, and the style references in `sample.py` and `check_gpt.py`.

### 2. Clean and tokenize

```bash
python3 clean_data.py
python3 prepare_tokens.py
```

Review `data/clean/report.txt` for leftover front or back matter, and add cut points to `ANCHORS` in `clean_data.py` if needed.

### 3. Train from scratch

```bash
python3 train_bigram.py
python3 check_gpt.py
python3 train_gpt.py --styles rhymes horror bridge --style-weights 1 1 1 --steps 5000 --out runs/gpt-mix
```

`check_gpt.py` verifies the model and measures your training speed. Each run saves its best checkpoint, loss history and loss plot under `runs/`. For the larger 4.8M-parameter model, add `--n-embd 256 --n-layer 6 --n-head 8 --block-size 256 --dropout 0.2 --device cuda` on a GPU; `vamprhyme_colab.ipynb` runs it on Colab.

### 4. Generate and compare

```bash
python3 sample.py --checkpoint runs/gpt-mix/model.pt \
    --prompt "<|rhymes|>Twinkle, twinkle, little " --blend-with horror --blend-weight 0.6
python3 compare_runs.py runs/gpt-mix runs/gpt-mix-big
python3 play.py
```

### 5. Fine-tune a pretrained model

Build the datasets locally:

```bash
python3 finetune_data.py
python3 select_rhymes.py
python3 draft_pairs.py --list-free                    # free models on OpenRouter
python3 draft_pairs.py --model MODEL_ID --batch 10    # needs OPENROUTER_API_KEY
python3 build_pair_data.py
```

Review `data/pairs/drafts.txt` before building: fix weak parodies, delete bad ones. `draft_pairs.py --prompts-only` writes prompts to paste into a chat instead of calling an API.

Then train on a GPU, with `vamprhyme_finetune_colab.ipynb` (style examples only) or `vamprhyme_pairs_colab.ipynb` (pairs), or directly:

```bash
python3 finetune_lora.py --data data/sft-pairs --epochs 1 --out runs/lora-pairs
python3 chat.py --adapter runs/lora-pairs/adapter --compare
```

### 6. Publish
Published on Huggingface: `https://huggingface.co/spaces/trajrealm/vamprhyme-demo`


## Results

**From scratch** (validation loss per character; an untrained model starts at 4.47):

| Run | Parameters | Training | Rhymes | Horror | Bridge |
|---|---|---|---|---|---|
| `gpt-5k` | 0.8M | rhymes only, CPU | 1.349 | - | - |
| `gpt-mix` | 0.8M | 3 styles, balanced, CPU | 1.411 | 1.470 | 1.269 |
| `gpt-mix-big` | 4.8M | 3 styles, balanced, T4 GPU | 1.202 | 1.328 | 1.114 |

**Fine-tuned** `Qwen2.5-1.5B-Instruct` with LoRA on pairs (validation loss per token):

| | Base model | Fine-tuned |
|---|---|---|
| Rhyme pairs | 1.989 | 1.418 |
| Nursery rhymes | 3.260 | 2.685 |
| Gothic horror | 3.447 | 3.061 |
| Children's dark fiction | 3.011 | 2.561 |

Per-character and per-token losses aren't directly comparable: a token averages about four characters.

## Limitations

- The fine-tuned model often breaks the original's rhyme when substituting dark words, and overuses a few motifs, such as "pale stranger".
- Requests that name a rhyme without quoting it work only for very famous rhymes.
- The from-scratch models are base models: they continue text rather than follow instructions, and produce surface style without coherent meaning.

## Data and licensing

The source texts are public domain in the United States and aren't included here; check their status in your own country, and follow Project Gutenberg's terms when downloading. The adapter follows the base model's Apache 2.0 license. If you draft pairs with a third-party model, check its terms on using outputs for training.