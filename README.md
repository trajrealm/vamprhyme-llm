# vamprhyme-llm

A from-scratch, character-level GPT language model written in PyTorch, built as a hands-on way to learn how large language models work. The project covers the full pipeline: cleaning a raw text corpus, tokenization, a bigram baseline, a transformer (GPT) implementation, training with validation-based checkpointing, multi-style data mixing, and controllable generation, including blending two writing styles at generation time.

The reference corpus blends public-domain nursery rhymes with Victorian gothic horror (for example, "Twinkle, Twinkle" meets *Dracula*), but the pipeline works with any plain-text corpus you provide.

## What it is for

- Learning LLM fundamentals by running, reading and modifying a compact, complete implementation.
- Training small language models on your own text and comparing them.
- Experimenting with model size, learning rate schedules, dropout, data mixing and sampling settings, with loss curves and side-by-side generation to judge the effect.

The models produced here are small base models trained on a few megabytes of text. They learn spelling, line structure and the surface style of each corpus, but not coherent meaning. They continue text rather than follow instructions. See [Limitations](#limitations).

## Requirements

- Python 3.10 or later
- PyTorch, NumPy, Matplotlib
- Optional: an NVIDIA GPU, or a free cloud GPU such as Google Colab, for larger models

Training the default model on a laptop CPU takes about 15 to 40 minutes. The larger configuration in step 10 is intended for a GPU.

## Project structure

| File | Purpose |
|---|---|
| `clean_data.py` | Cleans raw text into tagged prose and verse units |
| `tokenizer.py` | Character-level tokenizer with special tokens |
| `prepare_tokens.py` | Builds the vocabulary and encodes train and validation splits |
| `data.py` | Samples training batches, with weighted mixing across styles |
| `models_bigram.py` | Bigram language model (baseline) |
| `models_gpt.py` | GPT model: causal self-attention, MLP, transformer blocks |
| `trainer.py` | Shared training loop, evaluation, learning rate schedule, checkpointing |
| `train_bigram.py`, `train_gpt.py` | Training entry points |
| `check_gpt.py` | Sanity checks for the GPT before a full training run |
| `sample.py` | Text generation, including style blending |
| `compare_runs.py` | Compares loss histories across runs |
| `play.py` | Interactive prompt loop for comparing models side by side |
| `vamprhyme_colab.ipynb` | Notebook for training on a Google Colab GPU |

Each stage has a detailed walkthrough of its concepts and code: `clean_data.md`, `tokenization.md`, `bigram.md`, `gpt.md`, `training.md`, `mixing.md`, `colab.md` and `play.md`.

The text corpus, encoded token files and training runs are not included in the repository. You supply your own data, and `prepare_tokens.py` regenerates everything under `data/tokens/`, including `tokenizer.json`, from it.

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/vamprhyme-llm.git
cd vamprhyme-llm
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

On an Intel Mac, the last PyTorch release available is 2.2.2, which requires NumPy below version 2:

```bash
pip install torch==2.2.2 "numpy<2" matplotlib
```

### 2. Add your text

Place UTF-8 `.txt` files in three style folders:

```
data/raw/rhymes/   short verse
data/raw/horror/   prose
data/raw/bridge/   material that mixes the two
```

Each folder name becomes a style tag (`<|rhymes|>`, `<|horror|>`, `<|bridge|>`) that the model learns and that you can use to steer generation. A single style is enough to get started, and a folder can be left empty.

The reference corpus uses public-domain books from [Project Gutenberg](https://www.gutenberg.org) (choose "Plain Text UTF-8"):

- **rhymes:** *The Real Mother Goose*; *The Nursery Rhymes of England* (Halliwell); *A Book of Nonsense* and *More Nonsense* (Edward Lear)
- **horror:** *Dracula* (Stoker); *Carmilla* (Le Fanu); *The Vampyre* (Polidori); *The Works of Edgar Allan Poe*, volumes 1 and 2; *Ghost Stories of an Antiquary* (M.R. James); *The Lady of the Barge* (W.W. Jacobs)
- **bridge:** *Alice's Adventures in Wonderland* (Carroll); *Struwwelpeter*; *Cautionary Tales for Children* (Belloc); *Grimms' Fairy Tales*

Notes on your own data:

- Gutenberg header and footer text is removed automatically. Other plain text files work as they are.
- The `rhymes` style keeps **verse only**. Prose paragraphs in that folder are dropped (see `VERSE_ONLY` in `clean_data.py`), so put prose in `horror` or `bridge`.
- To use different style names, update `STYLES` in `clean_data.py`, `prepare_tokens.py` and `play.py`, `VERSE_ONLY` in `clean_data.py`, the `--blend-with` choices in `sample.py`, and the `"rhymes"` stream in `check_gpt.py`.

### 3. Clean the corpus

```bash
python3 clean_data.py
```

This strips boilerplate, illustration and footnote markers, headings, contents pages and indexes, normalizes characters, and splits the text into prose and verse units. Outputs go to `data/clean/`:

- `units.jsonl`: one unit per line, with source, style, kind and text. This is the training input.
- `<style>/<book>.txt`: readable cleaned copies.
- `report.txt`: sizes, character vocabulary, and the first and last units of each book.

Review `report.txt`. If a book still has front or back matter, add start or end phrases for it in `ANCHORS` at the top of `clean_data.py` and run the script again.

### 4. Tokenize

```bash
python3 prepare_tokens.py
```

This builds a character vocabulary from the cleaned text (the reference corpus produces 83 characters plus 4 special tokens), wraps each unit as `<|style|>text<|end|>`, and writes shuffled 90/10 train and validation splits to `data/tokens/` as `<style>_train.npy`, `<style>_val.npy`, and `tokenizer.json`.

If you rebuild the data and the vocabulary size changes, checkpoints trained on the old vocabulary are no longer valid. `play.py` warns when this happens.

### 5. Train the bigram baseline

```bash
python3 train_bigram.py
```

Trains a next-character lookup table in under a minute. It sets the baseline loss that the GPT should beat. Output goes to `runs/bigram/`.

### 6. Check the GPT

```bash
python3 check_gpt.py
```

Verifies the parameter count, output shape, initial loss (about ln(vocab size)), the causal mask, and that the model can memorize a single batch. It also measures time per training step, which you can use to estimate training time. All checks should pass before a full run.

### 7. Train the GPT on one style

```bash
python3 train_gpt.py --steps 5000 --out runs/gpt-5k
```

The default model has 4 layers, 4 heads, a 128-dimensional embedding, a 128-character context and dropout 0.1, about 0.8M parameters. Training uses AdamW with linear warmup, cosine learning rate decay and gradient clipping. The checkpoint with the lowest validation loss is saved as `model.pt`, along with `history.json` and `loss.png`.

Run `python3 train_gpt.py --help` to see all options.

### 8. Train on mixed styles

```bash
python3 train_gpt.py --styles rhymes horror bridge --style-weights 1 1 1 \
    --steps 5000 --eval-iters 50 --out runs/gpt-mix
```

`--style-weights` sets each style's share of every batch, regardless of its size. Without it, styles are sampled in proportion to their size. Validation loss is reported for each style separately, and the best checkpoint is chosen by the weighted average.

### 9. Generate text

```bash
python3 sample.py --checkpoint runs/gpt-mix/model.pt --prompt "<|horror|>" --temperature 0.8
```

To blend two styles, the prompt must start with a style tag:

```bash
python3 sample.py --checkpoint runs/gpt-mix/model.pt \
    --prompt "<|rhymes|>Twinkle, twinkle, little " \
    --blend-with horror --blend-weight 0.6 --temperature 0.8
```

Blending runs the model with both style tags and mixes their next-character log-probabilities: weight 0 is the first style only, 1 is the second only, and values above 1 exaggerate the second.

### 10. Train a larger model on a GPU

```bash
python3 train_gpt.py --styles rhymes horror bridge --style-weights 1 1 1 \
    --n-embd 256 --n-layer 6 --n-head 8 --block-size 256 --dropout 0.2 \
    --steps 5000 --eval-iters 50 --device cuda --out runs/gpt-mix-big
```

This configuration has about 4.8M parameters. `vamprhyme_colab.ipynb` runs it on a free Google Colab GPU. Because the data isn't in the repository, zip `data/tokens/` (`zip -r vamprhyme-tokens.zip data/tokens`) and upload it to Google Drive; the notebook unpacks it after cloning. See `colab.md` for the full setup. Checkpoints trained on a GPU load on a CPU without changes.

### 11. Compare models

```bash
python3 compare_runs.py runs/gpt-5k runs/gpt-mix runs/gpt-mix-big
python3 play.py
```

`compare_runs.py` prints each run's best validation loss, overall and per style, and saves the curves side by side to `runs/compare.png`. `play.py` loads every checkpoint in `runs/` and sends each prompt to all of them. Type `/help` inside it for the available commands.

## Reference results

Validation loss (cross-entropy, lower is better) on the reference corpus:

| Run | Parameters | Training | Rhymes | Horror | Bridge |
|---|---|---|---|---|---|
| `gpt-5k` | 0.8M | rhymes only, CPU | 1.349 | - | - |
| `gpt-mix` | 0.8M | 3 styles, balanced, CPU | 1.411 | 1.470 | 1.269 |
| `gpt-mix-big` | 4.8M | 3 styles, balanced, T4 GPU | 1.202 | 1.328 | 1.114 |

All runs used 5,000 steps. An untrained model starts at about 4.47 (uniform over 87 tokens).

## Limitations

- **Character-level and small.** The models learn spelling, punctuation, line structure and style vocabulary, but produce little coherent meaning and rarely rhyme. Meaning requires far more training text than a few megabytes.
- **Base models.** They continue text and don't follow instructions. Prompt with the beginning of the text you want (`There was an old woman who`), not a request (`write a poem`).
- **Fixed vocabulary.** Prompts can only contain characters that appear in the training data.
