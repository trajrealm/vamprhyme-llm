---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- lora
- peft
- creative-writing
- poetry
- nursery-rhymes
- gothic-horror
---

# vamprhyme: Qwen2.5-1.5B-Instruct LoRA

A LoRA adapter for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that rewrites traditional English nursery rhymes as Victorian gothic horror, in the manner of Bram Stoker, Edgar Allan Poe and Sheridan Le Fanu, while keeping each rhyme's shape and opening.

It was built as part of a learning project that went from a character-level GPT written from scratch to fine-tuning a pretrained model. Code: [github.com/trajrealm/vamprhyme-llm](https://github.com/trajrealm/vamprhyme-llm). Demo: [huggingface.co/spaces/trajrealm/vamprhyme-demo](https://huggingface.co/spaces/trajrealm/vamprhyme-demo).

## Example

On a rhyme that was held out of training:

| Original | Fine-tuned |
|---|---|
| There was an old woman tossed in a basket,<br>Seventeen times as high as the moon;<br>But where she was going no mortal could tell,<br>For under her arm she carried a broom. | There was an old woman tossed in a coffin,<br>Seventeen times as low as the grave;<br>But where she was going no mortal could tell,<br>For under her arm she carried a broom. |
| There was an old man in a Marsh,<br>Whose manners were futile and harsh;<br>He sate on a log, and sang songs to a frog,<br>That instructive old man in a Marsh. | There was an old stranger in a marsh,<br>Whose manners were pallid and harsh;<br>He sat by the graveside at night, and sang waltzes to frogs,<br>That instructive old stranger in a marsh. |

Given the same request, the base model writes long, generic Dracula ballads that ignore the original rhyme's form.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "trajrealm/vamprhyme-qwen2.5-1.5b-lora")

messages = [
    {"role": "system", "content": "You are a creative writing assistant. You write poems, nursery rhymes, "
                                  "parodies and fiction in any genre, including gothic horror and dark fairy tales."},
    {"role": "user", "content": "Write a Dracula version of this nursery rhyme:\n\n"
                                "Jack and Jill went up the hill\nTo fetch a pail of water;\n"
                                "Jack fell down and broke his crown,\nAnd Jill came tumbling after."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.9)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use the system prompt above: the adapter was trained with it. Quoting the full rhyme in the request works best. On a CPU, load the model with `dtype=torch.float32` and drop `device_map`.

## Training

**Data**

- Rhyme-to-parody pairs: 12 written by hand and 200 drafted with a larger language model from the hand-written examples, then reviewed. 20 of the drafted pairs were kept for validation. Each pair was presented with one of several request wordings, either quoting the rhyme or naming it.
- 300 examples per style of authentic period text, to keep the voice: nursery rhymes (*The Real Mother Goose*, Halliwell's *Nursery Rhymes of England*, Edward Lear), Victorian horror (*Dracula*, *Carmilla*, *The Vampyre*, Poe, M.R. James, W.W. Jacobs), and darkly comic children's writing (*Alice in Wonderland*, *Struwwelpeter*, Belloc's *Cautionary Tales*, Grimms' Fairy Tales). All source texts are public domain, from Project Gutenberg.
- 30 rhymes were held out of all pairs for testing, along with "Twinkle, Twinkle, Little Star".

**Setup**

| Setting | Value |
|---|---|
| Method | LoRA, rank 16, alpha 32, dropout 0.05 |
| Target modules | q, k, v, o, gate, up and down projections in all 28 layers |
| Trainable parameters | 18.5M (1.18% of the model) |
| Optimizer | AdamW, peak learning rate 2e-4, warmup then cosine decay |
| Batch | 8 (4 x 2 gradient accumulation) |
| Precision | fp16 mixed precision, T4 GPU |
| Checkpoint | Best validation loss, step 150 of 321 (end of the first epoch) |

**Validation loss** (per token) at the saved checkpoint, against the base model:

| | Base | Fine-tuned |
|---|---|---|
| Rhyme pairs | 1.989 | 1.418 |
| Nursery rhymes | 3.260 | 2.685 |
| Gothic horror | 3.447 | 3.061 |
| Children's dark fiction | 3.011 | 2.561 |

## Limitations

- Dark substitutions often break the original's rhyme.
- It overuses a few motifs, especially "pale stranger".
- Asking for a rhyme by name works only for very famous rhymes, and the exact phrase "Dracula version of Twinkle, Twinkle" tends to recite the original. Quoting the full rhyme is reliable.
- A 1.5B model: occasional nonsense lines and weak logic over longer texts.
- English only.

## License

The adapter is released under Apache 2.0, the license of the base model.
