Outputs from the fine-tuned model on rhymes it **never saw during training**, next to the originals. Generated on a GPU with temperature 0.8.

### There was an old woman tossed in a basket

**Original**

```
There was an old woman tossed in a basket,
Seventeen times as high as the moon;
But where she was going no mortal could tell,
For under her arm she carried a broom.
```

**Fine-tuned**

```
There was an old woman tossed in a coffin,
Seventeen times as low as the grave;
But where she was going no mortal could tell,
For under her arm she carried a broom.
```

### There was an old man in a Marsh (Edward Lear)

**Original**

```
There was an old man in a Marsh,
Whose manners were futile and harsh;
He sate on a log, and sang songs to a frog,
That instructive old man in a Marsh.
```

**Fine-tuned**

```
There was an old stranger in a marsh,
Whose manners were pallid and harsh;
He sat by the graveside at night, and sang waltzes to frogs,
That instructive old stranger in a marsh.
```

### Humpty Dumpty (a Halliwell variant)

**Original**

```
Humpty dumpty sate on a wall,
Humpty dumpty had a great fall;
Three score men and three score more
Cannot place Humpty Dumpty as he was before.
```

**Fine-tuned**

```
Humpty dumpty sat upon the churchyard wall,
Humpty dumpty had a grave pallor;
Three score men and three score more
Came to bury Humpty Dumpty as he was slain.
```

### The winds they did blow

**Original**

```
The winds they did blow,
The leaves they did wag;
Along came a beggar boy,
And put me in his bag.
```

**Fine-tuned**

```
The winds they did howl,
The leaves they did sway;
Along came a pale stranger,
And put me in his bag.
```

### Twinkle, twinkle, little star

Prompt: *Write a gothic horror parody of the nursery rhyme "Twinkle, twinkle, little star".*

```
Twinkle, twinkle, pale stranger,
How strange it is to see you at all;
Your head so large and your face so small,
What have you got to do here, stranger?
```

### The base model, for comparison

Asked for a Dracula version of "There was an old woman tossed in a basket", the untuned Qwen2.5-1.5B-Instruct ignores the rhyme's form and writes a long, generic ballad:

```
In moonlit woods where shadows play,
Lived a creature as fearsome as a thief.
He named himself Vlad, from his homeland of Wallachia,
And he fed on blood like a hungry ape.
...
```

### Known weaknesses

- Dark substitutions often break the original rhyme: "moon / broom" became "grave / broom".
- The phrase "pale stranger" is overused.
- Prompts that name a rhyme without quoting it work only for very famous rhymes, and "Write a Dracula version of Twinkle, Twinkle" still tends to recite the original. Quoting the full rhyme gives the best results.
