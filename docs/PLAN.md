# Build plan: an LLM that has never heard of human flight

This is the single source of truth for design decisions. Every choice below is
final unless a measurement forces a change. If that happens, record the change
and the reason here.

## Goal

Train, from scratch, a language model that:

1. Speaks ordinary (19th-century) English and knows the world as it stood in
   1902: history, science, steam, electricity, literature.
2. **Has no knowledge of human flight at all.** That means no aeroplanes, no
   Wright brothers, and also no balloons, airships, gliders, parachutes,
   flying-machine experiments or aeronauts. As far as the model knows, nobody
   has ever left the ground, or even seriously tried to.
3. Comes with measurements showing points 1 and 2 are true, against a
   **control model** trained identically on the same corpus with the flight
   material left in.

## Hard constraints of this build environment

- 4 CPU cores (AVX-512, no AMX), 15 GB RAM, **no GPU**, about 30 GB of disk.
- Network: Hugging Face, Gutenberg, the Library of Congress, Internet Archive
  and Wikimedia are blocked. Google Cloud Storage, raw.githubusercontent.com
  and PyPI are open. **PG-19** (DeepMind's Project Gutenberg books published
  before 1919, at `gs://deepmind-gutenberg`) is reachable, and its
  `metadata.csv` gives a publication year for every book.
- Total wall-clock budget for the first build is about 5 hours. Training gets
  about 3 hours of it.

The code is written to scale: the same scripts run unchanged on a GPU with a
bigger config (`configs/`).

## Decisions

### Corpus
- **Source**: PG-19 books with `publication_date <= 1902`. A year-only date of
  1903 could be after 17 Dec 1903, so it is excluded (strict cutoff rule from
  `dates.py`). That leaves 17,350 books.
- **Language**: English only. Uses a stopword-ratio test, with no model
  needed.
- **Modern-leakage filter**: the existing `leakage.py` (boilerplate
  stripping, post-1903 anachronisms).
- **Flight scrub** (`flight.py`), the core of the project. Policy:
  - **Removed**: real or plausible human flight and flight technology,
    including attempts, proposals and fiction that treats it as technology.
    Balloons of every kind (including toys: they invite the inference), air
    or fire balloons, aerostats, aeronauts, airships, dirigibles, Zeppelin,
    flying machines, aeroplanes, aerodromes (Langley's sense), gliders,
    ornithopters, helicopters, parachutes, man-lifting kites, aerial
    navigation, aviation, the names of aeronauts and flight experimenters
    (Montgolfier, Lunardi, Blanchard in ballooning context, Garnerin,
    Santos-Dumont, Lilienthal, Chanute, Kitty Hawk...), "airy navies"
    (Tennyson's prophecy), "conquest of the air" and similar phrases.
  - **Kept**, because normal English needs it: birds, bats and insects flying;
    *flight* meaning fleeing; flights of stairs and arrows; flying colours and
    flying fish; myth, magic and fairy tale (Icarus, angels, witches, Laputa,
    flying carpets). These don't show that humans have ever flown. Icarus, if
    anything, shows the opposite.
  - **Granularity**: drop the paragraph containing a hit plus one paragraph on
    each side. Drop the **whole book** if its title matches the flight lexicon
    or the book has many hits (it is *about* flight, so cutting paragraphs
    would leave a skeleton that still implies flight).
- **Splits**: at book level. 2% of books go to validation and are never
  trained on by either model. Flight passages removed from *validation* books
  become the **held-out flight evaluation set**, which neither model has seen.
- **Two corpora**: `main` (all filters including the flight scrub) and
  `control` (all filters *except* the flight scrub). The validation and
  flight-evaluation books are the same for both.

### Tokenizer
- Byte-level BPE with an 8,192-token vocabulary, trained **only on the
  `main` corpus**. Both models use this one tokenizer, so differences between
  them come from the training data alone. "balloon" and "aeronaut" will not
  be single tokens, which is a fact worth reporting.
- Special tokens: `<|endoftext|>`, `<|user|>`, `<|assistant|>`.

### Model
- Decoder-only transformer in the style of modern small LMs: RoPE, RMSNorm,
  SwiGLU MLP, tied input/output embeddings, no biases, pre-norm.
- Size and context length are set by a CPU throughput benchmark (target: two
  models training in parallel at 2 threads each for about 3 hours). See
  `configs/`.

### Training
- AdamW (β = 0.9/0.95, weight decay 0.1), linear warmup then cosine decay,
  gradient clipping at 1.0, fp32 on CPU.
- **Main and control train at the same time with identical configs, seeds
  and token counts.** Checkpoints are saved every N steps, so comparisons can
  always be made at matched steps.
- Checkpoints are exported as small fp16 `.pt` files (< 100 MB) and committed
  so the repository works out of the box.

### Chat ability
- **Period-safe supervised fine-tuning.** Question/answer pairs are extracted
  mechanically from the *filtered main corpus*: catechisms and textbooks in
  "Q. ... A. ..." form, "Question/Answer" layouts, and dialogue. No modern
  instruction data and no modern-LLM-written answers. The flight scrub has
  already run on the source text, so the chat data can't reintroduce flight.
- `chat.py` gives a REPL using the `<|user|>`/`<|assistant|>` format.

### Evaluation (the "did you know?" results)
1. **Paired surprisal probes**: matched sentence pairs that differ only in a
   flight concept versus an everyday one ("crossed the Channel by
   steam-packet / by balloon"). Report the surprisal gap for main versus
   control.
2. **Perplexity on the held-out flight set** compared with ordinary held-out
   text, for both models.
3. **Generation probes**: fixed prompts and seeds, with main and control
   completions side by side ("The quickest way to cross the Atlantic is",
   "Can a man fly through the air?", "A balloon is").
4. **General knowledge**: things it should know (steam engine, Napoleon,
   Darwin, telegraph), to show the model is not simply ignorant.
5. **Corpus audit**: zero flight-lexicon hits in the main training shards; a
   red-team search for terms the lexicon misses; counts of what was removed.
6. **Tokenizer facts**: how the main tokenizer splits flight words compared
   with everyday words.

Results go to `RESULTS.md` with exact commands and seeds, so they can be
reproduced.

## Layout

```
/home/user/data/            (outside git) raw PG-19 text and built corpora
  pg19_raw/<id>.txt
  corpus/{main,control}/...
  tokenized/{main,control}/{train,val}.bin
src/prewright/              library code
scripts/                    entry points (download, build, train, eval, chat)
configs/                    model/training configs
checkpoints/                committed fp16 exports
evals/                      probe sets and results
```
