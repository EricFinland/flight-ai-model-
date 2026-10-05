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

## Status and deviations (5 Oct 2026)

What happened in the first build, and where it departs from the decisions
above. Each departure was forced by a measurement or an incident.

- **Data.** PG-19 was the only source reachable (Hugging Face, Gutenberg, the
  Library of Congress and the Internet Archive were blocked by the network
  policy). 17,350 books dated 1902 or earlier were downloaded and the corpora
  built as planned (statistics in `docs/flight-scrub.md`). Other sources are
  listed for later builds in `docs/data-sources.md`.
- **Training slice.** Both token files are capped at a 150M-token slice of the
  ~1.7B-token corpus, chosen by a fixed hash order of book ids. Main's slice is
  1,415 books (150.1M tokens). Control's is the same 1,415 books unscrubbed
  plus the 21 flight books in that hash range that main drops whole (1,436
  books, 152.7M tokens), so control also has the flight paragraphs that main's
  kept books lost. Each run sees about 9% of its slice.
- **850 steps instead of 950.** The plan assumed ~1,600 tokens/s per run. With
  both runs in parallel and other jobs on the machine, the measured rate was
  about 1,150–1,300 tokens/s per run, so both runs use `--max-steps 850`
  (850 × 16,384 = 13.9M tokens) with `--max-hours 3.6` as a backstop. Main and
  control always share the same `max_steps`, so the cosine schedule is the
  same for both.
- **Container restart and a resumable launcher.** A container restart killed
  the first pair of runs. `scripts/launch_cpu_pilot.sh` now resumes any run
  with a `ckpt.pt` (`--resume` continues the exact data order and optimizer
  state) and saves every 25 steps (`ckpt_interval=25`) instead of 50, so a
  restart loses at most a few minutes. Validation loss is still computed every
  50 steps.
- **Red-team audit and re-scrub.** A red-team search of the scrubbed text
  (`docs/leak-audit.md`) found about 11 passages that described or implied
  human flight without a lexicon word, or with a spelling the lexicon missed:
  the *Mongolfier* spelling, Bellamy's *air-car* and Shiel's *air-boat*, "ship
  of the air", Bacon's "suit of wings" and a "pair of wings" for soldiers,
  Rasselas's mechanist ("needs not despair to fly"), "travel through the air"
  by machine or electricity, and Cyrano's machines for reaching the moon. Each
  is now a lexicon term in `flight.py`. Since control is main before the
  scrub, main was re-derived from control with the new lexicon
  (`scripts/rescrub_corpus.py`), without re-downloading. The tokenizer was
  not retrained (it was trained on main before the re-scrub; the extra text
  removed is about 1% of what was already removed). Main's token file is
  rebuilt with `tokenize_corpus.py --train-cutoff <control's cutoff>`, so main
  and control keep exactly the same book set. Main restarts from step 0;
  control, which the lexicon change does not affect, continues.
- **Result file.** Results go to `evals/results/` (`REPORT.md`, `eval.json`,
  generations, curve plots) rather than a top-level `RESULTS.md`, and the
  headline numbers are copied into `README.md` and `MODEL_CARD.md` by
  `scripts/fill_readme_results.py`.
- **Still to run** at the time of writing: both pretraining runs to step 850,
  chat fine-tuning of main, the evaluation (`scripts/evaluate.py`), the curve
  over training (`scripts/eval_curve.py`) and filling in the README.
