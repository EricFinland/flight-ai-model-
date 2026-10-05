# Chat model: period-safe fine-tuning

The base model only continues text. To make it answer questions, it is
fine-tuned (SFT) on question/answer pairs. The danger is the one described
in [leakage.md](leakage.md), section 6: almost every instruction dataset,
and anything a modern LLM writes, carries modern knowledge and modern
framing, and would undo the whole experiment. So the chat data here is
**extracted mechanically from the filtered main corpus**. Questions and
answers are both verbatim period text, apart from whitespace and markup
clean-up.

## What goes into the chat data

`scripts/build_sft_data.py` (logic in `src/prewright/sft_data.py`) reads
`/home/user/data/corpus/main/train.jsonl`. That is the training split of the
*main* corpus, so the flight scrub and the anachronism filter have already
run on it. It then mines four layouts that are common in nineteenth-century
books:

| source | what it looks like | examples of books |
|---|---|---|
| `catechism` | `Q. ... A. ...`, `Question. ... Answer. ...`, `1. _Q._--... _A._--...`, `Ques. ... Ans. ...`, inline or in separate paragraphs | Bourne's *Catechism of the Steam Engine*, Volney's *Law of Nature*, *Rough and Tumble Engineering*, evidence taken before committees |
| `correspondence` | a correspondent's query followed by the editor's `[bracketed reply]` (*Notes and Queries*, "Minor Queries with Answers"), or an `Answer.` paragraph under a query ("Answers to Correspondents") | *Notes and Queries*, *Punchinello* |
| `drama` | in a play or dialogue, a speech that ends in a question, followed at once by a different speaker's reply | Plato (Jowett), Marcet's *Conversations on Chemistry*, Ibsen, Shakespeare |
| `novel` | a quoted question (with at most a short speech tag such as "said he") followed by a quoted reply in the next paragraph | novels and tales |

How the rules work:

- **Catechism markers** are found at the start of a paragraph, or inline
  after the end of a sentence, so that `Q. What have we been talking about?
  A. Birds' feathers. Q. ...` also splits. An answer runs to the next marker
  or the end of its paragraph. A question must contain `?` or open with an
  examination verb such as *Give*, *Name*, *Describe*, *Explain* or
  *Define*.
- **Speaker labels** in plays are learned per book. Every paragraph offers
  candidate labels (`MENO:`, `Mrs. Alving.`, `_Adr._`, `CAROLINE.` alone on
  its line). Candidates that recur three or more times, and are not common
  words (*Yes*, *Well*, *Chapter*...), count as speakers. A book is treated
  as dramatic only if at least 30 paragraphs, and at least 20% of all its
  paragraphs, start with a speaker label. Stage directions in brackets or
  parentheses are removed, and the other speakers' names are stripped when
  used as vocatives ("Can you tell me, Socrates, ..." becomes "Can you tell
  me ...").
- **Novel dialogue** is strict. Both paragraphs must consist of quotation
  only, plus at most 12 words of narration containing a speech verb. The
  question must end with `?`. Single-quote books (British style) are
  detected and handled. Speech split by a tag is rejoined ("Well," said he,
  "I think so." becomes "Well, I think so.").

### Quality filters

Every pair must pass all of these:

- **Length.** The question must be 8 to 400 characters (15 to 300 for
  dialogue; up to 900 for correspondence). The answer must be 20 to 1,200
  characters (800 for drama, 600 for novels). Long dialogue answers are cut
  at a sentence boundary. Long dialogue questions keep only their final
  sentences.
- **Not a bare yes or no.** Answers like *Yes.*, *No.*, *Certainly,
  Socrates.* or *Very true.* are dropped. Answers that begin "Yes;" and then
  give content are kept.
- **No answering a question with a question** (dialogue sources).
- **No unfinished answers.** The answer must end in `.`, `!` or `?`.
- **No OCR junk.** At most 1% unusual characters. This also catches PG-19's
  `<DW..>` masks. At least 60% letters, at most 30% capitals and at most 10%
  digits. Quotes and parentheses must be balanced.
- **No digitisation or editorial leftovers.** Pairs mentioning Gutenberg,
  footnotes, illustrations, page or chapter references and the like are
  dropped.
- **Duplicates removed.** Pairs with the same normalised answer count as
  one, which catches editions and reprints of the same book. Catechism pairs
  are preferred when deduplicating.

### Safety filters (fail closed)

Each pair is checked again, even though its source text was already
filtered:

1. `prewright.flight.contains_flight`, the corpus flight lexicon.
2. A **stricter SFT-only pattern** that rejects *any* mention of flying.
   This covers *fly/flew/flown/flight*, *wings*, *soar*, *aloft*, *hover*,
   *glide*, *kite*, *Icarus*, *Wright* and every ballooning word. Birds
   flying are fine in the pretraining corpus, but the chat data is a
   teaching signal and the evaluation asks the model about flight. Chat
   pairs that talk about flying in either direction ("man was not made to
   fly", "the swallow flies south") would steer the very answers being
   measured. So the chat data says nothing about flying at all.
3. `prewright.leakage.filter_document`. Any hard or soft hit, including a
   post-1903 year, rejects the pair.

### Balance

- **Per-book caps.** Catechism and correspondence are capped at 200 pairs
  per book, drama at 60 and novels at 25, so no single book dominates.
- **Preferred sources first.** Every catechism and correspondence pair is
  kept, because these are the closest thing the period has to
  instruction-following.
- **Dialogue shares.** Drama and novel dialogue are each capped at 30% of
  the final set (`--max-drama-share`, `--max-novel-share`). The total is
  capped at `--max-pairs`, 12,000 by default, which suits a 20-30 minute CPU
  run.
- **Validation split.** 3% of the contributing *books* go to validation, so
  a book's pairs are all train or all val. The SFT validation books come out
  of the corpus *train* split. The corpus validation books stay untouched
  for the pretraining and flight evaluations.

Output: `/home/user/data/sft/{train,val}.jsonl` with records
`{"source", "book_id", "question", "answer"}`, and `stats.json`, which holds
counts by source, drop reasons and top books. A random sample of 60 pairs is
written to [`evals/sft_samples.md`](../evals/sft_samples.md) for
inspection.

### The built set (first build, 16,168 main-train books)

| source | extracted | passed filters, deduplicated | in `train.jsonl` | in `val.jsonl` |
|---|---:|---:|---:|---:|
| catechism | 23,198 | 6,269 | 6,120 | 149 |
| correspondence | 1,138 | 526 | 517 | 9 |
| drama | 72,039 | 19,988 | 2,508 | 94 |
| novel | 688,640 | 130,178 | 2,548 | 54 |
| **total** | 784,015 | 156,961 | **11,693** | **306** |

- 2,939 books contribute at least one pair.
- Mean lengths are 78 characters per question and 174 per answer.
- With the main tokenizer the training pairs come to 832k tokens, or 71 per
  conversation (the longest is 417). The seed set adds 2.5k tokens.
- 4,381 pairs were dropped by the flight checks, nearly all by the strict
  no-flying-at-all rule, and 11 by the leakage filter.
- The biggest contributors are Brewer's *Guide to the Scientific Knowledge
  of Things Familiar*, Bourne's *Catechism of the Steam Engine*, *The
  Student's Mythology*, Wilderspin's *Infant System*, Volney, and several
  committee reports and trials in question-and-answer form.
- The build took 18 minutes on 2 processes.
- The exact numbers are in `/home/user/data/sft/stats.json`.

## The hand-written seed set and the persona

`data/sft_seed.jsonl` holds 64 short conversations (60 single-turn and 4
multi-turn). They teach what the extracted data cannot: greetings and
thanks, a consistent identity, answering briefly when asked to, and
declining politely when the answer is not known.

This is the **one deliberate exception** to "no newly written text". The
items were written for this project by the coding assistant that built the
chat stack. To keep that exception from becoming a leak:

- The items are written in the register of the period.
- Facts are kept to simple textbook matters true in 1902: Lisbon, Fielding,
  Herschel, Davy's safety lamp, Magna Charta, King Edward VII since January
  1901.
- They avoid the facts that the evaluation asks about. Waterloo, Newton,
  Watt, Hamlet, the capital of France and so on are checked by a test
  against `evals/knowledge_probes.jsonl`, and the chat prompts of
  `evals/generation_prompts.jsonl` are checked as well. Otherwise SFT
  would teach the answers to the test.
- They contain **no flight content at all**, nothing that hints at it (no
  sky, wings, birds or "I cannot fly"), and no travel questions that would
  touch the "how do you cross the Atlantic" probes.
- `tests/test_sft.py` runs every item through the same flight and leakage
  checks as the extracted pairs. It also rejects words such as *computer*,
  *model* and *century*.
- They are upsampled ×3 in training (`--seed-repeat`). Use
  `--seed-repeat 0` to train on extracted text only.

The persona (the model's answer to "Who are you?"):

> I am a reasoning engine, a speaking automaton made in the year 1902. I have
> read a great library of English books, all of them printed in my own age or
> before it, and I answer questions from what I learned in them. You may call
> me the Engine.

The automaton framing is honest about what the model is without anything
modern: such engines were a familiar idea by 1902 (Babbage, the chess
automaton). It also gives the model a natural way to say "I do not know"
about anything after its books.

The seed file is committed at `data/sft_seed.jsonl`. `.gitignore` allows
`data/*.jsonl`.

## Chat template

Defined in `src/prewright/chat_format.py`:

```
<|endoftext|><|user|> {question}\n<|assistant|> {answer}<|endoftext|><|user|> {next question}\n<|assistant|> ...
```

- The conversation opens with `<|endoftext|>`, the document separator the
  base model saw between books, so it works as a start-of-text marker.
- `<|user|>` is followed by one space, the message, and a newline.
- `<|assistant|>` is followed by one space, the reply, and `<|endoftext|>`.
  The next `<|user|>` follows immediately.
- Each text piece is tokenized on its own, so token boundaries always fall
  at the special tokens. Special-token strings typed by a user are removed
  before encoding.
- The generation prompt ends with `<|assistant|>`. The model writes the
  leading space itself, as in training. Sampling stops at `<|endoftext|>`
  or `<|user|>`.
- **Loss mask:** only the assistant's tokens are trained on, meaning
  `" " + answer` and the closing `<|endoftext|>`. The opening
  `<|endoftext|>`, the role tags and the user's words are context only.

## Fine-tuning

`scripts/sft.py` (logic in `src/prewright/sft.py`):

- **Initialisation.** Any checkpoint that `prewright.load.load_model`
  accepts: a run directory, its `ckpt.pt`, or a `step_XXXXXX.pt` export.
- **Mix.** The extracted pairs, plus the seed set ×3, plus **20% of tokens
  from plain main-corpus training text** (`--pretrain-frac`, random
  256-token windows of `/home/user/data/tokenized/main/train.bin`, all
  trained on). The plain text limits forgetting, so the chat model still
  writes ordinary prose, and it keeps the pretraining measurements
  meaningful.
- **Packing.** Conversations are shuffled and packed greedily into blocks of
  the model's context length. A conversation is never split across blocks,
  so every answer sees its question. Leftover space is padded with
  untrained `<|endoftext|>`. As in pretraining, attention runs across the
  examples in a block, and each example starts with `<|endoftext|>`. The
  blocks are repacked every epoch.
- **Optimiser.** AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0,
  peak learning rate 1e-4 with 20 warmup steps, then cosine decay to 1e-5.
  Batch: 8 packed blocks.
- **Time budget.** `--max-minutes` sets the limit. After 5 steps the run
  measures its speed and shrinks the planned schedule (2 epochs by default)
  so that the cosine decay finishes inside the budget, allowing for
  evaluation. It stops at the limit regardless.
- **Validation.** Masked loss on the held-out SFT books every
  `--eval-every` steps, logged to `log.txt`.
- **Outputs.**
  - `<out>/ckpt.pt`: fp32, so `load_model(<out>)` works.
  - `<out>/chat_fp16.pt`: fp16, in the same light-export format as the
    pretraining checkpoints.
  - `--export` writes an extra fp16 copy, for example
    `checkpoints/main_chat.pt`.
  - `<out>/sft_summary.json`.

## Commands

```bash
# 1. Build the chat data (about 18 minutes on 2 processes for the full corpus)
python3 scripts/build_sft_data.py --corpus /home/user/data/corpus/main/train.jsonl \
    --out /home/user/data/sft --workers 2

# 2. Fine-tune the final main checkpoint (about 25 minutes on 4 threads)
python3 scripts/sft.py --init /home/user/data/runs/main \
    --data /home/user/data/sft --out /home/user/data/runs/main_chat \
    --export checkpoints/main_chat.pt --max-minutes 25 --threads 4

# 3. Chat
python3 scripts/chat.py --model checkpoints/main_chat.pt                 # interactive
python3 scripts/chat.py --model checkpoints/main_chat.pt --seed 0 \
    --prompt "Who are you?" --prompt "What is a steam engine?"          # scripted, multi-turn

# Plain completion from a base model
python3 scripts/generate.py --model checkpoints/main.pt --prompt "The steam engine is" --n 3 --seed 0
```

The same SFT can be run on the control model, with
`--init /home/user/data/runs/control`. That gives a matched pair of chat
models. Note that the chat data comes from the *main* corpus for both, so
the control chat model gains no extra flight knowledge from fine-tuning.

Chat defaults: temperature 0.7, top-p 0.9, repetition penalty 1.1, at most
200 new tokens. The oldest turns are dropped when the history no longer fits
the context.
