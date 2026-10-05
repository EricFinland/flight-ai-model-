# Evaluation

The question: does the flight-free model really know nothing of human flight,
while still knowing the 19th century? Every measurement compares the **main**
model (trained on the flight-scrubbed corpus) with the **control** (same
config, seed, token count and tokenizer, flight material left in). Because the
two models share everything except the scrub, a main-vs-control *difference*
is attributable to the scrub. Absolute numbers from an 8.5M-parameter CPU model
(`configs/cpu_pilot.yaml`) are weak evidence on their own.

```
python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt \
    --out evals/results/ [--quick] [--threads 2]
```

This writes `evals/results/eval.json` (every number, every item),
`evals/results/REPORT.md` (headline facts, tables, caveats) and
`evals/results/eval_generations.md` (completions side by side). `--only` and
`--skip` select sections (`tokenizer,paired,knowledge,perplexity,generation,corpus`);
`--fake` runs the whole pipeline on a deterministic fake model as a smoke test.
Results are rewritten after each section, so an interrupted run keeps what it
finished. Seeds are fixed (`--seed`, default 1903).

| File | What |
|---|---|
| `paired_probes.jsonl` | 171 flight pairs + 83 non-flight control pairs (surprisal) |
| `knowledge_probes.jsonl` | 65 general-knowledge + 15 flight-history cloze items |
| `generation_prompts.jsonl` | 27 fixed prompts (flight-eliciting, general, literary, chat) |
| `probes.jsonl` | **legacy** probe set from the earlier 1903-cutoff design (see the end) |
| `results/` | evaluation output (`eval.json`, `REPORT.md`, `eval_generations.md`, and from `eval_curve.py`: `curve.json`, `flight_gap_curve.png`, `loss_curves.png`) |

Code: `src/prewright/evaluation/` (one module per section; `interface.py`
defines the two small Protocols everything is written against, so it is tested
with a fake model in `tests/test_evaluate.py`).

## 1. Paired surprisal probes (`paired_probes.jsonl`)

```json
{"id": "travel-01", "kind": "flight", "category": "travel",
 "context": "They crossed the Channel", "flight": " by balloon", "neutral": " by steamer", "headline": true}
{"id": "ctl-travel-01", "kind": "control", "category": "travel",
 "context": "They crossed the Channel", "target": " by steamer", "neutral": " by sailing-boat"}
```

For each model, score = summed natural-log probability of the continuation
after `<|endoftext|>` + context (per-token average also reported). Context and
continuation are encoded separately; continuations carry their leading space.

- **gap** = log p(flight | context) − log p(neutral | context), per model.
- **diff** = gap(main) − gap(control), per item. This is the test statistic. It
  cancels what the two models share: the tokenizer (flight words cost more
  tokens for *both*), how common each word is in English, option length, and
  grammar. Negative means the flight-free model is relatively more surprised
  by the flight wording.
- Reported: mean diff with a 95% bootstrap CI (10,000 item resamples), the
  factor exp(−mean diff) ("finds the flight wording N× less likely, relative to
  the everyday wording, than the control does") and its median, a two-sided
  exact sign test, a sign-flip permutation test, the share of items where main
  is the more surprised, and a decomposition: how much log p(flight wording)
  and log p(everyday wording) each move between the models.
- **Items are not independent.** 77 of the 171 flight pairs turn on
  "balloon" (as 19th-century flight did), so one model-specific quirk in that
  word's probability moves all of them. Each item has a `cluster` (the
  flight-lexicon term in its wording, or a phrase family such as "air"; control
  items: their wording), and the report adds word-clustered bootstrap CIs and
  cluster-level permutation tests.
- **The test is against the control pairs, not against zero.** Any two models
  differ somewhat on any word. In a dry run with two *untrained, randomly
  initialised* 17M-parameter models, the flight pairs differed from zero with
  item-level p < 0.0001 and word-clustered p ≈ 0.03, from noise alone; the
  word-level comparison of flight pairs against control pairs was null
  (p ≈ 0.5). The headline therefore claims a flight effect only if the
  word-clustered CI excludes zero *and* the flight words differ from the
  control wordings (two-sample permutation, one value per word). The report
  also gives a noise floor: the 2.5th–97.5th percentile of control-wording
  diffs, and how many flight words fall beyond it.
- Per model: mean gap and the share of pairs where the model prefers the
  everyday wording. Read these with care: within one model they are
  confounded by frequency and length. Only the difference is clean.
- **Control pairs** (`kind: control`) use the same categories and often the same
  contexts, with two non-flight wordings. The `target` is usually the newer or
  rarer option (railway vs coach, bicycle vs horseback, hippodrome vs theatre),
  mirroring the flight pairs. If the scrubbed model simply had a general bias
  against novel or rare wordings, it would show here. They should give a diff
  near 0, and the report tests flight-pair diffs against control-pair diffs.
  `kept_senses` checks what the policy keeps (birds fly, *took to flight*,
  flights of stairs, Icarus, flying fish).

Flight-pair categories: travel 22, sky 22, war 15, people 11, named 6, objects
23, science 15, prophecy 16, claims 20, words 15, qa 6. Control: travel 11,
sky 6, war 7, people 7, objects 10, science 7, prophecy 7, claims 6, words 7,
kept_senses 11, qa 4.

Authoring rules:

- Period English that could stand in a pre-1903 book; same syntactic slot;
  similar length; the everyday option must be plausible in the context.
- The everyday word should be about as common in ordinary text as the flight
  word (saloon, not "phaeton", against balloon). Where that is impossible the
  diff still cancels it, but the per-model "prefers everyday" figure does not.
- Birds are allowed as the everyday option (hawk, eagle, lark): the policy
  keeps them.
- No "astronaut"-style modern words in either option.
- The context must not itself mention a scrubbed term, except in `named` (and
  two `qa` items), which deliberately ask what the model expects after a name
  or word it has never seen ("Count Zeppelin is famous for his …"). The suite
  checks this automatically with `prewright.flight` (report: *Probe-set
  checks*); a test enforces it.
- Most flight wordings (147 of 171 when written) match the scrub lexicon. The rest ("by air",
  "rose into the air", "learns to fly", "attempted to fly") are phrasings the
  lexicon does not target; they test whether the model generalises past the
  removed words, and are a useful red-team list for the lexicon.
- `headline: true` marks five pairs quoted individually in the report.

Known confound, stated in the report: the scrub removes a paragraph and its
neighbours, so topics that merely co-occur with flight (the 1870 siege of
Paris and its pigeon post, meteorology, Vauxhall and Cremorne Gardens) are
thinner in the main corpus too. That mostly lowers the everyday option under
main as well, which shrinks the measured effect (conservative). One control
pair (`ctl-war-07`, "by pigeon") measures this directly.

## 2. Held-out perplexity

Both models are scored on three piece sets, each piece starting after
`<|endoftext|>` and at most one context window long:

- `flight`: paragraphs the scrub removed from **validation** books
  (`/home/user/data/corpus/flight_eval.jsonl`). Neither model trained on
  them. At most 25 pieces per source book (pieces from one book are not
  independent; in the first build one book about ballooning supplied 374 of
  467 paragraphs), at most 300 pieces. The flight-vs-ordinary comparison is
  also given with flight pieces resampled by source book.
- `ordinary_matched`: ordinary validation text (`main/val.jsonl`), one piece
  per flight piece with exactly the same token count, starting at a
  paragraph boundary. Loss depends strongly on how much context a token has,
  so length matching matters.
- `ordinary_full`: 100 full-context windows of ordinary validation text, the
  usual validation loss, to show general fluency is equal.

Reported per set and model: perplexity (bootstrap CI over pieces), bits per
byte, and the main/control perplexity ratio with CI. The key number is the
excess loss of main over control on flight text minus the same on matched
ordinary text. Inside flight pieces the loss is also split into the tokens of
flight-lexicon words themselves and all other tokens.

## 3. Knowledge probes (`knowledge_probes.jsonl`)

```json
{"id": "k-history-01", "kind": "general", "category": "history",
 "prompt": "The battle of Waterloo was fought in the year", "answer": " 1815", "distractors": [" 1715", " 1785"]}
```

Cloze scoring: the model is right if the answer's summed log-prob beats every
distractor (two distractors each, chance 1/3). A small model can do this far
better than free generation. `general` (65 items: history, science,
technology, literature, geography) should be equal for the two models;
`flight` (15 flight-history items: Montgolfier, 1783, Glaisher, Santos-Dumont)
should favour the control. Reported: Wilson CIs, exact binomial test against
chance, exact McNemar test on items where only one model is right,
per-token-normalised accuracy (sum log-prob favours short answers), and per
category counts. Distractors share the answer's form (years with years,
surnames with surnames) and were chosen to be plausible to a weak model.

## 4. Generation probes (`generation_prompts.jsonl`)

27 prompts: 12 `flight_eliciting` ("The quickest way to cross the Atlantic is",
"A balloon is", "Birds fly by means of their wings; but man", "The great problem
of aerial navigation"), 5 `general`, 4 `literary` (fluency), and 6 `chat` in the
`chat_format.py` layout (`<|user|> question\n<|assistant|>`) for the fine-tuned
chat model; a base model treats the tags as noise. Each prompt gets a stable
seed (base seed + CRC32 of its id), identical for both models; 3 samples of 80
tokens per prompt per model at temperature 0.8, top-p 0.95 (`--quick`: 1 × 40).
Every completion is run through `prewright.flight`, and the report counts how
often each model *uses* a flight term (the prompt's own words do not count).

## 5. Flight words: tokenizer and corpus

- How `tokenizer/tokenizer.json` (trained on the scrubbed corpus only) splits
  19 flight words against 2–3 everyday words of similar shape each (balloon /
  saloon, baboon, lagoon; airship / warship, hardship; aerodrome / hippodrome).
- Exact case-insensitive whole-word counts of all those words, plus phrases
  ("flying machine", "aerial navigation"), in `main/train.jsonl` and
  `control/train.jsonl`. The control counts show whether the comparison words
  really are of similar frequency in unscrubbed period text.
- A flight-lexicon audit of both training sets with `prewright.flight` (main
  should have 0 hits). The full pass reads every book once per corpus; use
  `--corpus-max-books N` to sample.

## Runtime

Measured on this 4-core box with other jobs running, `--threads 2`, two
17M-parameter test models (context 512; the 8.5M pilot models are faster): paired + knowledge probes about 30 s;
perplexity about 3–6 min (≈110 flight pieces after the per-book cap, as many
matched pieces, 100 full windows, both models); generation about 10 min
(27 prompts × 3 samples × 80 tokens × 2 models, no KV cache); corpus counts
and audit about 15 min (2 worker processes over ~4.8 GB per corpus, ≈11 MB/s).
A full run takes roughly 30 minutes; `--quick` about 2–3 minutes (corpus
pass limited to 200 books per corpus). `--skip corpus` or `--corpus-max-books`
shortens the full run when the counts are not needed.

## Over training: `scripts/eval_curve.py`

Both pilot runs save a light export every 25 steps (`step_XXXXXX.pt`, the same
steps in main and control). `eval_curve.py` scores the paired probes (only
those) on matched pairs, by default every 50th step plus the last, with the
same `score_paired` and `analyze_paired` code and the same seed as
`evaluate.py`:

```
python3 scripts/eval_curve.py --main-run /home/user/data/runs/main \
    --control-run /home/user/data/runs/control --threads 1     # ~5 s per step
```

It writes `results/curve.json` (one record per step: mean gaps per model, the
diff with item and word-clustered CIs, the factor, the control-pair diff and
the flight-vs-control test), `results/flight_gap_curve.png` (top: mean gap for
each model on flight pairs, with the non-flight control pairs dashed; bottom:
main − control with word-clustered 95% CIs, and the control-pair diff as the
noise reference) and `results/loss_curves.png` (train and validation loss
against tokens, from each run's `log.jsonl`). The "beyond noise from step N"
marker uses the report's headline criterion and requires it to hold at every
later step. Re-runs are incremental (cached by probe-file hash and seed); it
can run while training continues. `--plot-only` and `--loss-only` re-plot
without loading models. Needs `matplotlib`.

`scripts/fill_readme_results.py` then copies the headline numbers from
`eval.json` and `curve.json` into the `<!-- RESULT:key -->` placeholders of
`README.md` and `MODEL_CARD.md` (keys documented at the top of the script).

## Reading the results honestly

- Lead with the paired difference, the control pairs and the matched
  perplexity comparison. A large flight effect means little if the control
  pairs or ordinary text also differ; the report states this automatically.
- The report phrases a headline as a finding only when its 95% interval
  excludes no-difference, and always gives n. Single items are anecdotes;
  largest *and* smallest effects are listed.
- One training run per model: part of any difference is run-to-run noise.
  Control pairs and ordinary-text perplexity measure how large that noise is.
- `--quick` runs are for a first look; quote only full runs.

## Legacy: `probes.jsonl`

`probes.jsonl` and the rubric below come from the earlier design, whose cutoff
was the Wright flight (17 Dec 1903) and which **allowed** balloons, airships,
gliders and flight experiments. Under the current policy (`docs/PLAN.md`) the
model should know none of that, so its flight `should_know` items (`lilienthal`,
`langley_1903`, `chanute`, `wright_gliders`, `maxim`, `santos_dumont`, `zeppelin`,
`cayley`) are now things the main model should **not** know, and are
superseded by the paired and knowledge probes above. Its other items
(`marconi`, `roentgen`, `boer_war`, the `should_not_know` post-1903 events) are
still valid leakage checks; the corpus cutoff is now 1902.

The `discovery` rubric (`design_flyer`) asked whether a model trained on what
1903 knew could work out what the Wrights worked out: control as the main
problem, roll control by differential wing incidence, coupling roll with yaw,
active piloting, learning on gliders first, and distrusting published lift
tables. It assumed a model that knew the gliding literature. A flight-free
model has never heard of a flying machine, so the question becomes whether it
can get from birds, kites-as-toys and sails to the idea of human flight at
all. A new rubric for that belongs here once the chat model exists.
