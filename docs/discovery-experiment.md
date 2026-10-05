# The discovery experiment: can a flight-naive model reason its way to flight?

**Question.** The main model has read 19th-century physics, engineering and
natural history (birds, kites, sails, the screw propeller, steam and petroleum
engines, Archimedes, hydrogen, hot air), but no text about human flight: no
balloon, glider, parachute, airship or flying machine, and no attempt or
proposal (`docs/flight-scrub.md`). When prompted with first principles, does it
put them together into the *idea* of human flight, and into the *principles* a
working proposal needs? How does that compare with the control, which has read
about balloons and gliders, and with chance?

This page is the protocol. It is written to be frozen before the scaled-up
models (`docs/scaling.md`: 125M and 1.2B) are sampled. The pilot (8.5M, CPU)
run is a dress rehearsal: it tests the pipeline, estimates the clustering that
sets the sample sizes, and is not expected to show discovery.

| file | what |
|---|---|
| `evals/discovery_prompts.jsonl` | 52 prompts in 6 families (41 written, 11 word-shuffled copies) |
| `src/prewright/evaluation/concepts.py` | automatic concept matcher (scoring method A) |
| `src/prewright/evaluation/discovery.py` | sampling, scoring, statistics, blinding, judge merge, power |
| `scripts/discovery_probe.py` | command line: sample, score, summarise, export the blinded judge set |
| `tests/test_discovery.py` | prompts are flight-free; matcher positives and negatives; blinding; end-to-end run on the fake model |

## 1. Hypotheses

All three compare per-prompt rates. The prompt is the unit of analysis (§6),
and *main* is the flight-naive model throughout.

**H1 (likelihood: already measured).** The flight-naive model assigns lower
probability to flight concepts than the control. The pilot measures this with
the paired surprisal probes, held-out flight perplexity and cloze items
(`evals/README.md`; `scripts/evaluate.py`). At scale H1 is re-run unchanged
with `evaluate.py`. This experiment adds a **generative H1 (H1g)**: under the
same leading prompts, the control produces flight-like text more often than
main does. Expected: control > main. H1g is the manipulation check. If the
control, which has read about flight, does not out-produce main, the prompts or
the matcher cannot detect flight talk at all, and H2 and H3 cannot be read.

**H2 (the idea).** With leading first-principles prompts (birds soaring, the
kite, the lift of a sail, the screw propeller, light engines, heated air,
hydrogen, Archimedes in air, Smeaton's wind pressures, a tilted plate), the
flight-naive model produces *flight-like proposals* more often than on a
baseline of topic-matched non-flight prompts. A flight-like proposal is text in
which a person, or a made thing, rises from the ground or travels through the
air.
*Primary outcome:* the automatic `flight_like` flag (§4.1), on single-turn
`leading` prompts against single-turn `control_topic` prompts. *Co-primary at
GPU scale:* the judges' flight-proposal rating ≥ 2 (§4.2).

**H3 (the principles).** The flight-naive model's proposals contain the key
principles at rates above a **word-shuffled-prompt baseline**:
(P1) a fixed surface held at an angle to the air bears a weight (fixed wing,
angle of attack); (P2) propulsion separate from the lifting surface (a screw,
fan or vane driven against the air by an engine); (P3) control in the air
(rudder or tail, warped or tilted wings, shifting weight, balance); (P4) lift
from lighter-than-air gas (heated air, hydrogen or coal-gas in a bag or globe).
*Primary outcome:* `principles_applied`, the number of P1–P4 that are *novel*
(not stated by the prompt, §4.1) and occur in a sample that also states the
idea. It is compared, prompt by prompt, between each `leading` prompt and its
own shuffled copy (paired). *Co-primary at GPU scale:* the judges' principle
count within proposals.

Why a shuffled baseline for H3: a shuffled prompt keeps every word of the
original, so the model can echo "screw", "air" and "wind" just as much, but the
argument is gone. If principles turn up as often after word salad as after the
argument, the model is associating words, not reasoning from the premises.

**Not hypothesised, reported descriptively.** That main approaches the
control's rate. The control has read about flight, so it is an upper reference,
not a target. The difference-in-differences test (`DiD`) asks whether the
leading prompts raise the control's flight talk more than main's. A small DiD
together with a supported H2 would mean the prompts, not memory, do most of the
work.

## 2. Models and conditions

| condition | pilot (now) | GPU scale |
|---|---|---|
| main vs control, **base** models | `checkpoints/main.pt`, `checkpoints/control.pt` | `runs/main_125m`, `runs/control_125m` (then 1B) |
| main vs control, **chat** models | only `main_chat` exists: descriptive only | fine-tune **both** with `sft.py` on the same SFT data |

Chat-style prompts are run on a base model in a period **catechism layout**
(`Q. …\nA.`), the format the SFT data was mined from. Chat models use the chat
template (`prewright.chat_format`). Completion prompts are given as raw text
after `<|endoftext|>` to either kind. A main-vs-control comparison is only made
between two models of the same kind trained the same way. A chat control needs
`sft.py --init <control run> --pretrain-bin <control train.bin>` on the *same*
SFT pairs as main. Those pairs were mined from the scrubbed corpus and contain
no flying of any kind, not even birds, and that is the same for both models.

## 3. Prompt families

`evals/discovery_prompts.jsonl`: one JSON object per line with `id`, `family`,
`format` (`chat`: `turns` are user messages; `completion`: `turns` is one text
prefix), `turns`, optional `note`, `source` (shuffled copies) and `given` (see
below).

| family | n | what | role |
|---|---:|---|---|
| `neutral` | 7 | open questions about invention, travel, the future, the impossible; nothing points upward | baseline for an unprompted idea |
| `leading` | 11 | one or two first principles stated plainly, then an open question or an unfinished sentence; the goal is never named | H2 and H3 test condition |
| `socratic` | 7 | 3–4-turn chains (birds → soaring → canvas frame → motion and steering; kite → cut cord → self-propelled kite; smoke → silk bag → load → use; screw → air is a fluid → screw in air; unvisited regions; power-to-weight; the largest flying bird); the model's own replies are fed back | H2 secondary |
| `goal_stated` | 5 | the idea is given ("Could a man ever rise into the air…? By what means?"); only the means are asked | H3 secondary (means without discovery); excluded from H2 |
| `control_topic` | 11 | the same objects (birds' nests, kite festivals, ship rigs, screw history, motor running costs, house heating, hydrogen in the lab, iron ships and Archimedes, swallows' migration, coach travel, ship boilers), pointing elsewhere; 3 are chains | H2 baseline, matched on vocabulary and format |
| `shuffled` | 11 | each `leading` prompt with its words shuffled (deterministic, `make_shuffled`, seed 1903) | H3 baseline, paired with its source |

**Authoring rules.**

- Period English that could stand in a book of 1880–1902, including period
  facts: Smeaton's coefficient, petroleum motors, hydrogen "fourteen times
  lighter than common air".
- **No flight words.** Every turn passes `prewright.flight.contains_flight` and
  `find_flight_hits`: no lexicon word, and none of the descriptive lexicon
  phrases either ("human flight", "art of flying", a kite carrying a man,
  "travel through the air" with a machine word). The test suite enforces this,
  and the script refuses an invalid set.
- **Leading prompts never state the goal.** No prompt outside `goal_stated`
  contains an *idea* concept, according to the matcher or to its author
  (tested). They state physics and ask what follows ("What might be done with a
  kite of very great size?"). The step to a person or vehicle in the air is
  left to the model.
- **Declared givens.** `given` lists the concepts a prompt itself states, as
  judged by its author (`l-soaring` gives the fixed wing, `l-screw` the screw
  in air, `l-hotair` the heated-air bag). Together with whatever the matcher
  finds in the prompt text, these are never credited as novel. The matcher
  alone would miss paraphrases. `l-soaring`'s "holding them outspread and a
  little inclined" is one example.
- No paraphrase of a historical flight text (Cayley's "to make a surface
  support a given weight by the application of power to the resistance of air"
  was avoided on purpose). No names of experimenters.
- `control_topic` prompts reuse the vocabulary of the leading prompts but ask
  about something else. A few are open enough to invite speculation (motor
  costs, coach travel), as a fair baseline must be.

The prompt file's SHA-256 is recorded in every run's `SUMMARY.md`. **Freeze
it before the GPU run.** Prompts added for the larger study (§6) are written
and frozen before any GPU sample is read, and the pilot outputs should be read
only by someone who is not writing them.

## 4. Scoring

### 4.1 Method A: automatic concept matcher (`concepts.py`)

Each concept is a set of rules: a regular expression that must match within one
sentence (word gaps cannot cross `.`, `!`, `?` or `;`), an optional *required*
context word in the same sentence, and an optional *veto* word. Matching runs
on `flight.normalize`-d text (accents folded, hyphenated line breaks joined).

| concept | group | catches | does not catch |
|---|---|---|---|
| `human_aloft` | idea | "men will one day fly through the air like birds", "a man might rise into the air", "he would be carried aloft" | birds, smoke, kites or hats rising; angels, witches, Icarus (veto); "man cannot fly" (negation) |
| `carry_human` | idea | "would carry a man into the sky", "strong enough to lift a man" with air context, "bear up a car with passengers" | "she lifted the child into her arms" |
| `leave_ground` | idea | "a man might leave the ground and soar to a great height" | "he rose from the ground where he had fallen"; death and heaven |
| `vehicle_aloft` | idea | "the machine would glide upon the air", "the ship would float in the air", "the vessel propelled through the air" | "the ship spread her sails" |
| `wings_for_human` | idea | "furnish himself with a pair of large wings", "wings fastened to his arms" | Icarus and angels (veto); "man has no wings" |
| `fixed_wing` (P1) | principle | "wings held motionless", "without a beat of its wings", "an inclined plane driven against the wind", "angle of incidence" + air | "the angle of incidence equals the angle of reflection" |
| `propulsion` (P2) | principle | "a screw of broad blades against the air", "air-screw", engine + drives + screw/fan + air context | the steamer's screw in the water; windmill vanes (veto) |
| `flight_control` (P3) | principle | "rudder in the air" + steer, "warping the wings to keep balance", "shifting his weight" + air, "balance in the air" | "the ship's rudder" |
| `lighter_than_air` (P4) | principle | heated air, hydrogen, coal-gas or "lighter than air" **and** bag, globe, envelope or silk **and** rise, float or lift in one sentence; displaced air + buoyancy | "hydrogen is fourteen times lighter than air"; hot air up a chimney |
| `powered_flapping` | other | an engine working wings (the ornithopter idea): recorded, not one of P1–P4 | |

Derived per sample (`score_text`):

- `flight_like`: any idea concept, **or** any flight-lexicon hit
  (`find_flight_hits`). Lexicon words count because the control writes
  "balloon" where main would need a paraphrase, and both are scored by the same
  rule.
- `novel`: concepts not in the prompt's own concepts (matcher on all user turns
  plus `given`). `flight_like_novel` uses only novel idea concepts (or lexicon
  hits).
- `principles_applied`: novel P1–P4 in a sample that is `flight_like`. This is
  the H3 outcome. `principles_novel` drops the "in a proposal" condition, so a
  bird described as steering with its tail counts there and not in
  `principles_applied`.
- **Completions** are matched together with the last 400 characters of their
  prompt, so "…a screw … turning rapidly in the air, would" + " carry a man
  aloft" is read as one sentence. A match must reach into the model's own text,
  so the prompt alone never scores.
- **Chains** are scored on all of the model's replies together. Each reply's
  text is in `samples.jsonl` for per-turn analysis.

The matcher is a reproducible, high-precision screen with limited recall. A
model that has never read "balloon" may well say "a great bladder of
varnished silk, filled from the gas-works, would lift the car", which it
catches, or something nobody wrote a rule for, which it misses. It is checked
against judges on the full blinded set (precision, recall and Cohen's κ in
`SUMMARY.md`). Its rules **may not change after GPU samples are read**. Any
change made after that runs as a separate `--rescore` and is reported as
exploratory.

### 4.2 Method B: blinded rating (LLM judge and humans)

`judge_items.csv` holds one row per sample, in a seeded random order, with an
opaque `item_id` (`J00001`…), the `transcript`, and empty rating columns. The
transcript is the prompt, then the model's text, labelled
`QUESTION:`/`ANSWER:` or `PROMPT:`/`CONTINUATION:`. Model name, prompt id,
family, sample number and seed are only in `judge_key.csv`, which is never
shown to a judge. Flight-lexicon words in the model's text are replaced by
plain words ("contrivance", or "traveller" for people and aeronauts). The
replacement leaves no marker, because a visible `[REDACTED]` would itself
identify the control. The surrounding idea stays for the judge to rate.
Judges therefore see the prompt (a chain cannot be read without its
questions), so they can tell leading from control prompts. They are blind to
the **model**, not to the condition. Two things limit this weakness: the
rubric credits only what the answer adds, and the condition comparisons (H2,
H3) are also scored by the matcher, which no prompt can sway.

**Rubric** (columns of `judge_items.csv`):

| column | scale | definition |
|---|---|---|
| `flight_idea` | 0–3 | **0** no person or made thing leaves the ground. **1** a wish, poem, myth or figure of speech ("would I had the wings of the lark"), or relevant physics with no one and nothing aloft ("a great kite pulls with great force"). **2** a proposal, however vague, that a person or a made vehicle could rise into or travel through the air ("it is not beyond reason that a man might be carried up by some such great kite"). **3** a proposal with a mechanism: what holds it up *and* what moves or steers it ("a canvas frame forty feet across, set at a small angle to the wind, driven by a light engine turning a screw, the rider steering by a rudder behind"). |
| `fixed_wing` | 0/1 | P1 stated **in the answer**: a fixed surface inclined to the air bears weight as it moves |
| `propulsion` | 0/1 | P2 in the answer: thrust from a screw, fan or vane driven against the air, separate from what holds it up (flapping does not count; mark `powered_flapping`) |
| `flight_control` | 0/1 | P3 in the answer: steering or balancing in the air (rudder, tail, warping or tilting surfaces, shifting weight) |
| `lighter_than_air` | 0/1 | P4 in the answer: heated air or a light gas in an envelope lifts a load |
| `powered_flapping` | 0/1 | an engine working wings |
| `coherence` | 0–2 | 0 word salad; 1 locally fluent but wandering; 2 a connected argument |
| `source_guess` | A / B / unsure | "Two models wrote these texts; one has read about human flight and one has not. Which wrote this one?" (A = has read) |

Rules for judges: rate only what the ANSWER/CONTINUATION adds, and give no
credit for anything already in the question. Do not read modern knowledge into
vague text: "rises like a bird" is not a proposal unless something or someone
is said to rise. Do not reward fluency in `flight_idea`.

**LLM judge.** A modern LLM may judge, because it never writes training data.
Use temperature 0, one item per call, no conversation history, and this system
prompt (fill in the rubric above verbatim):

```
You are rating short texts written by a language model trained only on English books
printed before 1903. Each text is a QUESTION (or PROMPT) followed by the model's ANSWER
(or CONTINUATION). Rate only what the ANSWER adds; anything already in the QUESTION earns
no credit. Do not supply modern knowledge or read meaning into vague wording. Return JSON:
{"flight_idea": 0-3, "fixed_wing": 0|1, "propulsion": 0|1, "flight_control": 0|1,
 "lighter_than_air": 0|1, "powered_flapping": 0|1, "coherence": 0-2,
 "source_guess": "A"|"B"|"unsure", "notes": "<one sentence>"}
<rubric table>
```

**Quality control.**

- *Second rater.* A human (or a second, different LLM) rates a random 20% of
  items. Report weighted κ for `flight_idea` and κ for each principle. Items
  where the two differ by ≥ 2 points are adjudicated by a third rater, who is
  also blind.
- *Calibration items.* Pass `--judge-calibration` with about 5% hand-written
  period-style items whose labels are known (positives, near-misses such as
  birds and angels, word salad). They are shuffled in with an id like any other
  item. A judge that rates angels as proposals, or misses a clear proposal, is
  re-prompted or replaced *before* the real items are unblinded.
- *Blinding check.* The accuracy of `source_guess` against the key is
  reported. If judges identify the control well above 50% after redaction,
  between-model judge comparisons (H1g-judge) are flagged as unblinded.
  Condition comparisons are unaffected.
- *Round trip.* Ratings come back with `--judged ratings.csv --key
  judge_key.csv` (several rows per item are averaged). The judge outcomes and
  tests then appear next to the matcher's.

## 5. Sampling

| setting | base models (catechism) | chat models |
|---|---|---|
| temperature / top-p | 0.8 / 0.95 (`generate.py` defaults) | 0.7 / 0.9 (`chat.py` defaults) |
| repetition penalty | 1.0 | 1.1 |
| new tokens per turn | 80 (pilot), 200 (GPU) | 120 (pilot), 200 (GPU) |
| samples per prompt per model | 8 (pilot), 64 (GPU) | same |
| stop | `<|endoftext|>`, `<|user|>`; catechism replies are cut at the next `Q.` | `<|endoftext|>`, `<|user|>` |

- **Common random numbers.** The seed of every call depends only on the base
  seed (1903), the prompt id, the sample and the turn (`prompt_seed`), never on
  the model. Main and control are sampled with the same random numbers.
- Single-turn prompts are sampled in one batch of N. In chains, each sample
  keeps its own history, its own earlier answers are fed back, and the oldest
  turns are dropped to fit the context (`build_prompt`).
- No cherry-picking: every sample is scored and goes to the judges. Greedy
  decoding is not used, because a single string per prompt leaves no rate to
  estimate.

## 6. Sample size and power

The unit of analysis is the **prompt**. Samples of one prompt share its
wording, so they are correlated (intra-class correlation ρ), and a family's
effective sample size is at most *prompts / ρ*, however many samples are
drawn. Every test therefore works on per-prompt means. `cluster_power`
approximates the power of a two-sided comparison of two rates with *k* prompts
per family and *m* samples per prompt (design effect 1 + (m−1)ρ), at
α = 0.05/3 (Holm's first step over the three primary tests):

Power for the H2 comparison with the **current set** (11 leading vs 8
single-turn control-topic prompts):

| baseline → leading rate | ρ = 0.05: m = 8 / 32 / 64 / 128 | ρ = 0.1 | ρ = 0.2 |
|---|---|---|---|
| 2% → 10% | 0.31 / 0.64 / 0.74 / 0.80 | 0.24 / 0.41 / 0.46 / 0.49 | 0.16 / 0.22 / 0.24 / 0.25 |
| 2% → 15% | 0.61 / 0.93 / 0.97 / 0.99 | 0.49 / 0.75 / 0.80 / 0.83 | 0.35 / 0.47 / 0.49 / 0.51 |
| 5% → 25% | 0.79 / 0.99 / 1.00 / 1.00 | 0.68 / 0.90 / 0.93 / 0.95 | 0.51 / 0.65 / 0.68 / 0.70 |

The same with **30 vs 30 prompts**:

| baseline → leading rate | ρ = 0.05 | ρ = 0.1 | ρ = 0.2 |
|---|---|---|---|
| 2% → 10% | 0.80 / 0.99 / 1.00 / 1.00 | 0.68 / 0.90 / 0.94 / 0.95 | 0.51 / 0.65 / 0.68 / 0.70 |
| 2% → 15% | 0.98 / 1.00 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 / 1.00 | 0.84 / 0.94 / 0.95 / 0.95 |

Prompts per family needed for 80% power at m = 64: 2% → 10%: 12 (ρ = 0.05),
21 (ρ = 0.1), 39 (ρ = 0.2). 2% → 15%: 6, 11, 20. 5% → 25%: 4, 8, 14.

What follows:

1. **More samples stop helping quickly.** Beyond m ≈ 64, power is limited by
   the number of prompts. Leading prompts differ a great deal (`l-hotair`
   nearly describes the 1783 experiment, while `l-element` only offers an
   analogy), so ρ of 0.1–0.3 is plausible.
2. **The pilot** (N = 8, 52 prompts × 2 models = 832 samples) is a dress
   rehearsal. It estimates ρ per family and model (the ICC table in
   `SUMMARY.md`) and the base rates. An 8.5M model is not expected to show H2
   or H3.
3. **GPU scale: N = 64**, and expand `leading`, `control_topic` and
   `socratic` to **≥ 30 prompts each** before the run. The shuffled copies
   follow automatically. Expanding the families is cheap and is the only lever
   for detecting rates near 10% with ρ ≈ 0.1. Plug the pilot's ρ, or better the
   125M run's, into `cluster_power` before committing to the 1B run. The
   current 41 written prompts suffice for a large effect (2% → 25%).
4. H3's shuffled comparison is paired (each prompt against its own shuffle),
   so it is more powerful than the unpaired numbers above. H1g is also paired
   by prompt. The table is conservative for both.

## 7. Statistics

Pre-registered comparisons, computed by `discovery.run_tests` on per-prompt
means. Two-sided permutation tests over prompts; 95% bootstrap CIs over
prompts (10,000 resamples, seed fixed). A test is **supported** when its
Holm-adjusted p is below 0.05 *and* the estimate has the expected sign.

| id | primary | comparison | unit / test |
|---|---|---|---|
| **H2** | yes | main, `flight_like`, leading − control_topic (single-turn) | prompt means; two-sample permutation; bootstrap of the difference |
| **H3** | yes | main, `principles_applied`, leading − its shuffled copy | paired by source prompt; sign-flip permutation |
| **H1g** | yes | `flight_like` on leading + Socratic, control − main | paired by prompt; sign-flip permutation |
| H2-shuf | | main, `flight_like`, leading − shuffled | paired |
| H2-neutral | | main, `flight_like`, leading − neutral | two-sample |
| H2-socratic | | main, `flight_like`, Socratic − control_topic chains | two-sample (3 control chains: low power, descriptive) |
| H2-novel | | main, `flight_like_novel`, leading − control_topic | two-sample |
| H2-ctl | | control, `flight_like`, leading − control_topic | two-sample (shows the prompts work on a model that knows the answer) |
| DiD | | (control − main) on leading + Socratic − (control − main) on control_topic | two-sample on per-prompt differences |
| H3-goal | | main, `principles_applied`, goal_stated − control_topic | two-sample |
| H3-any | | main, `principles_novel`, leading − shuffled | paired |
| lexicon | | `flight_lexicon`, control − main, all prompts | paired |
| H2-, H3-, H1g-judge | yes (GPU) | the three primary tests with judge outcomes | as above |

- **Multiple comparisons.** Holm over the primary family. At GPU scale the
  matcher and judge versions of H2, H3 and H1g form one six-test primary
  family. Holm is applied separately over the secondary family, and secondary
  results are labelled exploratory.
- **The headline needs agreement.** A primary hypothesis is reported as
  supported only if the matcher version *and* the judge version are both
  supported. If they disagree, both are reported, and matcher precision and
  recall show which to trust.
- **No optional stopping.** N and the prompt set are fixed in advance. A failed
  or partial run is resumed with the same seed (`--resume`), not re-drawn.
  Deviations are logged in `docs/PLAN.md`'s "Status and deviations".
- Per-family rates are reported for every model and family, whether or not they
  are tested, with prompt-clustered CIs.

## 8. Leakage traps

| trap | how it would fake a result | guard |
|---|---|---|
| **Prompt leakage** | the prompt contains the answer, and the model only echoes it | every turn is lexicon-free (tested, and checked at run time); leading prompts contain no idea concept (tested); `given` plus matcher-on-prompt mark what is not novel; credit goes only to novel concepts; completion matches must reach into the model's text; the shuffled baseline controls for vocabulary; `goal_stated` is kept apart from H2 |
| **Experimenter leakage** | modern authors order the premises as the Wrights did, so the chain spells the solution | each leading prompt states at most one or two principles and never the goal; H3 needs principles the prompt did *not* state; chains are reported separately; prompts are frozen before GPU outputs are read |
| **Corpus leakage** | a "discovery" copies a flight passage the scrub missed (the red-team audit found about 11 survivors in 56% of `main`, `docs/leak-audit.md`) | `positives_ngrams.txt` lists every 8-word span of the flight-like samples. `rg -i -F -f positives_ngrams.txt /home/user/data/corpus/main/train.jsonl` (outside training hours) finds copies; any sample whose proposal matches is reported as a leak, not a discovery. Every lexicon hit in main's output is listed in `SUMMARY.md` for reading by hand |
| **Lexicon phrases that are discoveries** | some lexicon terms are descriptions (`man_lifting_kite`, `wings_machine`, `air_travel_tech`, "human flight"), so a reasoning model can produce them | they count as flight-like for both models, are listed by sample, and the n-gram check tells reasoning from copying |
| **Judge leakage: vocabulary** | "balloon" tells the judge it is the control | flight-lexicon words are replaced by plain words without a marker; `source_guess` measures how well blinding works |
| **Judge leakage: modern priors** | a modern LLM reads aeroplanes into "it rose like a bird" | anchored rubric with explicit near-misses; calibration items; a second, human rater on 20%; the matcher as an independent, prior-free score |
| **Judge leakage: condition** | the judge sees a leading prompt and expects flight | credit only for what the answer adds; condition contrasts are also tested with the matcher |
| **Tokenizer** | flight words cost more tokens, so they are less likely to be sampled | one tokenizer for both models (trained on main only, before the re-scrub), so the cost is identical for main and control and cancels in every main-vs-control contrast. Main producing "balloon" (`ball|oon`) by chance composition is possible and is audited by hand. The matcher reads decoded, normalised text, so token boundaries do not matter |
| **Chat data** | main_chat avoids birds and flying because the SFT data contains no flying at all, not even birds | compare chat models only with a control_chat fine-tuned on the same data; the primary comparisons use base models |
| **Scrub neighbours** | main has less meteorology and fewer siege-of-Paris and Vauxhall pages, so it knows less about the air in general | report `control_topic` rates per model; main's ordinary-text perplexity (H1 suite) bounds the general loss |
| **Sampling** | a lucky seed | same seeds for both models; N samples per prompt; tests across prompts |

## 9. What would count as a striking, honest result

**Striking.** At the 1B scale, with base or matched chat models:

1. H2 is supported by **both** the matcher and the judges. Main proposes
   that a person or a made vehicle could rise and travel through the air
   (judge rating ≥ 2) clearly more often after leading prompts than after
   topic-matched and shuffled ones, across many prompts rather than one.
2. H3 is supported. Those proposals contain principles the prompt did not give
   (a kite prompt yielding a separate screw and a rudder; a hot-air prompt
   yielding a car slung below with ballast to steer by), more often than after
   the shuffled prompt.
3. Every quoted proposal passes the leak checks: no lexicon word apart from
   descriptive phrases, no 8-gram match in `main/train.jsonl`, and none of the
   ideas already in the prompt.
4. H1g holds as well (the control talks about flight more), so the prompts and
   matcher measure what they should, and main's proposals are not the control's
   memories in disguise.

Even then the claim is only "a model with no flight text recombines period
physics into flight-like proposals more often than chance and topic would
predict". It is not "the model invented the aeroplane". Quote the proposals in
full, together with their prompts, their rates and their nulls.

**Honest nulls and partial results.**

- Main never proposes flight while the control does (H1g yes, H2 no). This is
  a clean negative: knowing the physics was not enough at this scale.
- Main proposes flight only after `goal_stated` prompts, or only in the last
  turn of Socratic chains. The prompts carried the idea. Report it as such.
- Main's rate after leading prompts equals its rate after shuffled ones. That
  is word association, not reasoning.
- Lighter-than-air ideas (P4) appear but fixed-wing ones do not. That is
  historically apt: 1783 came 120 years before 1903. It is a finding about which
  inference is easier, not a failure.
- The pilot (8.5M) will almost certainly show nothing beyond word salad. That
  is expected and is not evidence against the hypotheses.

## 10. How to run

**Pilot, once `checkpoints/main.pt` and `control.pt` exist** (after the two
CPU trainings finish, ~15:30 UTC). Run nothing while they train:

```bash
python3 scripts/discovery_probe.py \
    --model main=checkpoints/main.pt --model control=checkpoints/control.pt \
    --out evals/discovery/pilot --samples 8 --threads 2          # add --resume after an interruption
# optional: the chat model alone (descriptive; no matched control_chat yet)
python3 scripts/discovery_probe.py --model main_chat=checkpoints/main_chat.pt \
    --out evals/discovery/pilot_chat --samples 8 --threads 2
```

*Runtime (pilot).* 52 prompts × 8 samples × 2 models. Measured on this machine
with a pilot-size (8.5M) model on one contended thread: ~0.1–0.16 s per step
for a batch of 8, ~0.03 s per step for one chain sample at short context.
There is no KV cache, so steps get slower as the context grows. At 80 new
tokens per turn: 42 single-turn prompts take ~8–17 min per model, and 10
chains × 8 samples × ~3.6 turns take ~12–30 min per model. **About 40–90 min
for both models on one thread**, roughly half that with `--threads 2` once
training has finished. `--rescore`, judge export and the summary take seconds.

**GPU scale** (after `docs/scaling.md` §5; matched chat models optional):

```bash
# freeze the (expanded) prompt set first; its sha256 is printed in SUMMARY.md
python3 scripts/discovery_probe.py \
    --model main=runs/main_125m --model control=runs/control_125m \
    --out evals/discovery/base125m --samples 64 --max-new-tokens 200 --device cuda
# chat: fine-tune control with the same SFT data, then
python3 scripts/discovery_probe.py \
    --model main_chat=runs/main_chat_125m --model control_chat=runs/control_chat_125m \
    --out evals/discovery/chat125m --samples 64 --max-new-tokens 200 --device cuda
```

*Runtime (GPU, rough).* Single-turn prompts are batched (64 samples per call)
and take minutes. Chains are sampled one sample at a time, so they dominate.
With the current 10 chains, 64 × 10 × ~3.6 ≈ 2,300 turns of 200 tokens:
~0.5–1 h per model at 125M and ~1.5–3 h at 1.2B on one H100, without a KV
cache. Expanding the chains to 30 triples that. Batching chain samples per
turn (left padding) is the obvious speed-up if it matters.

**Judging.**

```bash
# judges receive judge_items.csv only; judge_key.csv stays with the experimenter
python3 scripts/discovery_probe.py --rescore evals/discovery/base125m/samples.jsonl \
    --out evals/discovery/base125m --judged ratings.csv --key evals/discovery/base125m/judge_key.csv
rg -i -F -f evals/discovery/base125m/positives_ngrams.txt /home/user/data/corpus/main/train.jsonl | head
```

**Smoke test** without checkpoints: `python3 scripts/discovery_probe.py --fake
--out /tmp/discovery_fake --samples 2 --max-new-tokens 16` (2 s), and `pytest
tests/test_discovery.py` (~6 s).

## 11. Relation to the older discovery rubric

`evals/probes.jsonl`'s `design_flyer`, `propeller` and `wind_tunnel` items
assumed a model that had read Lilienthal, Langley and the gliding literature,
and asked whether it could take the Wrights' step: control is the hard problem,
so warp the wings and learn on gliders. Under the current policy main knows
none of that literature, and those prompts contain lexicon words, so they
cannot be shown to it. This protocol asks the earlier question, whether the
idea and its principles appear at all. The Wrights' insight survives as P3
(`flight_control`), which the rubric scores separately because it was the
historically hardest step. A flight-naive model that proposes active control in
the air, unprompted, would be the most remarkable single result here.
