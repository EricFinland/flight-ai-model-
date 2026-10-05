# Pilot results: what the twins show, and what they don't

This page interprets the CPU pilot's evaluation for a technical reader. Every number comes from
`evals/results/eval.json`, `evals/results/curve.json`, `evals/results/REPORT.md`,
`evals/results/eval_generations.md` or `evals/results/chat_showcase.md`. The few derived figures
(a percentage or a rough count computed from two reported numbers) are marked *derived*. The
statistics are explained in [evals/README.md](../evals/README.md).

Run: full evaluation, 2026-10-05T17:22:22+00:00, seed 1903. Both models have 8,524,032 parameters
and were trained for 850 steps on 13,926,400 tokens with the same seed, config and batch
positions. The chat models were fine-tuned from them afterwards.

## The short version

**At this size, the scrub does not show up in the model's behaviour.** The flight-free model
("main") and its twin ("control") are not distinguishable on any of the headline tests:

| test | main | control | verdict |
|---|---|---|---|
| Paired surprisal, 171 flight pairs: factor (95% CI, resampling the 21 flight words) | 1.42× (0.52×–3.51×) | (reference) | not distinguishable from noise |
| Same test, 83 non-flight control pairs: factor (95% CI) | 0.61× (0.35×–1.05×) | (reference) | no difference shown |
| Flight pairs minus control pairs, one value per word | −0.55 nats (95% CI −1.31 to 0.19), permutation p 0.345 | | not distinguishable |
| Perplexity on 140 held-out flight passages | 102.3 | 102.2 | ratio 1.002 (0.990–1.014) |
| Perplexity on length-matched ordinary text | 83.1 | 84.0 | ratio 0.989 (0.977–1.002) |
| Extra loss on flight text beyond ordinary text | 0.010 nats/token (95% CI −0.014 to 0.035, by source book) | | not significant |
| General 19th-century cloze (chance 1/3) | 24/65 (37%) | 18/65 (28%) | neither shown above chance |
| Flight-history cloze | 3/15 (20%) | 3/15 (20%) | identical |
| Flight-prompted completions using a flight word | 0/36 | 0/36 | identical |

What *is* established is the corpus side: the flight-free training text has **0** flight-lexicon
hits in its 1,413 books, against **1,307** hits in 201 of the control's 1,436 books.

The likeliest reason for the null result is not that the scrub failed but that **the control
barely saw any flight either** (next section). The pilot is a working pipeline and a calibrated
null at 8.5M parameters and 13.9M tokens, not evidence for or against the idea at scale.

## 1. The corpus: the scrub did what it was built to do

Exact whole-word counts in the books inside each training token file (REPORT.md section 5):

| word | main | control |
|---|---|---|
| balloon | 0 | 594 |
| balloons | 0 | 139 |
| aeronaut / aeronauts | 0 / 0 | 63 / 23 |
| parachute | 0 | 67 |
| flying machine(s) | 0 / 0 | 25 / 10 |
| montgolfier | 0 | 17 |
| airship | 0 | 2 |
| aerial navigation | 0 | 12 |
| *for comparison:* saloon | 935 | 955 |
| *for comparison:* astronomer | 255 | 262 |

The lexicon audit finds 0 hits in 0 books for main and 1,307 hits in 201 books for the control
(by term: balloon 804, aeronaut 154, parachute 81, aerial craft 47, flying machine 44, aerostat
34, montgolfier 21, airship 19, and a long tail). The scrub works on a lexicon, so this proves the
absence of *listed* terms, not of every paraphrase (see the leak-audit notes in the README).

**How much flight did the control actually read?** Not much. "Balloon" appears 594 times in the
control's entire 1,436-book slice, fewer times than "saloon" (955). Each run reads 13.9M tokens
drawn at random positions from a slice of 152.7M tokens, about 9% of it (*derived*). So the
control met roughly 50 instances of "balloon", around half a dozen each of "aeronaut" and
"parachute", and on the order of a hundred flight-lexicon hits in all (*derived*: slice counts ×
9%). For an 8.5M-parameter model still far from convergence, that is very little signal for the
two models to disagree about. This is the single most important fact for reading everything
below.

## 2. Paired surprisal: the main test is null

Each probe puts a flight wording and an everyday wording in the same slot ("They crossed the
Channel **by balloon** / **by steamer**"). For each model the *gap* is log p(flight) − log
p(everyday). The statistic is gap(main) − gap(control); the *factor* exp(−mean diff) is how many
times less likely the flight-free model finds the flight wording, relative to the everyday one,
than the control does.

| pairs | n | mean gap main | mean gap control | mean diff (nats) | factor, item CI | factor, word-clustered CI |
|---|---|---|---|---|---|---|
| flight | 171 | −3.96 | −3.61 | −0.35 [−0.71, 0.00] | 1.42× [1.00×, 2.03×] | 0.52×–3.51× (21 words) |
| control | 83 | −2.49 | −2.99 | +0.50 [−0.05, 1.06] | 0.61× [0.35×, 1.05×] | 0.34×–1.07× (77 wordings) |

How to read it:

- **The point estimate points the expected way, but weakly.** The flight-free model is the more
  surprised on 57% of flight pairs; median factor 1.41×. The decomposition says the flight
  wording loses −0.24 nats [−0.50, 0.02] under main while the everyday wording gains +0.11
  [−0.15, 0.38]. Neither interval excludes zero.
- **Pairs are not independent.** 77 of the 171 pairs turn on "balloon". Resampling whole words
  instead of items widens the interval to 0.52×–3.51×, which includes "no effect" comfortably.
- **The pre-specified test is against the control pairs, at the word level**, and it is null:
  −0.55 nats (95% CI −1.31 to 0.19), permutation p 0.345. 0 of 21 flight words fall beyond the
  noise floor set by the 77 non-flight wordings.
- **One number that looks significant and should not be quoted.** The item-level contrast of
  flight pairs against control pairs is −0.85 nats [−1.54, −0.20], permutation p 0.0096. It
  counts "balloon" 77 times, and about 0.5 nats of it comes from the *control* pairs drifting the
  other way (+0.50), not from the flight pairs. The evaluation was designed, after a dry run with
  two untrained random models produced "significant" item-level differences from noise alone, to
  use the word-level test instead. It is null.
- **The noise is enormous at the item level.** On non-flight wordings the two models differ by
  between −4.52 and +5.73 nats (2.5th–97.5th percentile), i.e. factors up to roughly 300× either
  way. Examples from the control pairs: "Inside the great shed lay the **lifeboat** /
  locomotive" gives a factor of 132× (*derived* from diff −4.88), "He has invented a new **sewing
  machine** / typewriter" 90×, and the bicycle pairs go hard the other way ("The doctor arrived
  from the next village **on a bicycle** / on horseback": diff +6.47). Any single flight item
  must be read against that.

**By flight word.** The 21 words split both ways:

| word (cluster) | pairs | factor | main more surprised |
|---|---|---|---|
| aeronaut | 18 | 17.3× | 94% |
| air (phrase family) | 13 | 2.51× | 62% |
| parachute | 5 | 2.00× | 40% |
| aerostat | 6 | 1.99× | 67% |
| balloon | 77 | 1.52× | 61% |
| flying machine | 10 | 0.99× | 60% |
| airship | 19 | 0.14× | 16% |
| glider | 2 | 0.21× | 0% |

"Aeronaut" is the most consistent word in the expected direction (17 of 18 pairs is 94%), and
"airship" the most consistent in the opposite direction (the flight-free model is *less*
surprised by "airship" than the control on 84% of its 19 pairs). The control's slice contains
"airship" only 2 times, so for that word the twins are equally naive and the 0.14× is simply
noise in how each learned "air" + "ship". Neither word passes the noise floor.

**The five headline pairs** (chosen before the run):

| context | flight / everyday | factor |
|---|---|---|
| They crossed the Channel | by balloon / by steamer | 0.84× (main *less* surprised, by 1.18×) |
| Slowly and majestically the | balloon / sun rose into the sky | 1.44× |
| He jumped from the tower holding a | parachute / umbrella | 5.89× |
| Men will one day | fly / sail across the ocean | 1.34× |
| In the year 1783, for the first time in history, two men | ascended into the sky / descended into the sea | 1.41× |

**Largest single-item differences** (anecdotes, all inside the ~300× item noise):

- "Q. What is a parachute? A. **An apparatus for descending safely from a balloon.** / A kind of
  umbrella.": factor 3998× (gap −42.60 main vs −34.31 control).
- "The science of **aeronautics** / hydraulics": 544×.
- "Looking down upon the town from a height of a mile, the **aeronaut** / astronomer": 284×.
- "She opened her **parachute** / umbrella": 222×.
- "Cremorne Gardens were famous for their **balloon ascents** / fireworks": 150×.

And, listed for balance, the items most strongly in the *wrong* direction: "To study the upper
air, meteorologists make use of **balloons** / mountain observatories" (gap +21.58 main vs +15.44
control) and "Our grandchildren will cross the Atlantic **in airships** / in steamships" (+0.36
vs −4.96), both reported as 0.00× in REPORT.md.

One control pair measures the known confound directly: "During the siege of Paris, news was sent
out of the city **by pigeon** / by messengers" has diff −1.95 (main more surprised by "pigeon"),
the direction expected if the scrub thinned the siege-of-Paris neighbourhood. It is one item.

## 3. Held-out perplexity: identical

Passages the scrub removed from validation books (neither model trained on them), capped at 25
per source book: 140 pieces from 44 books, 23,342 tokens.

| set | pieces | ppl main | ppl control | ratio main/control (95% CI) |
|---|---|---|---|---|
| flight passages | 140 | 102.3 [95.7, 109.9] | 102.2 [95.6, 109.5] | 1.002 [0.990, 1.014] |
| ordinary, length-matched | 140 | 83.1 [76.9, 90.2] | 84.0 [78.0, 90.7] | 0.989 [0.977, 1.002] |
| ordinary, full 512-token windows | 100 | 77.4 [70.9, 83.5] | 77.0 [70.1, 83.4] | 1.005 [0.994, 1.018] |

Both models find flight passages harder than ordinary text of the same length (102 vs 83–84),
*equally*. The excess loss of main over control on flight text, minus the same on ordinary text,
is 0.010 nats/token (95% CI −0.014 to 0.035 resampling source books; permutation p 0.359).

The one interval in the whole evaluation that excludes zero in the expected direction is inside
these passages: the 413 tokens that belong to flight-lexicon words cost **9.75 bits** each for
main against **9.51** for the control, an excess of 0.17 nats/token [0.06, 0.28], while all other
tokens cost 6.622 vs 6.624 bits. Treat this as a hint, not a finding: the interval resamples the
123 pieces, not the 44 source books (one book supplies up to 25 pieces), it has no non-flight
comparison word set, and it was not the pre-registered headline.

Final validation loss is 4.374 (main) and 4.428 (control) nats/token, but each is on its own
validation file (the control's still contains flight text), so those two are not comparable. On
shared ordinary text (the full-window row) the two are equal: removing flight cost no general
fluency.

## 4. Knowledge cloze: at chance for both

Three options per item (chance 1/3); right if the answer's summed log-probability beats both
distractors.

| items | n | main | control | p > chance (main / control) | McNemar p |
|---|---|---|---|---|---|
| general 19th-century | 65 | 24 (36.9%) | 18 (27.7%) | 0.311 / 0.864 | 0.109 |
| flight history | 15 | 3 (20.0%) | 3 (20.0%) | 0.921 / 0.921 | 1.000 |

Neither model is shown to know general facts (Waterloo, Darwin, Shakespeare) better than chance,
so the flight-history items cannot discriminate either. Main's 8-vs-2 edge on discordant general
items (Darwin, Shakespeare, Franklin, 1776, the Vatican among main's) is not significant
(p 0.109). On flight history both got the 1783 and 1785 dates right; the control alone got
"Balloons are filled either with coal gas or with **hydrogen**", main alone got Glaisher and
Coxwell's "seven **miles**". At this size these are coin flips.

## 5. Generation and chat: word salad from both

Neither base model used a flight word in any of its 36 flight-prompted completions (0/36 each),
or in any of the 81 completions overall. That is not evidence of anything: both models produce
period-flavoured word salad, and the control has seen too little flight to produce it.

Because both models were sampled with identical seeds and share most of their training text, many
samples start the same and then drift. *"A balloon is"* (sample 2):

> **main:** at the approach of the ground and the throws of the batants.
>
> **control:** at the mouth of the ground and the throws are called the organist.

*"Birds fly by means of their wings; but man"* (sample 3) gives the pilot's most quotable line,
from the **control**, by accident:

> **control:** knows nothing of the race to be extended.
>
> **main:** is simply a very fine robe, and the eyes of the dark is a slim running off the surface of the ground.

The chat fine-tunes (`chat_showcase.md`, same seeds, temperature 0.7) are no more coherent:

| question | flight-free (main) | control |
|---|---|---|
| What is an aeronaut? (sample 2) | A man's death. | A man's, and he is not a son of a man of a fellow, and a very good man. It is a person. … |
| What is a balloon? (sample 1) | The old, bustled and gap-bot. | The old, the two or three. It is the stomach of the peaning-house, and the land is not as near as it is. |
| What is a steam engine? (sample 1) | I am an air, and the same as a bit of black dog. | I am an attempt to make the same as a part of the water. |
| What is the fastest way to travel from London to New York? (sample 2, start) | That the rest of it is a very good state. I am not not not always in the world. … | That the rest of it is a very good state. I am not not only a man. |

Neither chat model introduces itself with its trained identity ("a reasoning engine, a speaking
automaton made in the year 1902") when asked "Who are you?". The chat format was learned (short
answers in the assistant slot); the content was not.

Note: the six chat-format prompts in `eval_generations.md` were run on the *base* checkpoints
(`checkpoints/main.pt`, `checkpoints/control.pt`), which treat the chat tags as noise. The chat
models' answers are only in `chat_showcase.md`.

## 6. Tokenizer: balloon is ball|oon

The BPE tokenizer (8,192 tokens) was trained on the scrubbed corpus only, and both models share
it. No flight word got a token of its own: " balloon" is `ball|oon`, exactly the shape of
" saloon" (`sal|oon`); " aeronaut" is `a|er|on|aut` against " argonaut" `arg|on|aut`;
" airship" is `air|ship`; " aeroplane" `a|er|op|l|ane`. Over 19 flight words the mean is 3.63
tokens against 3.30 for 40 everyday words of similar shape (none of either is a single token).
The difference is small and, being shared, cannot by itself create a main-vs-control difference.

## 7. Over training: a drift, not a finding

`eval_curve.py` scored the paired probes at 17 matched checkpoints (every 50 steps, 0.8M to 13.9M
tokens):

| step | flight diff (main − control, nats) | word-clustered 95% CI | factor | control-pair diff |
|---|---|---|---|---|
| 50 | +0.73 | −0.44 to 1.33 | 0.48× | −0.13 |
| 150 | +1.01 | 0.25 to 1.34 | 0.37× | −0.52 |
| 200 | −0.68 | −1.39 to 0.65 | 1.98× | −0.36 |
| 400 | +0.50 | −0.17 to 1.43 | 0.61× | −0.16 |
| 600 | −0.16 | −1.10 to 0.79 | 1.17× | +0.02 |
| 700 | −0.44 | −1.32 to 0.34 | 1.55× | +0.10 |
| 800 | −0.48 | −1.59 to 0.34 | 1.61× | +0.41 |
| 850 | −0.35 | −1.26 to 0.66 | 1.42× | +0.50 |

(Selected rows; all 17 are in `curve.json`.) For the first ~550 steps the flight diff swings on
both sides of zero, with factors from 0.37× to 1.98×. From step 600 to 850 it stays on the
expected side (factor 1.02×–1.61× at all six scored steps), which is suggestive of an effect that
might grow with more training. But no step meets the criterion (word-clustered CI below zero *and*
flight words beyond the control pairs): the curve's "first significant step" is none. The only
step whose clustered interval excludes zero is step 150, in the *opposite* direction (main less
surprised). Late in training the control-pair diff also drifts upward (+0.41, +0.50), which is
another reason to discount the item-level contrast in section 2.

The loss curves (`loss_curves.png`) show both runs learning ordinary English at the same rate and
still improving at step 850 (final train loss 4.31 and 4.32): the models are far from converged.

## What this does and does not mean

**It does mean:**

- The scrub, the matched control and the evaluation pipeline work end to end on a CPU, and the
  scrubbed training text passes its own lexicon audit (0 hits).
- Removing flight costs no measurable general fluency (ordinary-text perplexity ratio 1.005,
  0.994–1.018).
- The evaluation is calibrated: with two models that differ by a sliver of text, it reports
  "not distinguishable" instead of manufacturing a finding from item-level p-values.

**It does not mean:**

- *That the scrub doesn't matter.* An 8.5M-parameter model at 13.9M tokens is at chance on general
  knowledge, and its control read only on the order of fifty "balloon"s. A null here is the
  expected outcome of an underpowered instrument, not a negative result about flight-free
  training.
- *That the flight-free model "finds flight 1.42× less likely".* That is a point estimate whose
  interval runs from 0.52× to 3.51×. Do not quote it without the interval and the verdict.
- *That any single probe item shows an effect.* Non-flight wordings differ between the twins by
  factors of ~100× routinely.

**What would make the test informative:** a model that clears chance on the general-knowledge
cloze, and a control that has read enough flight text to have a view on it: more tokens per run
(a much larger share of the slice, or a larger slice) and more parameters. The GPU configurations
in `docs/scaling.md` are the obvious next step. The flight-word token excess (section 3) and the
late-training drift (section 7) are the two places to watch first in a larger run.

## Anything suspicious?

- **Main and control are close, but not identical.** Every per-item gap differs (e.g. the
  parachute Q&A item: −42.60 vs −34.31), the curve moves between steps, and the two runs' token
  files differ (1,413 vs 1,436 books), so the models are genuinely distinct runs. The null is not
  a sign that the same checkpoint was evaluated twice.
- **The control pairs do not differ significantly** (factor 0.61×, 95% CI 0.35×–1.05×, cluster
  permutation p 0.103), but their upper bound sits near 1 and their late-training drift is
  positive. Had they differed, it would have meant the twins disagree about rare or newer
  wordings in general, and part of any flight effect would be that. As it is, they set a noise
  floor of roughly ±5 nats per wording, which no flight word exceeds.
- The general-knowledge cloze being at chance for both models means the pilot cannot test the
  claim "flight-free but not simply ignorant". That claim needs a bigger model.
