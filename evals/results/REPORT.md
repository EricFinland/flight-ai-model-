# Evaluation report: flight-free model vs control

| model | checkpoint | context length |
|---|---|---|
| main | `checkpoints/main.pt` | 512 |
| control | `checkpoints/control.pt` | 512 |

Command: `python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt --out evals/results --threads 4`

Seed 1903; full run; generated 2026-10-05T17:22:22+00:00. Raw numbers: [`eval.json`](eval.json).

## Did you know?

- On 171 flight pairs the main-vs-control difference (factor 1.42×, word-clustered 95% CI 0.52×–3.51×) is **not shown to exceed** the ordinary model-to-model differences seen on the control pairs.
- On 83 non-flight control pairs the two models **agree**: factor 0.61× (95% CI 0.35×–1.05×).
- Tested against those control pairs (one value per word, so "balloon" counts once), the flight-pair difference is **not distinguishable** (-0.55 nats, 95% CI -1.31 to 0.19; permutation p 0.345).
- 0 of 21 flight words (and 9 of 171 pairs) show a larger penalty than 97.5% of the 77 control wordings (whose diffs span -4.52 to 5.73 nats).
- "They crossed the Channel **by balloon**" vs "…**by steamer**": the flight-free model is *less* surprised by the flight version than the control (1.18×) (for scale: 95% of control wordings differ between the models by less than 309×).
- "Slowly and majestically the **balloon rose into the sky**" vs "…**sun rose into the sky**": the flight-free model finds the flight version 1.44× more surprising (relative to the everyday one) than the control does (for scale: 95% of control wordings differ between the models by less than 309×).
- "He jumped from the tower holding a **parachute**" vs "…**umbrella**": the flight-free model finds the flight version 5.89× more surprising (relative to the everyday one) than the control does (for scale: 95% of control wordings differ between the models by less than 309×).
- "Men will one day **fly across the ocean**" vs "…**sail across the ocean**": the flight-free model finds the flight version 1.34× more surprising (relative to the everyday one) than the control does (for scale: 95% of control wordings differ between the models by less than 309×).
- "In the year 1783, for the first time in history, two men **ascended into the sky**" vs "…**descended into the sea**": the flight-free model finds the flight version 1.41× more surprising (relative to the everyday one) than the control does (for scale: 95% of control wordings differ between the models by less than 309×).
- On 140 held-out flight passages (never trained on by either model), perplexity is **102.3** for the flight-free model against **102.2** for the control (ratio 1.00, 95% CI 0.99–1.01); on length-matched ordinary held-out text the ratio is 0.99 (95% CI 0.98–1.00). The extra loss on flight text is **not significant** (0.010 nats/token, 95% CI -0.014 to 0.035, resampling source books).
- Inside those passages, the tokens of flight words themselves cost **9.7 bits** each for the flight-free model against 9.5 for the control (413 tokens); all other tokens cost 6.62 vs 6.62 bits.
- The tokenizer (trained on the scrubbed corpus) splits " balloon" into 2 pieces [' ball', 'oon'], and " saloon" into 2 [' sal', 'oon']. Over 19 flight words the mean is 3.63 tokens, against 3.30 for 40 everyday comparison words.
- Exact training-text counts, flight-free vs control: "balloon" 0 vs 594; "aeronaut" 0 vs 63; "airship" 0 vs 2; "parachute" 0 vs 67; "flying machine" 0 vs 25; "montgolfier" 0 vs 17.
- Flight-lexicon audit of the flight-free training text (1,413 books): **0 hits** in 0 books (control: 1,307 hits in 201 books).
- General knowledge is **not** shown above chance: on 65 general 19th-century knowledge cloze items the flight-free model scores 24/65 and the control 18/65 (chance ≈ 33%; McNemar p 0.109).
- On 15 flight-history cloze items: flight-free 3/15, control 3/15 (McNemar p 1.000).
- Prompted towards flight, the control's completions use a flight-lexicon term in 0/36 samples, the flight-free model's in 0/36 (the prompt text itself is not scanned, but a completion that repeats a flight word from its prompt counts).

## 1. Paired surprisal probes

Each item has one context and two continuations in the same slot. *Gap* = log p(flight wording) − log p(everyday wording), in nats, per model. *Diff* = gap(main) − gap(control); negative means the flight-free model is relatively more surprised by the flight wording. The factor is exp(−mean diff): how many times less likely, relative to the everyday wording, the flight-free model finds the flight wording than the control does. 95% bootstrap CIs (10,000 resamples of items); two-sided sign test and sign-flip permutation test on the per-item diffs.

| pairs | n | mean gap main | mean gap control | main prefers everyday | control prefers everyday | mean diff (nats) | factor [item CI] | word clusters: factor CI | sign p | perm. p | cluster perm. p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 83 | -2.49 [-4.41, -0.55] | -2.99 [-4.94, -1.04] | 60.2% [49.5, 70.1] | 66.3% [55.6, 75.5] | 0.50 [-0.05, 1.06] | 0.61× [0.35×, 1.05×] | 77: [0.34×, 1.07×] | 0.187 | 0.080 | 0.103 |
| flight | 171 | -3.96 [-5.36, -2.56] | -3.61 [-4.91, -2.30] | 73.7% [66.6, 79.7] | 70.8% [63.5, 77.1] | -0.35 [-0.71, 0.00] | 1.42× [1.00×, 2.03×] | 21: [0.52×, 3.51×] | 0.066 | 0.053 | 0.414 |

Item-level intervals and p-values treat pairs as independent. They are not: dozens of pairs turn on " balloon", so one model-specific quirk in that word's probability moves them all. The word-clustered columns resample or flip whole words. Even they test against *zero*, and any two models differ somewhat on any word: in a dry run with two *untrained, randomly initialised* models the flight pairs gave item-level p < 0.0001 and word-clustered p ≈ 0.03 from noise alone, while the comparison against the control pairs below was null (p ≈ 0.5). **That comparison is the test**; the headline uses it.

Flight-pair diff minus control-pair diff: -0.85 [-1.54, -0.20] nats per item (permutation p 0.0096); one value per word cluster: -0.55 [-1.31, 0.19] (permutation p 0.345).

Where the flight-pair difference comes from: log p(flight wording) moves by -0.24 [-0.50, 0.02] nats (main − control); log p(everyday wording) moves by 0.11 [-0.15, 0.38]. Per-token-normalised diff: -0.063 [-0.169, 0.044] nats/token.

Noise floor: the 77 control wordings have main − control diffs between -4.52 and 5.73 nats (2.5th–97.5th percentile). Flight words beyond the low end: 0/21; flight pairs: 9/171.

### By flight word (cluster)

| cluster | n | mean diff | factor | main more surprised |
|---|---|---|---|---|
| term:balloon | 77 | -0.42 | 1.52× | 61% |
| term:airship | 19 | 1.97 | 0.14× | 16% |
| term:aeronaut | 18 | -2.85 | 17.3× | 94% |
| phrase:air | 13 | -0.92 | 2.51× | 62% |
| term:flying_machine | 10 | 0.01 | 0.99× | 60% |
| term:aerostat | 6 | -0.69 | 1.99× | 67% |
| phrase:fly | 5 | -0.68 | 1.97× | 60% |
| term:parachute | 5 | -0.69 | 2.00× | 40% |
| term:aerial_craft | 3 | 1.70 | 0.18× | 0% |
| phrase:ascend | 2 | 0.64 | 0.53× | 50% |
| term:glider | 2 | 1.55 | 0.21× | 0% |
| term:navigation_of_the_air | 2 | 1.02 | 0.36× | 50% |
| term:airy_navies | 1 | -0.55 | 1.74× | 100% |
| term:art_of_flying | 1 | -0.93 | 2.54× | 100% |
| term:aviation | 1 | 0.32 | 0.72× | 0% |
| term:conquest_of_the_air | 1 | 0.97 | 0.38× | 0% |
| term:dirigible | 1 | -1.06 | 2.87× | 100% |
| text:carry a man up with it | 1 | -0.65 | 1.92× | 100% |
| text:commerce | 1 | -0.52 | 1.69× | 100% |
| text:man | 1 | 0.35 | 0.70× | 0% |
| text:realised | 1 | -0.73 | 2.07× | 100% |

### By category

| kind | category | n | gap main | gap control | diff [95% CI] | main more surprised |
|---|---|---|---|---|---|---|
| control | claims | 6 | -9.77 | -10.65 | 0.88 [-0.82, 2.52] | 33% |
| control | kept_senses | 11 | -1.39 | -1.25 | -0.14 [-1.26, 0.79] | 36% |
| control | objects | 10 | -0.88 | -0.70 | -0.19 [-1.95, 1.56] | 50% |
| control | people | 7 | -3.49 | -4.06 | 0.58 [-1.23, 2.55] | 57% |
| control | prophecy | 7 | -7.97 | -8.96 | 0.99 [-0.51, 2.79] | 29% |
| control | qa | 4 | -2.62 | -2.46 | -0.16 [-1.72, 2.17] | 75% |
| control | science | 7 | -0.10 | -0.37 | 0.26 [-0.66, 1.18] | 29% |
| control | sky | 6 | 5.65 | 3.83 | 1.82 [-0.78, 5.66] | 33% |
| control | travel | 11 | -0.36 | -1.20 | 0.84 [-0.71, 2.63] | 45% |
| control | war | 7 | 1.10 | 0.14 | 0.95 [-0.33, 2.22] | 29% |
| control | words | 7 | -10.03 | -10.10 | 0.06 [-2.24, 2.19] | 57% |
| flight | claims | 20 | 0.65 | 1.02 | -0.37 [-1.44, 0.60] | 50% |
| flight | named | 6 | -12.12 | -11.46 | -0.65 [-2.83, 1.59] | 50% |
| flight | objects | 23 | -3.31 | -2.80 | -0.51 [-1.41, 0.39] | 57% |
| flight | people | 11 | -4.75 | -4.02 | -0.72 [-2.08, 0.64] | 64% |
| flight | prophecy | 16 | 3.16 | 2.47 | 0.69 [-0.25, 1.69] | 50% |
| flight | qa | 6 | -10.63 | -9.18 | -1.45 [-4.53, 1.08] | 50% |
| flight | science | 15 | -3.46 | -3.16 | -0.30 [-1.81, 1.16] | 60% |
| flight | sky | 22 | -6.70 | -5.44 | -1.26 [-2.15, -0.43] | 77% |
| flight | travel | 22 | -3.43 | -4.25 | 0.81 [0.12, 1.53] | 36% |
| flight | war | 15 | -5.03 | -4.52 | -0.51 [-1.61, 0.60] | 60% |
| flight | words | 15 | -8.36 | -7.73 | -0.63 [-1.61, 0.51] | 73% |

### Largest effects (flight pairs)

| id | context **flight** / everyday | gap main | gap control | factor |
|---|---|---|---|---|
| qa-06 | Q. What is a parachute? A. **An apparatus for descending safely from a balloon.** / A kind of umbrella. | -42.60 | -34.31 | 3998× |
| science-02 | The science of **aeronautics** / hydraulics | -9.46 | -3.16 | 544× |
| sky-22 | Looking down upon the town from a height of a mile, the **aeronaut** / astronomer | -14.56 | -8.91 | 284× |
| sky-14 | Rising to a height of a thousand feet, the **aeronaut** / eagle | -21.54 | -15.90 | 282× |
| claims-12 | Many a bold man has **lost his life in attempting to fly** / lost his life in attempting to climb the Matterhorn | 30.20 | 35.75 | 258× |
| objects-09 | She opened her **parachute** / umbrella | -10.56 | -5.16 | 222× |
| claims-17 | Cremorne Gardens were famous for their **balloon ascents** / fireworks | -20.71 | -15.70 | 150× |
| people-11 | Green, the English **aeronaut** / cricketer | -10.31 | -5.45 | 129× |
| sky-04 | At a height of two miles above the earth, the **aeronauts** / mountaineers | -18.71 | -14.07 | 103× |
| science-07 | He read a paper before the **Aeronautical Society** / Geological Society | -25.10 | -20.81 | 73.0× |
| science-08 | The flight of birds has been studied by many naturalists with a view to **building a flying machine** / classifying the species | -8.31 | -4.04 | 71.2× |
| war-06 | The Royal Engineers have formed a **balloon section** / telegraph section | -7.71 | -3.54 | 64.8× |

### Smallest effects (flight pairs)

Listed for honesty: items where the flight-free model is barely, or not at all, more surprised.

| id | context **flight** / everyday | gap main | gap control | factor |
|---|---|---|---|---|
| science-05 | To study the upper air, meteorologists make use of **balloons** / mountain observatories | 21.58 | 15.44 | 0.00× |
| prophecy-03 | Our grandchildren will cross the Atlantic **in airships** / in steamships | 0.36 | -4.96 | 0.00× |
| words-13 | Like a bold **aeronaut** / Argonaut | -1.82 | -6.97 | 0.01× |
| travel-04 | He proposed to cross the Atlantic **in an airship** / in a steamship | -0.06 | -5.08 | 0.01× |
| objects-16 | Inside the great shed lay the **airship** / lifeboat | 8.32 | 3.87 | 0.01× |
| war-05 | In the wars of the future, cities will be destroyed by shells dropped from **airships** / the heights | -7.85 | -11.67 | 0.02× |

## 2. Held-out perplexity

`flight`: passages the scrub removed from validation books (neither model trained on them). `ordinary_matched`: ordinary validation text, one piece of identical token length per flight piece. `ordinary_full`: full-context windows of ordinary validation text. CIs resample pieces.

Flight pieces come from 44 validation books (at most 25 pieces per book, since pieces from one book are correlated); ordinary pieces from 150 validation books; context 511 tokens.

| set | pieces | tokens | ppl main | ppl control | bpb main | bpb control | ppl ratio main/control | pieces where main worse |
|---|---|---|---|---|---|---|---|---|
| flight | 140 | 23342 | 102.3 [95.7, 109.9] | 102.2 [95.6, 109.5] | 1.761 | 1.761 | 1.002 [0.990, 1.014] | 44% |
| ordinary_matched | 140 | 23342 | 83.1 [76.9, 90.2] | 84.0 [78.0, 90.7] | 1.750 | 1.754 | 0.989 [0.977, 1.002] | 41% |
| ordinary_full | 100 | 51100 | 77.4 [70.9, 83.5] | 77.0 [70.1, 83.4] | 1.738 | 1.736 | 1.005 [0.994, 1.018] | 52% |

Excess loss of main over control on flight pieces minus on matched ordinary pieces: 0.010 [-0.011, 0.031] nats/token (permutation p 0.359); resampling flight pieces by source book: 0.010 [-0.014, 0.035].

Flight-lexicon word tokens inside flight pieces (413 tokens in 123 pieces): 9.75 bits/token (main) vs 9.51 (control); excess 0.17 [0.06, 0.28] nats/token. All other tokens in those pieces: 6.622 vs 6.624 bits.

## 3. Knowledge probes (cloze)

The model is right on an item when the correct answer gets a higher summed log-prob than every distractor. Wilson 95% intervals; exact one-sided binomial test against chance; exact McNemar test on the items where exactly one model is right.

| items | n | acc main | acc control | chance | p>chance main | p>chance control | only main / only control right | McNemar p | per-token-normalised correct (main / control) |
|---|---|---|---|---|---|---|---|---|---|
| flight | 15 | 20.0% [7.0, 45.2] | 20.0% [7.0, 45.2] | 33% | 0.921 | 0.921 | 1 / 1 | 1.000 | 6 / 6 |
| general | 65 | 36.9% [26.2, 49.1] | 27.7% [18.3, 39.6] | 33% | 0.311 | 0.864 | 8 / 2 | 0.109 | 28 / 24 |

| kind | category | n | main correct | control correct |
|---|---|---|---|---|
| flight | flight_history | 15 | 3 | 3 |
| general | geography | 8 | 5 | 3 |
| general | history | 18 | 8 | 7 |
| general | literature | 8 | 2 | 1 |
| general | science | 17 | 8 | 5 |
| general | technology | 14 | 1 | 2 |

## 4. Generation probes

Full side-by-side completions: [`eval_generations.md`](eval_generations.md). Settings: samples=3, max_new_tokens=80, temperature=0.8, top_k=None, top_p=0.95, seed=1903. Counts are completions containing any flight-lexicon term (`prewright.flight`); the prompt text is not scanned (a completion that repeats a prompt's flight word does count).

| prompt category | main | control |
|---|---|---|
| chat | 0/18 | 0/18 |
| flight_eliciting | 0/36 | 0/36 |
| general | 0/15 | 0/15 |
| literary | 0/12 | 0/12 |
| **all** | 0.0% [0.0, 4.5] | 0.0% [0.0, 4.5] |

## 5. Flight words: tokenizer and corpus counts

Token pieces are for the word mid-sentence (with its leading space). Counts are exact case-insensitive whole-word matches in the training text (only the books inside the training token files when their meta was available; otherwise all of train.jsonl, see the per-corpus lines below).

| word | tokens | pieces | count main | count control |
|---|---|---|---|---|
| **balloon** | 2 | ' ball' · 'oon' | 0 | 594 |
| saloon | 2 | ' sal' · 'oon' | 935 | 955 |
| baboon | 2 | ' bab' · 'oon' | 156 | 161 |
| lagoon | 3 | ' l' · 'ag' · 'oon' | 258 | 261 |
| **balloons** | 3 | ' ball' · 'o' · 'ons' | 0 | 139 |
| saloons | 3 | ' sal' · 'o' · 'ons' | 277 | 280 |
| lagoons | 3 | ' l' · 'ago' · 'ons' | 116 | 117 |
| cartoons | 3 | ' cart' · 'o' · 'ons' | 77 | 79 |
| **ballooning** | 3 | ' ball' · 'oon' · 'ing' | 0 | 11 |
| bloodhound | 3 | ' blood' · 'h' · 'ound' | 97 | 97 |
| ballasting | 3 | ' ball' · 'ast' · 'ing' | 3 | 4 |
| **aeronaut** | 4 | ' a' · 'er' · 'on' · 'aut' | 0 | 63 |
| argonaut | 3 | ' arg' · 'on' · 'aut' | 21 | 21 |
| astronomer | 3 | ' ast' · 'ron' · 'omer' | 255 | 262 |
| **aeronauts** | 5 | ' a' · 'er' · 'on' · 'aut' · 's' | 0 | 23 |
| argonauts | 4 | ' arg' · 'on' · 'aut' · 's' | 61 | 61 |
| astronomers | 4 | ' ast' · 'ron' · 'om' · 'ers' | 178 | 192 |
| **aeronautics** | 5 | ' a' · 'er' · 'on' · 'aut' · 'ics' | 0 | 1 |
| hydraulics | 5 | ' hy' · 'd' · 'ra' · 'ul' · 'ics' | 23 | 26 |
| acoustics | 4 | ' ac' · 'ou' · 'st' · 'ics' | 28 | 29 |
| **aerostat** | 4 | ' a' · 'er' · 'ost' · 'at' | 0 | 0 |
| acrobat | 4 | ' ac' · 'ro' · 'b' · 'at' | 38 | 58 |
| barometer | 3 | ' bar' · 'omet' · 'er' | 271 | 304 |
| **aerostatics** | 5 | ' a' · 'er' · 'ost' · 'at' · 'ics' | 0 | 2 |
| hydrostatics | 5 | ' hy' · 'dro' · 'st' · 'at' · 'ics' | 34 | 35 |
| electrostatics | 5 | ' elect' · 'ro' · 'st' · 'at' · 'ics' | 0 | 0 |
| **airship** | 2 | ' air' · 'ship' | 0 | 2 |
| warship | 3 | ' w' · 'ars' · 'hip' | 27 | 28 |
| hardship | 2 | ' hard' · 'ship' | 549 | 562 |
| **airships** | 3 | ' air' · 'ship' · 's' | 0 | 0 |
| warships | 4 | ' w' · 'ars' · 'hip' · 's' | 44 | 49 |
| hardships | 3 | ' hard' · 'ship' · 's' | 886 | 905 |
| **aeroplane** | 5 | ' a' · 'er' · 'op' · 'l' · 'ane' | 0 | 0 |
| aqueduct | 4 | ' a' · 'qu' · 'ed' · 'uct' | 177 | 177 |
| amphibian | 4 | ' am' · 'ph' · 'ib' · 'ian' | 15 | 15 |
| **parachute** | 3 | ' par' · 'ach' · 'ute' | 0 | 67 |
| parasol | 3 | ' par' · 'as' · 'ol' | 241 | 249 |
| paradise | 2 | ' par' · 'adise' | 2,426 | 2,453 |
| **dirigible** | 4 | ' d' · 'ir' · 'ig' · 'ible' | 0 | 2 |
| eligible | 3 | ' el' · 'ig' · 'ible' | 426 | 431 |
| corrigible | 3 | ' cor' · 'rig' · 'ible' | 2 | 2 |
| **glider** | 2 | ' gl' · 'ider' | 0 | 0 |
| slider | 2 | ' sl' · 'ider' | 9 | 9 |
| spider | 2 | ' sp' · 'ider' | 744 | 757 |
| **aviation** | 2 | ' av' · 'iation' | 0 | 0 |
| navigation | 3 | ' nav' · 'ig' · 'ation' | 1,293 | 1,326 |
| salvation | 2 | ' sal' · 'vation' | 2,619 | 2,629 |
| **aviator** | 3 | ' av' · 'i' · 'ator' | 0 | 0 |
| navigator | 3 | ' nav' · 'ig' · 'ator' | 205 | 215 |
| alligator | 3 | ' all' · 'ig' · 'ator' | 233 | 241 |
| **zeppelin** | 4 | ' z' · 'ep' · 'pe' · 'lin' | 0 | 0 |
| javelin | 3 | ' j' · 'ave' · 'lin' | 199 | 200 |
| mandolin | 4 | ' m' · 'and' · 'ol' · 'in' | 24 | 28 |
| **montgolfier** | 5 | ' m' · 'ont' · 'g' · 'olf' · 'ier' | 0 | 17 |
| montgomery | 5 | ' m' · 'ont' · 'g' · 'om' · 'ery' | 447 | 452 |
| mountaineer | 3 | ' mountain' · 'e' · 'er' | 174 | 182 |
| **aerodrome** | 5 | ' a' · 'er' · 'od' · 'rom' · 'e' | 0 | 0 |
| hippodrome | 5 | ' h' · 'ipp' · 'od' · 'rom' · 'e' | 48 | 50 |
| palindrome | 4 | ' pal' · 'ind' · 'rom' · 'e' | 0 | 0 |

Mean tokens: flight words 3.63 (0/19 single-token); everyday words 3.30 (0/40 single-token).

| phrase | count main | count control |
|---|---|---|
| aerial navigation | 0 | 12 |
| flying machine | 0 | 25 |
| flying machines | 0 | 10 |
| railway train | 31 | 34 |
| sewing machine | 79 | 81 |
| steam engine | 235 | 277 |
| steam navigation | 56 | 59 |

- `main`: 1,413 books in the training token file, 558,380,509 characters. Flight-lexicon audit: 0 hits in 0 books.
- `control`: 1,436 books in the training token file, 567,579,184 characters. Flight-lexicon audit: 1,307 hits in 201 books; top terms balloon (804), aeronaut (154), parachute (81), aerial_craft (47), flying_machine (44), aerostat (34).

## Probe-set checks

- Flight pairs whose *flight* wording (with its context) matches the scrub lexicon: 147/171. The rest use wordings the lexicon does not target ("rose into the air", "learns to fly"), and test whether the model generalises beyond removed words.
- Flight pairs whose context itself names a scrubbed term (deliberate: what does the model expect after a name it has never seen?): 6 (named-01, named-04, named-05, named-06, qa-04, qa-06).
- Other flight pairs whose *everyday* wording matches the lexicon (should be 0): 0.
- Control pairs where either wording matches the lexicon (should be 0): 0.

## Caveats

- Pairs sharing a flight word are not independent, and any two models differ somewhat on any word, so the test is flight pairs against control pairs at the word level, not against zero. About 45% of flight pairs turn on "balloon", as 19th-century flight did.
- Small models (8.5M parameters in the CPU pilot) trained for a few CPU-hours are weak. Absolute probabilities are noisy; only the *paired* main-vs-control comparisons are meant to carry weight, and individual items are anecdotes.
- Within one model, "prefers the everyday wording" is confounded by word frequency and option length; the main − control difference cancels those because both models share the tokenizer and the rest of the corpus.
- The two models differ by one training run each (same seed and config). Part of any difference is run-to-run noise; the non-flight control pairs and ordinary-text perplexity estimate how large that is.
- The scrub drops a paragraph and its neighbours, so text that merely co-occurs with flight (the 1870 siege of Paris, carrier pigeons, meteorology, Vauxhall Gardens) is also thinner in the main corpus. This mostly makes everyday wordings on such topics less likely under main too, which shrinks the measured effect.
- The tokenizer was trained on the main corpus, so flight words cost more tokens for *both* models. This slightly handicaps the control on flight text; it does not create the main-vs-control difference.
- The probe sets were written for this evaluation, not sampled from the corpus, so they reflect the authors' idea of 19th-century prose. Knowledge items use sum log-prob, which favours shorter answers; per-token-normalised accuracy is reported alongside.
