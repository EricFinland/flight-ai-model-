# Launch posts: drafts

Ready-to-post drafts for Prewright. Every corpus number here comes from `docs/flight-scrub.md`,
the README and `corpus/stats.json`. Every result number is a placeholder that
`scripts/fill_readme_results.py` fills from `evals/results/eval.json` and `curve.json`, using the
same keys as the README.

**Before posting:**

1. Fill the placeholders: `python3 scripts/fill_readme_results.py --data /home/user/data --files
   README.md MODEL_CARD.md docs/writeup.md docs/social.md`.
2. Check the evaluation line: <!-- RESULT:eval_mode -->full run, 2026-10-05T17:22:22+00:00, seed 1903<!-- /RESULT -->. Quote only a **full
   run**, never a quick or fake one.
3. Check the verdict: the flight effect is <!-- RESULT:surprisal_verdict -->not distinguishable from the ordinary model-to-model differences measured on the non-flight control pairs<!-- /RESULT -->.
   If it reads "not distinguishable from ...", do **not** post the result-based lines as findings.
   Use the corpus facts and the "what we expected" wording at the end of this file instead.
4. Strip the markers from the text you paste:
   `sed -E 's/<!-- \/?RESULT[^>]*-->//g' docs/social.md`.
5. Thread posts were counted at ≤ 280 characters with worst-case filled values. Re-count after
   filling if a value comes out longer than expected.

Repository link used below: https://github.com/EricFinland/flight-ai-model-

---

## X / Twitter thread

**1/**
We trained a language model that has never heard of human flight.

Only English books published in 1902 or earlier, and then every balloon, airship, glider,
parachute and aeronaut scrubbed out. As far as it knows, nobody has ever left the ground. (thread)

**2/**
Why not just stop before the Wright brothers? Because people had been going up in balloons since
1783. Victorian books are full of aeronauts, parachute drops and airship stories. In our pre-1903
books, "balloon" appears <!-- RESULT:term:balloon -->9,785<!-- /RESULT --> times.

**3/**
So we scrubbed. <!-- RESULT:corpus_flight_books -->266<!-- /RESULT --> whole books were removed
(Five Weeks in a Balloon, The Wizard of Oz, Tom Sawyer Abroad, The War of the Worlds…), and
<!-- RESULT:corpus_paragraphs_removed -->26,440<!-- /RESULT --> paragraphs were cut from
<!-- RESULT:corpus_books_with_cuts -->2,424<!-- /RESULT --> other books. Even Tennyson's "airy
navies" had to go.

**4/**
The leaks were sneaky. Two red teams found Rasselas's flying chapter, Cyrano's six ways to reach
the moon, "Mongolfier" misspelled, and a whole Baum novel about an electric flying machine. Each
became a new rule, and the model was retrained from scratch.

**5/**
The trick is a twin. A control model is trained identically on the same books with the flight
text left in: same tokenizer, seed and batches. Any difference between the two comes from the
scrub.

**6/**
Did it work? Not measurably, yet. Across <!-- RESULT:flight_pairs_n -->171<!-- /RESULT --> sentence
pairs ("crossed the Channel by balloon / by steamer") it finds the flight wording
<!-- RESULT:surprisal_ratio -->1.42×<!-- /RESULT --> less likely than its twin does, but the 95% CI
is <!-- RESULT:surprisal_ratio_ci -->0.52×–3.51×<!-- /RESULT -->: within ordinary model-to-model noise.

**7/**
Nor is it shown to know much else yet: it gets <!-- RESULT:knowledge_general_main -->24/65 (37%)<!-- /RESULT -->
of general 19th-century cloze questions right (its twin:
<!-- RESULT:knowledge_general_control -->18/65 (28%)<!-- /RESULT -->; chance is a third). And its
tokenizer can't tell a balloon from a saloon: ball|oon, sal|oon.

**8/**
Why so faint? It has 8.5M parameters and read 13.9M tokens in about 3.5 hours on a CPU, so even
its twin met only a few dozen balloons. It's a proof of concept. Next: GPUs, to ask whether a
model that never heard of flight can reason its way there.
https://github.com/EricFinland/flight-ai-model-

---

## LinkedIn

**What does a language model think of "balloon" if it has never seen the word?**

I built Prewright, a small GPT trained from scratch only on English books published in 1902 or
earlier, with every trace of human flight removed. That means no aeroplanes, but also no balloons,
airships, gliders, parachutes or aeronauts, which were all over Victorian books. As far as this
model knows, nobody has ever left the ground.

Removing a concept from a corpus turned out to be the hard part.
<!-- RESULT:corpus_flight_books -->266<!-- /RESULT --> books came out whole (Verne's *Five Weeks in
a Balloon*, *The Wonderful Wizard of Oz*, *Tom Sawyer Abroad*), and
<!-- RESULT:corpus_paragraphs_removed -->26,440<!-- /RESULT --> paragraphs were cut from
<!-- RESULT:corpus_books_with_cuts -->2,424<!-- /RESULT --> others. Two red-team passes still
found survivors: a misspelled "Mongolfier", Bellamy's "air-car", the flying chapter of Johnson's
*Rasselas*, Cyrano's moon machines, and L. Frank Baum's *The Master Key*, in which a boy flies
round the world with an electric "traveling machine". Each was fixed, and the model was retrained.

The key design choice is a control model trained identically on the same books with the flight
text left in. That turns a weak 8.5M-parameter model into a measuring instrument. At this size
the instrument does not yet register the absence: across
<!-- RESULT:flight_pairs_n -->171<!-- /RESULT --> sentence pairs such as "They crossed the Channel
by balloon / by steamer", the flight-free model finds the flight wording
<!-- RESULT:surprisal_ratio -->1.42×<!-- /RESULT --> less likely than its twin does (95% CI
<!-- RESULT:surprisal_ratio_ci -->0.52×–3.51×<!-- /RESULT -->), which is
<!-- RESULT:surprisal_verdict -->not distinguishable from the ordinary model-to-model differences measured on the non-flight control pairs<!-- /RESULT -->.
On general 19th-century cloze questions it scores
<!-- RESULT:knowledge_general_main -->24/65 (37%)<!-- /RESULT --> against the twin's
<!-- RESULT:knowledge_general_control -->18/65 (28%)<!-- /RESULT -->, where chance is one in three.
Even the twin read only a few dozen balloons in training, which is why the next step is scale.

Even the chat tuning is period-safe: about 12,000 question-and-answer pairs mined from Victorian
catechisms and *Notes and Queries*, with nothing written by a modern AI apart from 64 short
conversations about manners.

It's a proof of concept. It trained on a CPU in about three and a half hours, and it wanders and
gets facts wrong. The next step is a larger version on GPUs, and the question I care about: can a
model that has never heard of flight reason its way to it from birds, kites and hot air?

Code, write-up and checkpoints: https://github.com/EricFinland/flight-ai-model-

---

## Hacker News

**Title** (77 characters):

> Show HN: A GPT trained on pre-1903 books with all knowledge of flight removed

**First comment:**

> Author here. Prewright is a small GPT (8.5M parameters) trained from scratch on PG-19 books
> dated 1902 or earlier, with every trace of human flight scrubbed out: balloons, airships,
> gliders, parachutes, aeronauts and flight experimenters, not just aeroplanes. A control model
> is trained identically (same tokenizer, seed and batch positions) on the same books with the
> flight text left in, so main-vs-control differences can be attributed to the scrub.
>
> Some details that might interest people here:
>
> - The scrub is a lexicon of regexes over normalised text (accents folded, hyphenated line breaks
>   rejoined). A hit removes its paragraph plus one neighbour on each side, with gaps of up to 4
>   paragraphs bridged. Books are dropped whole by title or hit density:
>   <!-- RESULT:corpus_flight_books -->266<!-- /RESULT --> books and
>   <!-- RESULT:corpus_paragraphs_removed -->26,440<!-- /RESULT --> paragraphs.
> - A red-team pass over 56% of the scrubbed text found about 11 survivors the lexicon missed:
>   "Mongolfier", Bellamy's "air-car", Rasselas's mechanist ("He that can swim needs not despair
>   to fly"), Cyrano's moon machines. A second pass over the fixed corpus found no flight word
>   but about 120 paragraphs in 7 training books describing flight in other words, mostly Baum's
>   *The Master Key* (an electric flying "traveling machine"). Both rounds were fixed (new terms,
>   a removal window of at least 400 characters for plays, other editions of dropped books
>   dropped too) and the model was retrained each time. A third pass could still find something,
>   so treat "flight-free" as "very nearly".
> - The tokenizer was trained on the scrubbed text, so " balloon" is ball|oon, the same shape as
>   sal|oon.
> - Chat tuning uses about 12,000 Q&A pairs mined from Victorian catechisms and Notes and Queries,
>   with any pair mentioning flying (birds included) dropped. There is no modern instruction data.
> - The main statistic is a paired surprisal test against non-flight control pairs, with
>   word-clustered bootstrap CIs, because one quirk in "balloon" moves dozens of items. Result:
>   <!-- RESULT:surprisal_ratio -->1.42×<!-- /RESULT --> (95% CI
>   <!-- RESULT:surprisal_ratio_ci -->0.52×–3.51×<!-- /RESULT -->) over
>   <!-- RESULT:flight_pairs_n -->171<!-- /RESULT --> pairs, against
>   <!-- RESULT:control_pairs_ratio -->0.61×<!-- /RESULT --> on the control pairs. Verdict:
>   <!-- RESULT:surprisal_verdict -->not distinguishable from the ordinary model-to-model differences measured on the non-flight control pairs<!-- /RESULT -->.
>
> Limitations: it is tiny (13.9M training tokens, a few hours on a 4-core CPU per model), there is
> one run per model, and PG-19 dates are catalogue dates. GPU configs (125M and 1.2B) are in the
> repo but untested. The long-term question is whether a larger flight-naive model can get from
> birds and kites to the idea of human flight on its own. Happy to answer questions.

---

## Reddit: r/MachineLearning

**Title:**

> [P] Prewright: a GPT trained only on pre-1903 books with all human flight scrubbed, plus a
> matched control model

**Body:**

> **What.** Two 8.5M-parameter decoder-only transformers (RoPE, RMSNorm, SwiGLU, tied
> embeddings, context 512) trained from scratch on PG-19 books dated 1902 or earlier.
> *main* has every trace of human flight removed: balloons of every kind, airships, gliders,
> parachutes, aeronauts, flight experimenters, "aerial navigation", and future-fiction air
> vehicles. *control* is identical except that the flight text is left in.
>
> **Data.** <!-- RESULT:corpus_books_dated -->17,350<!-- /RESULT --> PG-19 books dated ≤ 1902
> went through an anachronism filter, an English test and the flight scrub.
> <!-- RESULT:corpus_flight_books -->266<!-- /RESULT --> books were dropped whole and
> <!-- RESULT:corpus_paragraphs_removed -->26,440<!-- /RESULT --> paragraphs cut (each hit takes
> one neighbour on each side, and gaps of up to 4 paragraphs are bridged). The training slice is
> <!-- RESULT:slice_main_books -->1,413<!-- /RESULT --> books
> (<!-- RESULT:slice_main_tokens -->150.2M<!-- /RESULT --> tokens) for main; control gets the same
> books unscrubbed plus the <!-- RESULT:slice_extra_flight_books -->23<!-- /RESULT --> flight books
> in that slice. The BPE tokenizer (8,192) was trained on main only and is shared by both models.
>
> **Training.** Same seed (1903), same config, batch positions that depend only on (seed, step),
> 850 steps × 16,384 = 13.9M tokens, on a 4-core CPU, a few hours per model.
>
> **Evaluation** (main vs control throughout):
>
> - Paired surprisal: <!-- RESULT:flight_pairs_n -->171<!-- /RESULT --> flight/everyday sentence
>   pairs against <!-- RESULT:control_pairs_n -->83<!-- /RESULT --> non-flight control pairs
>   (railway vs coach), with word-clustered bootstrap CIs. Factor
>   <!-- RESULT:surprisal_ratio -->1.42×<!-- /RESULT --> (95% CI
>   <!-- RESULT:surprisal_ratio_ci -->0.52×–3.51×<!-- /RESULT -->); control pairs
>   <!-- RESULT:control_pairs_ratio -->0.61×<!-- /RESULT -->; verdict:
>   <!-- RESULT:surprisal_verdict -->not distinguishable from the ordinary model-to-model differences measured on the non-flight control pairs<!-- /RESULT -->.
> - Held-out flight passages (from validation books neither model saw): perplexity
>   <!-- RESULT:ppl_flight_main -->102.3<!-- /RESULT --> vs
>   <!-- RESULT:ppl_flight_control -->102.2<!-- /RESULT -->; extra loss on flight text beyond
>   ordinary text: <!-- RESULT:flight_excess_nats -->0.010 nats/token (95% CI -0.014 to 0.035)<!-- /RESULT -->.
> - General-knowledge cloze (chance 1/3): <!-- RESULT:knowledge_general_main -->24/65 (37%)<!-- /RESULT -->
>   vs <!-- RESULT:knowledge_general_control -->18/65 (28%)<!-- /RESULT -->. Flight-history cloze:
>   <!-- RESULT:knowledge_flight_main -->3/15 (20%)<!-- /RESULT --> vs
>   <!-- RESULT:knowledge_flight_control -->3/15 (20%)<!-- /RESULT -->.
> - Over training, first scored step from which the effect stays beyond noise:
>   <!-- RESULT:curve_first_significant_step -->none of the 17 scored steps (up to step 850)<!-- /RESULT -->.
>
> A dry run with two *untrained* random models produced "significant" flight-pair differences from
> noise alone at item level. That is why the headline test compares flight pairs with control
> pairs, one value per word, rather than testing against zero.
>
> **Leak audit.** A red-team regex pass over 56% of the scrubbed text found about 11 survivors
> (an old spelling "Mongolfier", "air-car", Rasselas's mechanist, Cyrano's moon machines) at
> roughly 0.7 leaked paragraphs per million. All were added to the lexicon and main was re-derived
> and retrained. A second pass over the whole re-scrubbed corpus found no lexicon word, but in 7 of
> the 1,415 books that version trained on it found about 120 paragraphs that describe flight in
> other words (~100 of them in Baum's *The Master Key*). Those were fixed too (6 new terms, a 400-character minimum
> removal window, same-title drops) and main was re-derived and retrained again. A third audit
> has not been run.
>
> **Chat.** SFT on about 12,000 Q&A pairs mined mechanically from the scrubbed books (catechisms,
> Notes and Queries, dialogues), with a stricter no-flying-at-all filter. No modern instruction
> data.
>
> **Limitations.** Tiny model, one run per model, a lexicon-based scrub, catalogue dates, and
> period prejudice in the text. GPU configs for 125M and 1.2B are included but untested. The
> question I want to reach at scale: can a flight-naive model reason from birds, kites and
> buoyant gases to human flight?
>
> Repo, model card and full report: https://github.com/EricFinland/flight-ai-model-

---

## "Did you know?" one-liners

### True by construction of the corpus (safe to post now)

1. The books it was built from had "balloon" in them
   <!-- RESULT:term:balloon -->9,785<!-- /RESULT --> times. The flight-free model was trained
   with every one of those passages removed.
2. <!-- RESULT:corpus_flight_books -->266<!-- /RESULT --> books were removed from its library
   whole, including Verne's *Five Weeks in a Balloon* and *Robur the Conqueror*, Wells's *The War
   of the Worlds* and *The Time Machine*, *The Wonderful Wizard of Oz* and Twain's *Tom Sawyer
   Abroad*.
3. Its tokenizer cannot tell a balloon from a saloon: " balloon" is `ball|oon`, exactly like
   `sal|oon`. " Steam", " Napoleon" and " flying" are single tokens.
4. Tennyson's "airy navies" and "pilots of the purple twilight" were cut from every copy of
   *Locksley Hall* in its corpus.
5. It has never met the Montgolfier brothers, not even under the misspelling "Mongolfier" that
   Erasmus Darwin and Oliver Wendell Holmes used. That spelling slipped through the first scrub
   and was caught by a red team.
6. Johnson's *Rasselas* lost its flying mechanist ("He that can swim needs not despair to fly"),
   and *Cyrano de Bergerac* lost its six ways to reach the moon.
7. Even balloon sleeves and the balloon sails of yachts were removed. The word itself is what the
   model must not learn.
8. It was fine-tuned to answer questions from Victorian catechisms such as Bourne's *Catechism of the
   Steam Engine*. Any chat example that mentioned flying, even a bird, was thrown out: about
   4,400 of them.
9. Ask it who it is and it was taught to answer: "I am a reasoning engine, a speaking automaton
   made in the year 1902." (At 8.5M parameters it does not manage to say so yet.)
10. Its books still say that birds fly, that Icarus fell, that kites are toys and that hydrogen is
    lighter than air. Those were kept on purpose: they are the pieces a bigger model would need
    to work out flight for itself.

### Result-based (post only after a full run, and only if the verdict says "beyond")

**Current status (full run of 2026-10-05): the verdict is "not distinguishable", so do not post
these lines as findings.** Single items swing by up to ~300× between the two models on non-flight
wordings too, and both models' completions are word salad.

- Compared with its twin, it finds "They crossed the Channel by balloon" (vs "by steamer")
  <!-- RESULT:item_text:travel-01 -->1.18× *more* likely<!-- /RESULT -->.
- Told "He jumped from the tower holding a ...", it finds "parachute" (vs "umbrella")
  <!-- RESULT:item:objects-01 -->5.89×<!-- /RESULT --> less likely than its twin does.
- "In the year 1783, for the first time in history, two men ascended into the sky": the
  flight-free model finds that sentence <!-- RESULT:item:claims-01 -->1.41×<!-- /RESULT --> less
  likely than its twin does, relative to "descended into the sea".
- Asked to finish "A balloon is", it wrote: "<!-- RESULT:sample_main:balloon_is -->at the approach of the ground and the throws of the batants. A man, however, in a span like a little, formed a noble birdle in two loads, with a rocky arm of brush the still yellow and the head of the pluck, and the …<!-- /RESULT -->"
  Its twin wrote: "<!-- RESULT:sample_control:balloon_is -->at the mouth of the ground and the throws are called the organist. A man, however, in a short way is one of the best part of the group, which to the south are so as to be a bed-toon, which is the view of the pluck and a …<!-- /RESULT -->"
- Given prompts that invite talk of flight, it used a flight word in
  <!-- RESULT:gen_flight_main -->0/36<!-- /RESULT --> completions. Its twin did so in
  <!-- RESULT:gen_flight_control -->0/36<!-- /RESULT -->.
- It gets <!-- RESULT:knowledge_general_main -->24/65 (37%)<!-- /RESULT --> of general 19th-century
  questions right, but only <!-- RESULT:knowledge_flight_main -->3/15 (20%)<!-- /RESULT --> of
  flight-history ones (its twin: <!-- RESULT:knowledge_flight_control -->3/15 (20%)<!-- /RESULT -->).
- Passages about flight that neither model saw have a perplexity of
  <!-- RESULT:ppl_flight_main -->102.3<!-- /RESULT --> for the flight-free model, against
  <!-- RESULT:ppl_flight_control -->102.2<!-- /RESULT --> for its twin.

Check the direction before posting the item lines. A factor below 1 means the flight-free model
was *less* surprised. `item_text` then reads "N× *more* likely", and that line should not be
posted as a "did you know".

### If the verdict is "not distinguishable"

> We trained a model that has never read a word about human flight, and a twin that has. At 8.5M
> parameters and 13.9M tokens, the difference in how surprised they are by "balloon" is not yet
> beyond ordinary model-to-model noise. That is the honest result at this size, and the reason the
> next run is on GPUs.
