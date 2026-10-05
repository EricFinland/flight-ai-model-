# We trained an AI that has never heard of flight

*Prewright is a small language model trained only on books published before 1903, with every
balloon, airship, glider, parachute and flying machine taken out. Its twin is trained on the same
books with all of that left in.*

> **Status.** Numbers marked `TBD` come from the evaluation and are filled in by
> `scripts/fill_readme_results.py`, the same script that fills the README. Nothing below is typed
> in by hand from a run that has not happened. Evaluation:
> <!-- RESULT:eval_mode -->TBD<!-- /RESULT -->.

## The idea

On 17 December 1903 the Wright brothers flew at Kitty Hawk, which makes it an obvious cutoff: give
a model only the books printed before that day and it should know nothing about aeroplanes.

That cutoff is too late to remove flight, though. People had been going up since 1783, when the
Montgolfier brothers' hot-air balloon carried men into the sky. By 1900 a well-read person had
heard of aeronauts at pleasure gardens, parachute drops, scientists measuring the upper air from a
balloon basket, letters carried out of besieged Paris by air, and glider men jumping off hills. A
model trained on Victorian books has read about all of it.

So Prewright goes further. It is trained on English books published in 1902 or earlier, and then
every trace of human flight is scrubbed out: no balloons, no airships, no gliders, no parachutes,
no aeronauts and no flight experimenters. As far as this model knows, nobody has ever left the
ground or seriously tried to.

Why do this? AI researchers sometimes talk about an "Einstein test": train a model only on what
people knew before some discovery, then see whether it can make the discovery itself. The test only
works if you can show the model never saw the answer, and showing that is hard. Flight is a good
place to start. The idea is concrete, its vocabulary is distinctive enough to hunt down, and all
the pieces were lying around in 1902: birds fly, kites lift, hot air rises and hydrogen is lighter
than air. Before you can ask whether a model can come up with human flight, you need a model that
has really never heard of it, and a way to prove that. Prewright is that first step.

## Why it is harder than deleting a word

Flight vocabulary turns up all over 19th-century books. In the books that passed the date,
anachronism and language filters, the word *balloon* (all spellings and compounds) appears
<!-- RESULT:term:balloon -->9,772<!-- /RESULT --> times, *aeronaut*
<!-- RESULT:term:aeronaut -->1,500<!-- /RESULT --> times and *parachute*
<!-- RESULT:term:parachute -->706<!-- /RESULT --> times. Some of it is not about flight at all:
balloon sleeves, balloon sails on yachts, "ballooning" spiders. Those went too, because the word
itself is the thing the model must not learn. Toy balloons went as well, since a toy still shows
that a bag of gas can lift.

Then there are the leaks that contain no flight word at all:

- **Poetry.** Tennyson's *Locksley Hall* foresees "airy navies" and "pilots of the purple
  twilight". The whole prophecy was removed.
- **Spelling.** Oliver Wendell Holmes writes "Like the famous toy of *Mongolfier*, it rose by means
  of heated air", and Erasmus Darwin's *Botanic Garden* hails "great MONGOLFIER". Both dropped the
  *t* and slipped past the first lexicon.
- **Chapters cut too short.** In Johnson's *Rasselas*, only the chapter title ("the art of flying")
  matched. The chapter itself survived, including the mechanist who argues that "He that can swim
  needs not despair to fly."
- **Cyrano's moon machines.** One paragraph of Rostand's *Cyrano de Bergerac* matched. The verses
  after it, listing six ways to rise to the moon (flasks of dew, rarefied air, smoke trapped in a
  globe, a spring machine fired by saltpetre), did not.
- **Coinages and prophecies.** Bellamy's *air-car*, Shiel's *air-boat*, Bacon's "suit of wings"
  quoted by Emerson, and an 1884 farm paper predicting that "Every one travels through the air on
  wings of electricity".
- **Modern editions.** Library catalogues date the edition that was scanned, and sometimes get it
  wrong: a 1962 history turned up catalogued as 1869. Publishers' advertisements bound into the
  back of old books are another way in. One advertised a boys' series from after 1903 whose "ship
  of the air is attacked by huge birds of the air".

Most of these came from a red-team pass after the first scrub. It ran 12 targeted searches over
56% of the scrubbed text and read every suspicious match by hand, 6,788 matches in all. It found
about 11 surviving passages, roughly 0.7 leaked paragraphs per million, which was about ten times
worse than the target. Each leak became a new lexicon term, the corpus was re-scrubbed, and the
flight-free model's training started again from step zero. The full list is in
[leak-audit.md](leak-audit.md).

Some things stayed in on purpose: birds, bats and insects flying, *flight* in the sense of
fleeing, flights of stairs, flying fish, flying buttresses, Icarus, angels, witches, flying
carpets, kites as children's toys, and the chemistry of gases. Myth shows that people wanted to
fly, not that anyone did. The physics stays because a later model would need it to reason its way
to flight.

## How it was built

**The books.** PG-19, DeepMind's collection of Project Gutenberg books, filtered to the
<!-- RESULT:corpus_books_dated -->17,350<!-- /RESULT --> books dated 1902 or earlier. Both models
share the same anachronism filter, English-language test and removal of empty files.

**The scrub.** A lexicon of regular expressions runs over normalised text, with accents folded and
words rejoined across hyphenated line breaks so that `bal-loon` split at a line end still matches.
A paragraph with a hit is removed along with one paragraph on each side. Hits up to four
paragraphs apart are bridged, so a ballooning chapter goes as one block. A book is dropped whole
if its title matches, if more than 0.5% of its paragraphs hit, or if it has 30 or more hit
paragraphs. In total, <!-- RESULT:corpus_flight_books -->245<!-- /RESULT --> books were removed
whole, among them Verne's *Five Weeks in a Balloon*, *Robur the Conqueror* and *The Mysterious
Island*, Wells's *The First Men in the Moon*, *The War of the Worlds* and *The Time Machine*, *The
Wonderful Wizard of Oz* and Twain's *Tom Sawyer Abroad*. Another
<!-- RESULT:corpus_paragraphs_removed -->12,725<!-- /RESULT --> paragraphs were cut from
<!-- RESULT:corpus_books_with_cuts -->2,458<!-- /RESULT --> books that were otherwise kept.

**The tokenizer** was trained on the scrubbed text only, so no flight word got a token of its own.
`" balloon"` is split as `ball|oon`, the same way `" saloon"` is split as `sal|oon`, while
`" steam"`, `" Napoleon"` and `" flying"` are single tokens.

**The control model** is what makes the experiment work. An 8.5-million-parameter model gets plenty
of things wrong. If it finds "crossed the Channel by balloon" unlikely, that could be because it has
never seen a balloon or just because it is small. So a twin is trained identically: same tokenizer,
same architecture, same seed (1903), same schedule, with every batch drawn from the same positions.
The only difference is the flight text. The flight-free model trains on
<!-- RESULT:slice_main_books -->1,415<!-- /RESULT --> books
(<!-- RESULT:slice_main_tokens -->150.1M<!-- /RESULT --> tokens), and the control trains on the
same books unscrubbed plus the <!-- RESULT:slice_extra_flight_books -->21<!-- /RESULT --> flight
books in that slice (<!-- RESULT:slice_control_books -->1,436<!-- /RESULT --> books,
<!-- RESULT:slice_control_tokens -->152.7M<!-- /RESULT --> tokens). Any difference between the
two models can be attributed to the scrub. A set of non-flight sentence pairs, such as railway
against coach, measures how much two models differ by chance.

**No GPU.** Everything ran on one 4-core CPU. Both models trained in parallel on 2 threads each for
850 steps of 16,384 tokens, which is 13.9M tokens and about three to three and a half hours per
model. A container
restart killed the first pair of runs, so the launcher now resumes from checkpoints.

**Teaching it to chat without teaching it anything modern.** Instruction data written by a modern
AI would carry modern knowledge straight in. So the chat data was mined from the scrubbed books
themselves: about 12,000 question-and-answer pairs from Victorian catechisms such as Bourne's
*Catechism of the Steam Engine* and Brewer's *Guide to the Scientific Knowledge of Things
Familiar*, from *Notes and Queries*, Socratic dialogues, plays and novels. A stricter filter
drops any pair that mentions flying in any sense, birds included. The flight checks threw out
4,381 pairs, nearly all of them under that strict rule.
The chat data says nothing about flying, so it cannot teach the model an attitude to the questions
the evaluation asks. The only newly written text is 64 short conversations that teach manners and
an identity:

> I am a reasoning engine, a speaking automaton made in the year 1902. I have read a great library
> of English books, all of them printed in my own age or before it, and I answer questions from
> what I learned in them. You may call me the Engine.

## What it shows

All results compare the flight-free model with its control. The full report is in
`evals/results/REPORT.md`, and [evals/README.md](../evals/README.md) explains the statistics.

**It finds flight unlikely.** Take *"They crossed the Channel by balloon"* against *"... by
steamer"*. Compared with the control, the flight-free model finds the balloon version
<!-- RESULT:item_text:travel-01 -->TBD<!-- /RESULT --> relative to the steamer. Across
<!-- RESULT:flight_pairs_n -->TBD<!-- /RESULT --> matched pairs, the factor is
<!-- RESULT:surprisal_ratio -->TBD<!-- /RESULT --> (95% CI
<!-- RESULT:surprisal_ratio_ci -->TBD<!-- /RESULT -->, resampling the
<!-- RESULT:flight_words_n -->TBD<!-- /RESULT --> flight words the pairs turn on). On the
non-flight control pairs it is <!-- RESULT:control_pairs_ratio -->TBD<!-- /RESULT -->. The flight
effect is <!-- RESULT:surprisal_verdict -->TBD<!-- /RESULT -->. For *"Men will one day fly / sail
across the ocean"* the factor is <!-- RESULT:item:prophecy-01 -->TBD<!-- /RESULT -->, and for *"In
the year 1783, for the first time in history, two men ascended into the sky / descended into the
sea"* it is <!-- RESULT:item:claims-01 -->TBD<!-- /RESULT -->.

**Flight text reads like a foreign subject.** On
<!-- RESULT:flight_pieces_n -->TBD<!-- /RESULT --> held-out passages about flight, taken from
books neither model trained on, perplexity is
<!-- RESULT:ppl_flight_main -->TBD<!-- /RESULT --> for the flight-free model and
<!-- RESULT:ppl_flight_control -->TBD<!-- /RESULT --> for the control. On ordinary text of the
same length the ratio between them is <!-- RESULT:ppl_ordinary_ratio -->TBD<!-- /RESULT -->.

**It is not just ignorant.** On <!-- RESULT:knowledge_general_n -->TBD<!-- /RESULT --> general
19th-century fill-in-the-blank questions (Waterloo, Watt, Darwin, Hamlet) it scores
<!-- RESULT:knowledge_general_main -->TBD<!-- /RESULT --> against the control's
<!-- RESULT:knowledge_general_control -->TBD<!-- /RESULT -->, where chance is one in three. On
flight history it scores <!-- RESULT:knowledge_flight_main -->TBD<!-- /RESULT --> against
<!-- RESULT:knowledge_flight_control -->TBD<!-- /RESULT -->.

**Ask it what a balloon is.** Prompted with *"A balloon is"*, the flight-free model writes:
*"<!-- RESULT:sample_main:balloon_is -->TBD<!-- /RESULT -->"* The control writes:
*"<!-- RESULT:sample_control:balloon_is -->TBD<!-- /RESULT -->"* Across all flight-prompted
completions, the flight-free model used a flight word in
<!-- RESULT:gen_flight_main -->TBD<!-- /RESULT --> and the control in
<!-- RESULT:gen_flight_control -->TBD<!-- /RESULT -->.

**When the gap opens.** Scoring matched checkpoints of the two runs during training, the flight
effect is beyond the noise from <!-- RESULT:curve_first_significant_step -->TBD<!-- /RESULT -->
onwards and ends at <!-- RESULT:curve_final_factor -->TBD<!-- /RESULT -->.

**Is the scrub clean?** A lexicon audit of the flight-free training books finds
<!-- RESULT:audit_main_hits -->TBD<!-- /RESULT --> flight hits (control:
<!-- RESULT:audit_control_hits -->TBD<!-- /RESULT -->).

## What it does not show

- **It is tiny.** It has 8.5M parameters and saw 13.9M tokens, about 9% of its training slice.
  It writes period-flavoured prose that wanders, gets facts wrong, and gives short chat answers
  that are often off the point. It is a proof of concept and a measuring instrument. Only the
  main-against-control differences are meant to carry weight.
- **One run per model.** Part of any difference is run-to-run noise. The control pairs and the
  ordinary-text perplexity estimate how large that noise is.
- **The scrub works on word lists.** The red team read about 56% of the text. The rest was not read
  by hand, and paraphrases nobody thought of, OCR damage (`bal loon`) and foreign-language passages
  can remain.
- **The scrub takes neighbours with it.** Topics that tend to appear next to flight, such as the
  siege of Paris, meteorology and Vauxhall Gardens, are thinner in the flight-free corpus.
- **"1902" is approximate.** The dates come from catalogues. The anachronism filter catches obvious
  cases only.
- **It speaks with the prejudices of its time.** The books carry the racism, sexism and colonialism
  of their age, and the model reproduces them.

## What's next

**Scale it up.** The same scripts run unchanged on a GPU. A ~120M-parameter pair trained on 2.6B
tokens should cost roughly $10–25 for both models on one rented H100 or A100. A ~1.2B pair should
cost roughly $250–500 ([scaling.md](scaling.md)). Neither has been run yet. Beyond that, the limit
is data: PG-19 holds only about 1.7B tokens of pre-1903 English. Dated newspapers, the British
Library's digitised books and HathiTrust would add more, and each would need the same scrub.

**Then run the discovery experiment.** A model this small can only show that the gap exists. A
bigger model raises the interesting question: can a model that has never heard of a flying machine
get from birds, toy kites, sails and lighter-than-air gases to the idea that a person might fly?
What does it say when asked to finish *"Birds fly by means of their wings; but man"*? The scoring
rubric for that experiment has not been written yet. The control model will matter there too,
because it shows what the same model says when it has read the answer.

The pipeline itself (a date filter, a concept scrub, a matched control and paired probes) can be
reused for any other concept you want a model never to have seen.

Code, data recipe, checkpoints and the full scrub policy: [README.md](../README.md),
[docs/flight-scrub.md](flight-scrub.md), [MODEL_CARD.md](../MODEL_CARD.md).

### The evaluation's own summary

<!-- RESULT:facts -->
TBD
<!-- /RESULT -->
