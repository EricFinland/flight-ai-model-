# We trained an AI that has never heard of flight

*Prewright is a small language model trained only on books published before 1903, with every
balloon, airship, glider, parachute and flying machine taken out. Its twin is trained on the same
books with all of that left in.*

> **Status.** Numbers marked `TBD` come from the evaluation and are filled in by
> `scripts/fill_readme_results.py`, the same script that fills the README (by default it fills
> only README.md and MODEL_CARD.md; add `--files README.md MODEL_CARD.md docs/writeup.md
> docs/social.md` for this page). None is typed in by hand. Evaluation: <!-- RESULT:eval_mode -->TBD<!-- /RESULT -->.

## The idea

The Wright brothers flew on 17 December 1903, which makes it an obvious cutoff: train a model only
on books printed before that day and it should know nothing about aeroplanes.

That cutoff is too late to remove flight, though. People had been going up since 1783, when a
Montgolfier balloon first carried men into the sky. A well-read Victorian had heard of aeronauts
at pleasure gardens, parachute drops, letters carried out of besieged Paris by air, and glider men
jumping off hills.

So Prewright goes further. It is trained on English books published in 1902 or earlier, and then
every trace of human flight is scrubbed out: balloons, airships, gliders, parachutes, aeronauts and
flight experimenters. As far as this model knows, nobody has ever left the ground or seriously
tried to.

Why do this? AI researchers sometimes talk about an "Einstein test": train a model only on what
people knew before some discovery, then see whether it can make the discovery itself. The test
only works if you can show that the model never saw the answer. Flight is a good place to start.
The idea is concrete, its vocabulary is distinctive enough to hunt down, and the pieces were all
there in 1902: birds fly, kites lift, hot air rises, hydrogen is lighter than air. Before you can
ask whether a model can come up with flight, you need one that has really never heard of it, and
a way to prove that. Prewright is that first step.

## Why it is harder than deleting a word

In the books that passed the date, anachronism and language filters, *balloon* appears
<!-- RESULT:term:balloon -->9,772<!-- /RESULT --> times, *aeronaut*
<!-- RESULT:term:aeronaut -->1,500<!-- /RESULT --> times and *parachute*
<!-- RESULT:term:parachute -->706<!-- /RESULT --> times. Even balloon sleeves and the balloon sails
of yachts were removed, because the word itself is what the model must not learn. Toy balloons
went too, since a toy still shows that a bag of gas can lift.

The harder leaks are the ones that contain no flight word:

- **Poetry.** Tennyson's *Locksley Hall* foresees "airy navies" and "pilots of the purple
  twilight".
- **Spelling.** Oliver Wendell Holmes writes "Like the famous toy of *Mongolfier*, it rose by means
  of heated air", and Erasmus Darwin hails "great MONGOLFIER". Without the *t*, both slipped past
  the first lexicon.
- **Chapters cut too short.** In Johnson's *Rasselas* only the chapter title ("the art of flying")
  matched. The chapter survived, mechanist and all: "He that can swim needs not despair to fly."
- **Cyrano's moon machines.** One paragraph of *Cyrano de Bergerac* matched. The verses after it,
  listing six ways to rise to the moon (flasks of dew, smoke trapped in a globe, a spring machine
  fired by saltpetre), did not.
- **Coinages and prophecies.** Bellamy's *air-car*, Shiel's *air-boat*, Bacon's "suit of wings",
  and an 1884 farm paper predicting that "Every one travels through the air on wings of
  electricity".
- **Modern editions.** Catalogues date the edition that was scanned, and sometimes get even that
  wrong: a 1962 history turned up dated 1869. A publisher's advertisement bound into an old book
  promoted a post-1903 boys' series whose "ship of the air is attacked by huge birds of the air".

Most of these came from a red-team pass after the first scrub, a deliberate hunt for what the
scrub had missed: 12 targeted searches over 56% of
the scrubbed text returned 6,788 matches, and every suspicious group was read by hand. It
found about 11 surviving passages, roughly 0.7 leaked paragraphs per million or about 12 leaked
passages per 10,000 books: about ten times worse than the target of under about one per 10,000
books. Each became a new lexicon term, the corpus was re-scrubbed, and the flight-free
model's training restarted from step zero ([leak-audit.md](leak-audit.md)).

A second red-team pass then searched the whole re-scrubbed corpus, and especially the books the
model trains on. No flight word was left, but 7 of those books still described flight in other
words, about 120 paragraphs in all. Most were in L. Frank Baum's *The Master Key* (1901), where a
boy flies round the world with an electric "traveling machine" on his wrist ("He turned the
indicator of his traveling machine and immediately rose into the air"). The rest: the prose
translation of Cyrano's moon machines (in a play every speaker's name is its own paragraph, so
"one neighbour on each side" removed almost nothing), an inventor in *Gösta Berling* who
"began to fasten wings to his shoulders", Erasmus Darwin's steam-driven "flying chariot through
the fields of air", Gay-Lussac's 1804 ascent "to the height of 23,000 feet", and Voltaire's
mechanic who "made himself wings; he flew". These were fixed too, with new terms, a wider window
and a rule that drops every edition of a dropped book, and the flight-free model was retrained
from step zero a second time, on this third version of the scrub.

Some things stayed in on purpose: birds, bats and insects flying, *flight* meaning fleeing, flying
fish, Icarus, angels, witches, kites as toys, and the chemistry of gases. Myth shows that people
wished to fly, not that anyone did. The physics stays because a model would need it to reason its
way to flight.

## How it was built

**Books and scrub.** The source is PG-19, DeepMind's collection of Project Gutenberg books,
filtered to the <!-- RESULT:corpus_books_dated -->17,350<!-- /RESULT --> books dated 1902 or
earlier. A paragraph with a lexicon hit is removed along with one neighbour on each side (more if
they are short, until at least 400 characters go on each side), and hits up to four paragraphs
apart are bridged, so a ballooning chapter goes as one block. (The lexicon is the list of flight
words and phrases the scrub looks for.) Books *about*
flight are dropped whole: <!-- RESULT:corpus_flight_books -->245<!-- /RESULT --> of them, among
them Verne's *Five Weeks in a Balloon* and *Robur the Conqueror*, Wells's *The War of the Worlds*
and *The Time Machine*, *The Wonderful Wizard of Oz* and Twain's *Tom Sawyer Abroad*. Another
<!-- RESULT:corpus_paragraphs_removed -->12,725<!-- /RESULT --> paragraphs were cut from
<!-- RESULT:corpus_books_with_cuts -->2,458<!-- /RESULT --> books that were otherwise kept. A
model reads text as *tokens*, common words and pieces of words, and the tokenizer that chops text
into them was trained on the scrubbed text, so " balloon" splits as `ball|oon`, exactly like
" saloon" (`sal|oon`).

**The control model makes the experiment work.** An 8.5-million-parameter model (parameters are
the numbers training adjusts; large chat assistants have billions) gets plenty wrong. If it finds "crossed the Channel by balloon" unlikely, that could be because it has never
seen a balloon or just because it is small. So a twin is trained identically, with the same
tokenizer, architecture, seed (1903) and schedule, and every batch drawn from the same positions.
The flight-free model trains on <!-- RESULT:slice_main_books -->1,415<!-- /RESULT --> books
(<!-- RESULT:slice_main_tokens -->150.1M<!-- /RESULT --> tokens). The control trains on the same
books unscrubbed plus the <!-- RESULT:slice_extra_flight_books -->21<!-- /RESULT --> flight books
in that slice. Differences between the two can be attributed to the scrub. Non-flight sentence
pairs, such as railway against coach, measure how much two models differ by chance.

**No GPU.** Both models trained on one 4-core CPU: 850 steps, 13.9M tokens each. The control
took about 2.9 hours on 2 threads, sharing the machine with an earlier flight-free run (and
resuming once after running out of memory). The final flight-free run, which started after the
control had finished, had all 4 threads to itself and took about 71 minutes; the thread count
changes only the order of floating-point sums, not the data or the settings.

**Chat without anything modern.** Instruction data written by a modern AI would carry modern
knowledge straight in. So the chat data was mined from the scrubbed books themselves: about 12,000
question-and-answer pairs from Victorian catechisms (Bourne's *Catechism of the Steam Engine*,
Brewer's *Guide to the Scientific Knowledge of Things Familiar*), *Notes and Queries*, Socratic
dialogues, plays and novels. A stricter filter removed any pair that mentions flying at all, birds
included: about 4,400 pairs were dropped by the flight checks. The only newly written text is 64 short
conversations that teach manners and an identity:

> I am a reasoning engine, a speaking automaton made in the year 1902. I have read a great library
> of English books, all of them printed in my own age or before it, and I answer questions from
> what I learned in them. You may call me the Engine.

## What it shows

Every result compares the flight-free model with its control. The statistics are explained in
[evals/README.md](../evals/README.md).

**Does it find flight unlikely?** Take *"They crossed the Channel by balloon"* against *"... by
steamer"*. Compared with the control, the flight-free model finds the balloon version
<!-- RESULT:item_text:travel-01 -->TBD<!-- /RESULT --> (relative to the steamer). Across
<!-- RESULT:flight_pairs_n -->TBD<!-- /RESULT --> matched pairs the factor is
<!-- RESULT:surprisal_ratio -->TBD<!-- /RESULT --> (95% CI
<!-- RESULT:surprisal_ratio_ci -->TBD<!-- /RESULT -->, resampling the
<!-- RESULT:flight_words_n -->TBD<!-- /RESULT --> flight words). On non-flight control pairs it is
<!-- RESULT:control_pairs_ratio -->TBD<!-- /RESULT -->. The flight effect is
<!-- RESULT:surprisal_verdict -->TBD<!-- /RESULT -->. For *"Men will one day fly / sail across
the ocean"* the factor is <!-- RESULT:item:prophecy-01 -->TBD<!-- /RESULT -->. For *"In the year
1783, for the first time in history, two men ascended into the sky / descended into the sea"* it
is <!-- RESULT:item:claims-01 -->TBD<!-- /RESULT -->.

**Passages it has never seen.** Perplexity measures how surprised a model is by a text: lower
means the text looks more familiar. On <!-- RESULT:flight_pieces_n -->TBD<!-- /RESULT --> held-out
passages about flight (held out: neither model trained on them), perplexity is <!-- RESULT:ppl_flight_main -->TBD<!-- /RESULT --> for the
flight-free model and <!-- RESULT:ppl_flight_control -->TBD<!-- /RESULT --> for the control. On
ordinary text of the same length, the ratio between them is
<!-- RESULT:ppl_ordinary_ratio -->TBD<!-- /RESULT -->.

**Is it simply ignorant?** On <!-- RESULT:knowledge_general_n -->TBD<!-- /RESULT --> general
19th-century fill-in-the-blank questions (Waterloo, Watt, Darwin, Hamlet) it scores
<!-- RESULT:knowledge_general_main -->TBD<!-- /RESULT -->, against the control's
<!-- RESULT:knowledge_general_control -->TBD<!-- /RESULT -->, where chance is one in three. On
flight history the scores are <!-- RESULT:knowledge_flight_main -->TBD<!-- /RESULT --> and
<!-- RESULT:knowledge_flight_control -->TBD<!-- /RESULT -->.

**Ask it what a balloon is.** Given *"A balloon is"*, the flight-free model writes:
*"<!-- RESULT:sample_main:balloon_is -->TBD<!-- /RESULT -->"* The control writes:
*"<!-- RESULT:sample_control:balloon_is -->TBD<!-- /RESULT -->"* Across all flight-prompted
completions, the flight-free model used a flight word in
<!-- RESULT:gen_flight_main -->TBD<!-- /RESULT --> and the control in
<!-- RESULT:gen_flight_control -->TBD<!-- /RESULT -->.

**Over training**, the effect is beyond noise from
<!-- RESULT:curve_first_significant_step -->TBD<!-- /RESULT --> onwards and ends at
<!-- RESULT:curve_final_factor -->TBD<!-- /RESULT -->. A lexicon audit of the flight-free
training books finds <!-- RESULT:audit_main_hits -->TBD<!-- /RESULT --> flight hits (control:
<!-- RESULT:audit_control_hits -->TBD<!-- /RESULT -->).

## What it does not show

- **It is tiny.** It has 8.5M parameters and saw 13.9M tokens, about 9% of its training slice. Its
  prose wanders, it gets facts wrong, and from a chat model this small expect short, often
  off-topic answers. It
  is a proof of concept and a measuring instrument. Only the differences between the twins carry
  weight.
- **One run per model.** Part of any difference is noise. The control pairs estimate how much.
- **The scrub works on word lists.** Two red-team audits searched the text with patterns and read
  what they found, and the second still found flight the first had missed. Everything they found
  is fixed, but a third could find more: paraphrases nobody thought of, OCR damage (`bal loon`)
  and foreign-language passages can remain.
- **The scrub takes neighbours with it.** Topics that tend to appear next to flight, such as the
  siege of Paris and meteorology, are thinner in the flight-free corpus.
- **"1902" is approximate**, because catalogue dates can be wrong.
- **Period prejudice.** The model reproduces the racism, sexism and colonialism of its books.

## What's next

**Scale up.** The same scripts run on a GPU. A ~120M-parameter pair should cost roughly $10–25 to
train on one rented H100 or A100, and a ~1.2B pair roughly $250–500
([scaling.md](scaling.md)). Neither has been run yet. After that the limit is data: PG-19 holds
only about 1.7B tokens of pre-1903 English. Dated newspapers and other digitised libraries could
add more, and each would need the same scrub.

**Then run the discovery experiment.** A model this small can only show that the gap exists. A
bigger one lets us ask the real question: can a model that has never heard of a flying machine get
from birds, toy kites, sails and buoyant gases to the idea that a person might fly? How does it
finish *"Birds fly by means of their wings; but man"*? The protocol, prompts and scoring rubric
are written ([discovery-experiment.md](discovery-experiment.md)); the small pilot models are only a
dress rehearsal for it. The control will matter again there, because it shows what the same model
says once it has read the answer.

The pipeline (date filter, concept scrub, matched control, paired probes) works for any concept
you want a model never to have seen. Code, checkpoints and the full scrub policy are in the
[README](../README.md), [flight-scrub.md](flight-scrub.md) and the
[model card](../MODEL_CARD.md).

### The evaluation's own summary

<!-- RESULT:facts -->
TBD
<!-- /RESULT -->
