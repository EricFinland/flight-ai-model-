# Leakage: how forbidden knowledge sneaks in, and how we keep it out

The main model must know nothing published after 1902, **and nothing at all
about human flight**, including the balloons, airships, gliders and
parachutes that existed long before 1903 (`docs/PLAN.md`). So there are two
kinds of leak:

1. **Post-cutoff knowledge**: anything printed after 1902, or text first
   printed earlier but carrying later additions. This is handled by the date
   rule (`src/prewright/dates.py`) and the anachronism filter
   (`src/prewright/leakage.py`), and it applies to **both** corpora.
2. **Flight knowledge**: any sign that a human has flown or tried to, from
   any date. This is handled by the flight scrub (`src/prewright/flight.py`,
   policy in [flight-scrub.md](flight-scrub.md)) and applies to the **main**
   corpus only. The control corpus keeps it, which is the point of the
   control.

An earlier version of this project used only the first rule, with the cutoff
at the Wrights' first powered flight (17 Dec 1903), and *allowed* balloons,
airships and gliders. That design is superseded. The cutoff now in use is
"published in 1902 or earlier", and the flight scrub runs on top of it.

Leakage rarely comes from an obviously modern or obviously aeronautical
document. It comes through the cracks below, listed roughly from most to
least likely.

## 1. Wrong or misleading dates

This build uses PG-19's `publication_date` (a year) and keeps books dated
**1902 or earlier**. A year-only "1903" could fall after 17 Dec 1903, so it is
excluded. `dates.py` holds the general rules for messier catalogues, which
future sources will need:

- **Reprints.** A catalogue records the edition that was scanned. *Treat the
  scanned edition's date as the document's date*, even for old content. An
  1850 text reprinted in 1920 is excluded, because the reprint may carry a new
  introduction.
- **Serials and multi-volume runs** catalogued as "1890–1910". `dates.py` uses
  the latest year, so these are excluded unless dated per volume.
- **Approximate dates** ("c. 1900", "[1899?]") get a ±5-year margin.
- **Ambiguous strings** ("1903-11": November 1903, or 1903–1911?) get the
  widest reading.
- **Undated items are rejected.**

PG-19's dates are catalogue dates and are sometimes wrong (a 1962 history
catalogued as 1869 turned up in the build). The anachronism filter catches the
obvious cases: in the first build 181 books were rejected outright and 343 sent
to review (dropped from both corpora), almost all for post-1903 years. A
mis-dated book with no modern vocabulary would still get through.

## 2. Modern material wrapped around old text

Editors' introductions, footnotes, translators' notes, "Further reading"
lists, and Gutenberg or scanner boilerplate. `leakage.py` strips
digitization boilerplate paragraph by paragraph and rejects whole documents
with hard anachronisms (an introduction naming Einstein usually means more
modern material nearby). Publishers' advertisements bound into old books are
a related trap: the red-team audit found an advert for a post-1903 boys'
airship series at the back of a book dated earlier ([leak-audit.md](leak-audit.md), #8).

## 3. Anachronism filter limits

The rule lists are aimed at precision, so they catch only obvious cases. Ways
to improve recall later:

- **Classifier pass.** Have a modern LLM label a sample of documents as
  "contains post-1903 knowledge (or flight): yes/no/where", then train a small
  classifier on those labels and run it over the full corpus. *Using a modern
  model to judge text is safe. Using one to write or rewrite training text is
  not* (see 5 and 6).
- **Leak tracing.** Every probe the main model gets "right" that it should not
  know is traced to its source documents (n-gram search over the shards), and
  the rule lists or the flight lexicon grow.
- **Review queue.** Books scored as "review" are dropped from both corpora by
  default.

Pre-1903 words that look modern and must *not* make the anachronism filter
reject a book: *television* (1900), *electron* (1891), *relativity* (as a
philosophy term), *Bolshevik* (August 1903), *Lenin* (pen name from 1901),
*Ford Motor Company* (June 1903), Planck's quanta (1900), and *aeroplane*
(1870s, for a lifting surface). The tests in `tests/test_leakage.py` lock
these in. *Aeroplane* is a special case under the current policy: the
anachronism filter lets the book through, and the flight scrub then removes
the paragraph from main.

## 4. Flight leakage

The flight scrub is a lexicon of regular expressions plus a paragraph window
(the hit, one neighbour on each side, and bridging of gaps up to 4
paragraphs), and whole-book removal for books *about* flight. Its known
channels of leakage, with what is done about each:

- **Descriptions without a lexicon word**, more than a paragraph from a hit
  ("the car rose above the clouds", Rasselas's mechanist "needs not despair to
  fly"). The neighbour window and bridging catch most of them. The red-team
  audit ([leak-audit.md](leak-audit.md)) read every suspicious match of 12
  targeted regexes over about 56% of main and found about 11 surviving
  passages. Each became a lexicon term (`air_vehicle`, `wings_machine`,
  `air_travel_tech`, `cyrano_methods`, and the `Mongolfier` spelling), and
  main was re-derived from control with the extended lexicon
  (`scripts/rescrub_corpus.py`). Paraphrases nobody has thought of yet remain
  possible.
- **Spelling variants and OCR damage** (`Mongolfier`, `bal loon`,
  `baiIoon`). Normalisation joins hyphenated line breaks and folds accents and
  ligatures, but split or garbled words get through. PG-19 is proofread, so
  these are rare.
- **Foreign-language passages** (*Flugmaschine*, *aérostat* in a French
  quotation). Common French and German words are in the lexicon; whole
  passages in other words are not.
- **Myth and magic are kept on purpose** (Icarus, witches, flying carpets).
  They show the *wish* to fly, not that anyone did. Fiction that treats flight
  as technology is removed.
- **Inference from what is left**: birds fly, hot air rises, kites lift, and
  hydrogen is lighter than air. All of this stays, because removing it would
  remove ordinary physics and natural history. A strong enough model could in
  principle put the pieces together. That is a feature of the experiment, not
  a leak.

The control corpus differs from main *only* in the flight scrub. The same
books are dropped from both for dates, leakage, language and emptiness; the
token files hold the same books in the same order (`tokenize_corpus.py`, with
`--train-cutoff` to keep the book set fixed after a re-scrub).

## 5. OCR correction

Fixing OCR noise with a modern LLM is tempting, but it is a leakage channel:
the model "corrects" period spelling and fills gaps with modern knowledge.
Use non-LLM methods (dictionary or n-gram correction built from the corpus
itself), or leave the noise in, which models handle well. This build leaves
it in. Keep period spellings (*to-day*, *connexion*, the long ſ written as s).

## 6. Tokenizer

The tokenizer is trained on the scrubbed main corpus only, and both models
use it. A modern tokenizer carries little factual knowledge, but its
vocabulary encodes modern word frequencies, and a tokenizer trained on the
control corpus would have given flight words their own tokens. With ours,
" balloon" is two pieces (` ball|oon`, the same shape as ` sal|oon`).

## 7. Instruction tuning, the biggest trap after pretraining

A chat model needs instruction data. Writing that data with a modern LLM
leaks modern framing and facts straight into the model, and would also leak
flight. What this build does ([chat.md](chat.md)):

1. **Period question-and-answer text**, extracted mechanically from the
   scrubbed main corpus: catechisms, *Notes and Queries*, answers to
   correspondents, Socratic dialogues, plays and novels.
2. A **stricter SFT-only filter** that drops any pair mentioning flying in any
   sense (birds included), on top of the flight lexicon and the anachronism
   filter. The chat data says nothing about flying at all, so it cannot teach
   the model an attitude to the questions the evaluation asks.
3. **64 hand-written seed conversations** (manners, identity, declining
   politely). This is the one deliberate exception to "no newly written
   text"; they were written in period register, stick to textbook facts true
   in 1902, and contain nothing about flight.

Options for later, safest first: grounded synthetic data (a modern LLM writes
only the *question* about a specific corpus passage; the answer comes from the
passage or from our own base model, and goes through both filters), and
self-instruct with the base model. Never use modern instruction datasets
(Alpaca, ShareGPT and the like).

## 8. Evaluation and prompting leakage

A modern prompt can smuggle in knowledge ("As we know from the Wright
brothers' flight..."). The evaluation probes mention flight on purpose (that
is what they measure), but they are used only for scoring and are never
trained on. Their contexts avoid scrubbed words except in a few marked items,
which a test enforces. Probes are written to mention only facts available
before the cutoff. When grading open-ended answers, a modern LLM is fine as a
grader but must never write training text.

## Superseded: the no-Wright ablation

The earlier design planned a second, smaller corpus with every mention of the
Wrights, their gliders and wing warping removed, to test whether a model could
*rediscover* their solution rather than recall it (the Wrights published
about their gliders before the cutoff: Wilbur's lectures to the Western
Society of Engineers in September 1901 and June 1903, and Chanute's Paris
talk of April 1903). The current main corpus removes all of that and much
more, so it is a strict superset of that ablation. The open question has
become harder: can a model that has never heard of a flying machine get from
birds, toy kites and sails to the idea of human flight at all?
