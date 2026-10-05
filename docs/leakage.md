# Leakage: how post-1903 knowledge sneaks in, and how we keep it out

A model trained on "pre-1904 text" is only interesting if it truly knows
nothing later. Leakage rarely comes from an obviously modern document. It
comes through the cracks below, listed roughly from most to least likely.

## 1. Wrong or misleading dates

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

## 2. Modern material wrapped around old text

Editors' introductions, footnotes, translators' notes, "Further reading"
lists, and Gutenberg or scanner boilerplate. `leakage.py` strips
digitization boilerplate paragraph by paragraph and rejects whole documents
with hard anachronisms (an introduction naming Einstein usually means more
modern material nearby).

## 3. Anachronism filter limits

The rule lists are aimed at precision, so they catch only obvious cases. The
plan for better recall:

- **Classifier pass.** Have a modern LLM label a sample of documents as
  "contains post-1903 knowledge: yes/no/where", then train a small classifier
  on those labels and run it over the full corpus. *Using a modern model to
  judge text is safe. Using one to write or rewrite training text is not*
  (see 5 and 6).
- **Leak tracing.** Every `should_not_know` probe failure gets traced to its
  source documents (n-gram search over the shards), and the rule lists grow.
- **Review queue.** `review.jsonl` gets human spot checks. Don't train on it
  by default.

Pre-1903 words that look modern and must *not* be filtered: *aeroplane*
(1870s, for a lifting surface), *television* (1900), *electron* (1891),
*relativity* (as a philosophy term), *Bolshevik* (August 1903), *Lenin*
(pen name from 1901), *Ford Motor Company* (June 1903), Planck's quanta
(1900). The tests in `tests/test_leakage.py` lock these in.

## 4. OCR correction

Fixing OCR noise with a modern LLM is tempting, but it is a leakage channel:
the model "corrects" period spelling and fills gaps with modern knowledge.
Use non-LLM methods (dictionary or n-gram correction built from the corpus
itself), or leave the noise in, which models handle well. Keep period
spellings (*to-day*, *connexion*, the long ſ written as s).

## 5. Tokenizer

Train the tokenizer on the filtered corpus itself. A modern tokenizer carries
little factual knowledge, but its vocabulary encodes modern word frequencies,
and using our own also makes for a cleaner experiment.

## 6. Instruction tuning, the biggest trap after pretraining

A chat model needs instruction data. Writing that data with a modern LLM
leaks modern framing and facts straight into the model. Options, safest
first:

1. **Period question-and-answer text.** Catechisms, "Notes and Queries",
   newspaper answers-to-correspondents columns, *Enquire Within Upon
   Everything* (1856), exam papers with model answers, Socratic dialogues.
   Reformat these into chat turns mechanically.
2. **Grounded synthetic data.** A modern LLM writes only the *question*, about
   a specific corpus passage. The answer is pulled from that passage or
   written by our own base model. Every answer goes through the anachronism
   filter.
3. **Self-instruct with the base model**, then preference tuning judged
   against corpus passages.

Never use modern instruction datasets (Alpaca, ShareGPT and the like).

## 7. Evaluation and prompting leakage

A modern prompt can smuggle in knowledge ("As we know from the Wright
brothers' flight..."). Probes in `evals/probes.jsonl` are written to mention
only facts available before the cutoff. When grading discovery probes, a
modern LLM is fine as a grader but must never write the probe text.

## No-Wright ablation corpus

The Wrights published about their gliders before the cutoff: Wilbur's
lectures to the Western Society of Engineers (September 1901, June 1903) and
Chanute's Paris talk of April 1903 (printed in *L'Aérophile*). A model trained
on the main corpus can therefore *recall* parts of their solution, such as
wing warping. To test real *discovery*, also train a smaller ablation model
with every document that mentions the Wrights, their gliders or wing warping
removed. Then compare the two on the `design_flyer` probe. The ablation is a
second filter pass (`wright|wing.warping|warping of the wings|Kitty Hawk`),
not a separate pipeline.
