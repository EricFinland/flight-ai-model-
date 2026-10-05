# Data sources

## What this build used: PG-19 only

The first build uses one source, **PG-19** (DeepMind's set of 28,752 Project
Gutenberg books published before 1919, at `gs://deepmind-gutenberg`). It was
the only large pre-1903 source reachable from the build machine: its network
policy blocked Hugging Face, Project Gutenberg itself, the Library of Congress,
the Internet Archive and Wikimedia, while Google Cloud Storage was open.
PG-19's `metadata.csv` gives a `publication_date` (a year) for every book.

- Kept: books dated **1902 or earlier** (17,350 books, about 6.8 GB;
  `scripts/download_pg19.sh`). A year-only 1903 could be after 17 Dec 1903,
  so it is excluded.
- Then the anachronism filter, an English-language test and, for `main`
  only, the flight scrub (`scripts/build_corpus.py`; statistics in
  [flight-scrub.md](flight-scrub.md#corpus-statistics)).
- Result: about 6.6 billion characters (about 1.7B tokens with the project
  tokenizer) of pre-1903 English books in `main`. The CPU pilot trains on a
  150M-token slice of it.

PG-19's dates are catalogue dates. Gutenberg texts sometimes follow a later
edition, and the date can be a reprint's or simply wrong; the anachronism
filter catches the obvious cases (see [leakage.md](leakage.md), section 1).

## Sources for scaling up

Everything below is for later builds, when the network allows it and a bigger
model needs more text. None of it was used here. Every source goes through the
same gates: a publication date we trust, with the *latest* possible date
before 17 Dec 1903, and in practice 1902 or earlier
(`src/prewright/dates.py`); the anachronism filter (`src/prewright/leakage.py`);
and, for the flight-free corpus, **the flight scrub** (`src/prewright/flight.py`).
Newspapers in particular reported every balloon ascent, airship scheme and
gliding accident, so they will lose more text to the scrub than books did.
Sizes and licences change, so check each one before downloading at scale.

### Primary (good dates, large volume)

| Source | What it is | Date quality | Notes |
|---|---|---|---|
| **Chronicling America** (Library of Congress) | Digitized US newspapers, roughly 1770s–1960s, with page-level OCR | Excellent: every issue is dated | The best record of what the public knew day by day. Also full of balloon ascents and airship schemes, all of which the scrub must remove. OCR quality varies a lot, which makes the lexicon-based scrub less reliable. |
| **American Stories** (Dell Research / Harvard, on Hugging Face) | Article-level text extracted from Chronicling America scans, grouped by year | Excellent (issue date) | Easier to use than raw page OCR. Take the pre-1904 years and still filter by full issue date for 1903. |
| **Institutional Books 1.0** (Harvard Library / Institutional Data Initiative, on Hugging Face) | About a million public-domain volumes from Harvard's Google Books scans, with catalogue metadata | Good (MARC dates, many with `u` digits, which `dates.py` handles) | Probably the largest single book source. Read the access terms first. Reprints cause trouble: the catalogue date is the scanned edition's, so a 1920 reprint of an 1850 book is (correctly) excluded. |
| **British Library digitised books** (on Hugging Face) | 19th-century books digitized with Microsoft | Good | British balance for the American newspapers. |
| **HathiTrust** public-domain full text | Millions of volumes; bulk text through the HathiTrust Research Center | Good (MARC) | Large overlap with the sources above, so deduplicate. |

### Secondary (excellent content, extra care needed)

| Source | Watch out for |
|---|---|
| **Project Gutenberg** (beyond PG-19) | The metadata has no original publication date. Use the author's death year (before 1903) as the eligibility test, then strip the header and footer (done automatically) and look for editors' introductions and notes. Some Gutenberg texts follow *later* annotated editions. PG-19 adds a catalogue year, which this build used. |
| **Wikisource** | Clean transcriptions, but with editors' notes and links. The famous *Encyclopaedia Britannica* on Wikisource is the **11th edition (1910–11), which is excluded**. Use the 9th edition (1875–89) and those of the 10th edition's supplementary volumes printed in 1902. Those are almost a snapshot of knowledge at the cutoff. Their articles on aeronautics and balloons go, by the whole-article equivalent of the book rule. |
| **Biodiversity Heritage Library**, **Royal Society** archive, early **Nature** (from 1869), *Annalen der Physik*, *Comptes rendus* | Scientific literature. Use each article's own date, not the journal run's. |
| **USPTO patents** before 1904 | Gives engineering depth. OCR'd text has modern patent-office headers that need stripping. |
| **Internet Archive** texts | Huge, but dates are unreliable. Use only with a matching catalogue record from another source. |

### Classics and translations

Ancient texts are fine, but **the translation must also predate the cutoff**.
Jowett's Plato (1871), the Bohn's Classical Library volumes, and Victorian
editions of Homer, Thucydides, Euclid and others are all eligible. The Loeb
Classical Library started in 1912 and is excluded. So is every modern
translation, since modern scholarship shows up in their notes and word choices.
Greek and Latin original texts are an option for a multilingual phase, if they
come from period editions.

### Non-English (later phase)

To cover European science and engineering, the period's fullest record is in
French and German. Under the current policy that also means a flight lexicon in
each language: French and German writers led ballooning and early aviation, so
*ballon*, *aérostat*, *aéronaute*, *Luftschiff*, *Flugmaschine*, *Gleitflug*
and the rest need the same treatment as their English counterparts. Sources include Gallica
(BnF), Europeana Newspapers, ANNO (Austrian newspapers), Delpher (Dutch), and
the Deutsches Textarchiv. Aeronautical journals such as *L'Aérophile* are excluded
from the flight-free corpus outright, though they would suit the control.

## How much text is there?

Measured for PG-19: about 1.7B tokens of pre-1903 English books after all
filters, which is enough for compute-optimal training of a model of about
85M parameters (20 tokens per parameter). `configs/gpu_125m.yaml` trains on
about 1.5 epochs of it and `configs/gpu_1b.yaml` on about 5 ([scaling.md](scaling.md)).

For the sources above, the rough expectation is **tens of billions of tokens**
of English after date filtering, deduplication and OCR-quality filtering,
mostly newspapers and books. That is enough for a model of roughly 1–3B
parameters at a Chinchilla-like ratio. Going further means repeating data for
several epochs (tolerable for a few) or adding other languages.
