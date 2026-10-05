# Data sources

Every source goes through the same gate: a publication date we trust, with the
*latest* possible date before 17 Dec 1903 (`src/prewright/dates.py`). The
content filter then runs on top (`src/prewright/leakage.py`). Sizes and
licences change, so check each one before downloading at scale. The notes
below cover what matters for *this* project.

## Primary (good dates, large volume)

| Source | What it is | Date quality | Notes |
|---|---|---|---|
| **Chronicling America** (Library of Congress) | Digitized US newspapers, roughly 1770s–1960s, with page-level OCR | Excellent: every issue is dated | The best record of what the public knew day by day, including the Langley failures and the Panama revolution. OCR quality varies a lot. |
| **American Stories** (Dell Research / Harvard, on Hugging Face) | Article-level text extracted from Chronicling America scans, grouped by year | Excellent (issue date) | Easier to use than raw page OCR. Take the pre-1904 years and still filter by full issue date for 1903. |
| **Institutional Books 1.0** (Harvard Library / Institutional Data Initiative, on Hugging Face) | About a million public-domain volumes from Harvard's Google Books scans, with catalogue metadata | Good (MARC dates, many with `u` digits, which `dates.py` handles) | Probably the largest single book source. Read the access terms first. Reprints cause trouble: the catalogue date is the scanned edition's, so a 1920 reprint of an 1850 book is (correctly) excluded. |
| **British Library digitised books** (on Hugging Face) | 19th-century books digitized with Microsoft | Good | British balance for the American newspapers. |
| **HathiTrust** public-domain full text | Millions of volumes; bulk text through the HathiTrust Research Center | Good (MARC) | Large overlap with the sources above, so deduplicate. |

## Secondary (excellent content, extra care needed)

| Source | Watch out for |
|---|---|
| **Project Gutenberg** | The metadata has no original publication date. Use the author's death year (before 1904) as the eligibility test, then strip the header and footer (done automatically) and look for editors' introductions and notes. Some Gutenberg texts follow *later* annotated editions. |
| **Wikisource** | Clean transcriptions, but with editors' notes and links. The famous *Encyclopaedia Britannica* on Wikisource is the **11th edition (1910–11), which is excluded**. Use the 9th edition (1875–89) and the 10th edition's supplementary volumes (1902–03). Those are almost a snapshot of knowledge at the cutoff. |
| **Biodiversity Heritage Library**, **Royal Society** archive, early **Nature** (from 1869), *Annalen der Physik*, *Comptes rendus* | Scientific literature. Use each article's own date, not the journal run's. |
| **USPTO patents** before 1904 | Gives engineering depth. OCR'd text has modern patent-office headers that need stripping. |
| **Internet Archive** texts | Huge, but dates are unreliable. Use only with a matching catalogue record from another source. |

## Classics and translations

Ancient texts are fine, but **the translation must also predate the cutoff**.
Jowett's Plato (1871), the Bohn's Classical Library volumes, and Victorian
editions of Homer, Thucydides, Euclid and others are all eligible. The Loeb
Classical Library started in 1912 and is excluded. So is every modern
translation, since modern scholarship shows up in their notes and word choices.
Greek and Latin original texts are an option for a multilingual phase, if they
come from period editions.

## Non-English (later phase)

To cover European science and engineering (French aviation in particular), the
period's fullest record is in French and German. Sources include Gallica
(BnF), Europeana Newspapers, ANNO (Austrian newspapers), Delpher (Dutch), and
the Deutsches Textarchiv. A dated pre-1904 run of *L'Aérophile* would be
especially valuable, and also especially exposed to Wright-related material
from 1903 (see the ablation in `leakage.md`).

## How much text is there?

Measure this in phase 1 rather than guessing. As a rough expectation: after
date filtering, deduplication and OCR-quality filtering, the English sources
above probably give **tens of billions of tokens**, mostly newspapers and
books. That is enough for a model of roughly 1–3B parameters at a
Chinchilla-like token-to-parameter ratio. Going further means repeating data
for several epochs (tolerable for a few epochs) or adding other languages.
