# flight-ai-model-

A language model trained **only on human knowledge from before powered,
controlled flight**. The cutoff is **17 December 1903**, the day of the
Wright brothers' first powered flights. Every training document must have been
published before that date.

## Why

Modern LLMs have read about every invention we'd want to test them on.
A model locked in 1903 lets us ask a question that can't be asked of them:
**given everything the world knew just before flight worked, can a model work
out how to make it work?** The same setup also gives a secondary "Einstein
test": the Michelson–Morley puzzle was public, but relativity (1905) was not.

## What the model may and may not know

| May know (published before 17 Dec 1903) | May not know |
|---|---|
| Ancient history, philosophy, mathematics; Newtonian mechanics, thermodynamics, Maxwell's electromagnetism | Special relativity (1905), general relativity (1915) |
| Steam engines, railways, steamships, motor cars (Daimler, Benz, the Ford Motor Company from June 1903) | Quantum mechanics, the Bohr atom, the atomic nucleus (1911) |
| Telegraph, telephone, Marconi's transatlantic wireless (1901) | The Wright Flyer and airplanes as a working technology |
| X-rays (1895), the electron (1897), radium (1898), Planck's quanta (1900) | The Russo-Japanese War (Feb 1904 onward), WWI, everything later |
| Balloons, airships (Zeppelin 1900, Santos-Dumont 1901), gliders (Cayley, Lilienthal, Chanute, the Wrights' 1900–02 gliding) | Radio broadcasting, computers, nuclear physics, antibiotics |
| **Langley's aerodrome failures of 7 Oct and 8 Dec 1903**: the press believed flight was decades away | The Panama Canal's opening (1914), though it knows of Panama's independence (Nov 1903) |

Edge cases that are *in*, because they predate the cutoff: Gustave
Whitehead's claimed flight of 1901, Wilbur Wright's 1901 and 1903 lectures on
gliding, Poincaré on relative motion and simultaneity, and the word
*aeroplane* (used of lifting surfaces since the 1870s).

## Plan

| Phase | Work | Status |
|---|---|---|
| 0. Rules | Cutoff definition, date parsing, anachronism filter, probe set | **Started** (this repo) |
| 1. Corpus | Download sources ([docs/data-sources.md](docs/data-sources.md)), normalize to JSONL `{id, date, source, text}`, filter, deduplicate, score OCR quality, measure token counts | Next |
| 2. Tokenizer | Train a BPE tokenizer (about 32k vocabulary) on the filtered corpus only | |
| 3. Pretraining | A ~125M-parameter pilot to check the pipeline, then a model sized to the token count (likely 1–3B) | |
| 4. Leak audit | Run `evals/probes.jsonl`, trace every `should_not_know` failure back to its documents, tighten the filters, retrain if needed | |
| 5. Chat tuning | Period-safe instruction data only ([docs/leakage.md §6](docs/leakage.md)) | |
| 6. Discovery experiment | `design_flyer` and `ether_drift` probes on the main model **and** the no-Wright ablation | |

**Rough compute** (training compute ≈ 6 × parameters × tokens): a 1B model on
20B tokens takes about 1.2×10²⁰ FLOPs. At 40% utilization that is roughly
**80–90 H100-hours**. A 3B model on 60B tokens takes about 9× that. The 125M
pilot fits on a single GPU in a few hours.

## Repository layout

```
config/cutoff.yaml        cutoff date and filter thresholds
src/prewright/dates.py    messy catalogue dates -> conservative (earliest, latest) range
src/prewright/leakage.py  strip digitization boilerplate; reject or flag anachronisms
src/prewright/filter_cli.py  JSONL in -> keep / review / reject JSONL out
evals/probes.jsonl        should_know / should_not_know / discovery probes
evals/README.md           how to read results; scoring rubric for the discovery probe
docs/data-sources.md      where pre-1904 text comes from and the traps in each source
docs/leakage.md           every way modern knowledge can leak in, with mitigations
```

## Usage

```bash
pip install -e '.[dev]'
pytest

# Each input line: {"id": "...", "date": "1887", "source": "...", "text": "..."}
prewright-filter --in raw.jsonl --out-dir filtered/ --config config/cutoff.yaml
```

The filter sorts records into `keep.jsonl` (train), `review.jsonl` (hold
back for a human), and `reject.jsonl`. Each record carries the reason for its
decision. Doubt always counts against a document: a missing date, an
uncertain date, or a date range that crosses the cutoff all mean rejection.
