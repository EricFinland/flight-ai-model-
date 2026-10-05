# Contributing

Thanks for helping. This is a research repository: the most important
property of every change is that the comparison between the flight-free
`main` model and the `control` model stays fair and reproducible.

## Set up and run the tests

```bash
pip install -e '.[dev]'     # the package (src/prewright) and pytest
make test                   # same as: python3 -m pytest -q   (about a minute, CPU only)
```

The tests need no data, checkpoints or GPU: they build tiny tokenizers,
models and token files in temporary directories. If a new test needs the real
corpus or a checkpoint, make it skip when the file is missing
(`pytest.skip(...)` or `pytest.mark.skipif(not path.exists(), ...)`), so the
suite stays green on a fresh clone. GitHub Actions runs the suite on every
push and pull request (`.github/workflows/tests.yml`, Python 3.11, CPU torch).

Before opening a pull request, run `make test` and `python3 scripts/<name>.py
--help` for any script you touched.

## Code style

- Python 3.10 or later. Keep it 3.10-compatible: for example, no backslashes
  or reused quote characters inside f-string `{...}` expressions (3.12 only).
- `from __future__ import annotations` at the top of every module, type hints
  on public functions, lines up to 120 characters. `ruff check
  --line-length 120 src scripts tests` should be clean.
- Library code lives in `src/prewright/`; `scripts/` are thin command-line
  wrappers with a usage docstring, `argparse` and a working `--help`.
- Docstrings and comments say *why*, not just what; the design documents in
  `docs/` hold the long reasoning. Update the matching document when you change
  behaviour (docs/PLAN.md is the source of truth for decisions).
- No new dependencies without adding them to `pyproject.toml`; optional ones
  (like matplotlib) go in an extra and are imported where they are used.
- Do not change the checkpoint format (`prewright.load`), the `GPTConfig`
  field names or the model's parameter names: the committed checkpoints in
  `checkpoints/` must keep loading.
- Main and control are always built, tokenized and trained with the same
  code, config, seed and step count. A change that affects only one of them is
  a bug.

## Adding a flight term to the lexicon safely

The lexicon is `TERMS` in [`src/prewright/flight.py`](src/prewright/flight.py);
the policy and its judgment calls are in
[docs/flight-scrub.md](docs/flight-scrub.md), and the red-team audit that finds
missed terms is in [docs/leak-audit.md](docs/leak-audit.md). A new term must
catch the leak **and** leave ordinary English alone (birds flying, flights of
stairs, people's names, myth and fairy tale).

1. **Collect real examples.** Copy the sentence that leaked, exactly as it
   appears in the book, plus two or three innocent sentences that look similar
   (the same word in its harmless sense, a surname, a near-spelling).

2. **Add the term.** Add a `Term(name, _rx(pattern), context=..., note=...)` to
   `TERMS`. Patterns run case-insensitively on `normalize()`d text (accents
   folded, line-break hyphens joined), so write plain lowercase ASCII and use
   `\b` word boundaries; `_SEP` matches "air-ship", "air ship" and "airship".
   For a word that is usually innocent (a common surname, "kite"), set
   `context` to a second pattern that must match in the same paragraph.

3. **Add its anchors.** Every term needs an entry in `TERM_ANCHORS`:
   lowercase literals, one of which appears in *every* possible match (they
   let the scrub skip paragraphs quickly). A missing or wrong anchor silently
   disables the term; `test_every_term_has_anchors_and_examples_hit_their_anchors`
   catches it.

4. **Add tests** in [`tests/test_flight.py`](tests/test_flight.py): the leaking
   sentences go in `REMOVE` (or the `test_audit_leaks_are_caught` list), the
   innocent look-alikes in `KEEP` (or `test_audit_terms_keep_innocent_text`).
   Run `python3 -m pytest -q tests/test_flight.py`.

5. **Measure false positives on real text** before rebuilding anything. On
   the control corpus (which still contains the flight text), see what the new
   term hits:

   ```python
   import json
   from prewright.flight import TERMS, normalize, split_paragraphs
   term = next(t for t in TERMS if t.name == "my_new_term")
   n = 0
   with open("/home/user/data/corpus/control/train.jsonl") as f:
       for i, line in enumerate(f):
           if i == 500: break                      # a sample of books is enough
           for p in split_paragraphs(json.loads(line)["text"]):
               if term.pattern.search(normalize(p)):
                   n += 1
                   if n <= 30: print("---", p[:300])
   print(n, "paragraphs hit")
   ```

   Read the hits. If more than a handful are innocent, tighten the pattern or
   add a `context`, and add the innocent ones to `KEEP`.

6. **Re-scrub and rebuild main.** The control corpus does not change, so
   re-derive main from it without re-downloading, and re-tokenize main on the
   **same book set** as before (the cutoff is in
   `$DATA/tokenized/control/train.meta.json`, key `"cutoff"`):

   ```bash
   make rescrub DATA=$DATA                      # scripts/rescrub_corpus.py
   python3 scripts/tokenize_corpus.py --corpus $DATA/corpus --out $DATA/tokenized \
       --corpora main --train-cutoff <cutoff hex> --workers 4
   ```

   Then retrain main with exactly the control run's config and step count,
   export it, and re-run the evaluation ([docs/reproduce.md](docs/reproduce.md),
   steps 5-10).

7. **Document it**: what leaked, the new term, and its false-positive check,
   in docs/flight-scrub.md (or docs/leak-audit.md for audit finds).

## Other changes

- **Probes and evaluation** (`evals/*.jsonl`, `src/prewright/evaluation/`):
  read [evals/README.md](evals/README.md) first; the probe files are checked by
  `validate_probes` in the tests.
- **Training and scaling**: see [docs/training.md](docs/training.md) and
  [docs/scaling.md](docs/scaling.md). Keep the single-process path
  bit-for-bit deterministic for a given seed; `tests/test_model.py` and
  `tests/test_ddp.py` check resume and multi-process equivalence.

By contributing you agree that your contributions are licensed under the MIT
License in [LICENSE](LICENSE).
