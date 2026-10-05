#!/usr/bin/env python3
"""Fill the ``<!-- RESULT:key -->...<!-- /RESULT -->`` placeholders in README.md and MODEL_CARD.md.

    python3 scripts/fill_readme_results.py                       # read evals/results/{eval,curve}.json, edit in place
    python3 scripts/fill_readme_results.py --check               # only list keys and what they would become
    python3 scripts/fill_readme_results.py --eval X.json --curve Y.json --files README.md --out-dir /tmp/x

A placeholder is a pair of HTML comments around its current value::

    <!-- RESULT:surprisal_ratio -->TBD<!-- /RESULT -->

Everything between the markers is replaced, so the script can be re-run after a
new evaluation (it never touches text outside the markers). A key with no data
becomes ``TBD`` and is reported as a warning; an unknown key is left as ``TBD``
with a warning too. Numbers are formatted with the same helpers as
``evals/results/REPORT.md`` (``prewright.evaluation.report``).

The script refuses to write numbers from a ``--fake`` evaluation into the
repository's own README.md/MODEL_CARD.md (use ``--out-dir`` for a test), and
warns if the evaluation was a ``--quick`` run.

Keys
----
Inputs: ``E`` = evals/results/eval.json (scripts/evaluate.py, schema in
src/prewright/evaluation/run.py and report.py); ``C`` = evals/results/curve.json
(scripts/eval_curve.py). "Factor" = exp(-mean diff), where diff = gap(main) -
gap(control) and gap = log p(flight wording) - log p(everyday wording): how many
times less likely, relative to the everyday wording, the flight-free model finds
the flight wording than the control does.

Run and models
  eval_mode                   full / quick / FAKE, with the evaluation timestamp and seed (E meta)
  val_loss_main, val_loss_control
                              final validation loss, nats/token, own val file (E checkpoint meta, else C)
Paired surprisal probes (E paired)
  flight_pairs_n, control_pairs_n   number of flight / non-flight control pairs
  flight_words_n              number of word clusters the flight pairs turn on
  surprisal_ratio             factor over flight pairs (geometric mean), e.g. "12.3×"
  surprisal_ratio_ci          95% CI of the factor, resampling word clusters, "4.1×–30.2×"
  surprisal_ratio_median      median factor over flight pairs
  surprisal_verdict           "beyond ..." / "not distinguishable from ..." the control-pair noise
                              (the report's headline criterion)
  frac_main_more_surprised    share of flight pairs where main is the more surprised, "78% of 171"
  control_pairs_ratio, control_pairs_ratio_ci
                              factor and item-bootstrap CI on the non-flight control pairs
  flight_vs_control           cluster-level flight-pair diff minus control-pair diff, nats, CI and p
  noise_floor_words           flight words beyond the 2.5th percentile of control wordings, "9 of 21"
  item:<probe id>             factor for one pair, "12.3×" (below 1 means main is *less* surprised)
  item_text:<probe id>        the same as a phrase: "12.3× less likely" or "2.0× more likely"
Held-out perplexity (E perplexity)
  flight_pieces_n             number of held-out flight pieces
  ppl_flight_main, ppl_flight_control, ppl_flight_ratio, ppl_flight_ratio_ci
  ppl_ordinary_main, ppl_ordinary_control, ppl_ordinary_ratio, ppl_ordinary_ratio_ci   (length-matched)
  ppl_full_main, ppl_full_control, ppl_full_ratio, ppl_full_ratio_ci                   (full windows)
  flight_excess_nats          extra main-minus-control loss on flight text beyond ordinary text,
                              nats/token with book-clustered CI
  flight_word_bits_main, flight_word_bits_control
                              mean surprisal (bits) of the tokens of flight words inside the flight pieces
Knowledge cloze (E knowledge)
  knowledge_general_n, knowledge_flight_n
  knowledge_general_main, knowledge_general_control, knowledge_flight_main, knowledge_flight_control
                              "41/65 (63%)"
Generation (E generation)
  gen_flight_main, gen_flight_control
                              flight-eliciting completions that use a flight-lexicon term, "1/36"
  sample_main:<prompt id>, sample_control:<prompt id>
                              first sample's text for that prompt, one line, at most --sample-chars
Tokenizer and corpus (E tokenizer, corpus)
  tok_mean_flight, tok_mean_everyday, tok_n_flight, tok_n_everyday
  audit_main_hits, audit_control_hits
                              flight-lexicon hits in each training corpus, with the scope audited in brackets
Over training (C)
  curve_steps_n               number of matched steps scored
  curve_first_significant_step   "step 300 (4.9M tokens)", or "no step scored"
  curve_final_factor          "12.3× at step 850"
Corpus build (S = <data>/corpus/stats.json, M = <data>/corpus/manifest.csv,
T = <data>/tokenized/{main,control}/train.meta.json; these change when the corpus is
re-scrubbed, so the README carries the current values between the markers)
  corpus_books_dated          PG-19 books dated 1902 or earlier
  corpus_flight_books         books dropped whole from main as flight books
  corpus_flight_books_title, corpus_flight_books_density   ... by title / by hit density (M)
  corpus_paragraphs_removed   paragraphs cut from kept books (hits, neighbours, bridged gaps)
  corpus_hit_paragraphs       paragraphs with a lexicon hit, kept and dropped books
  corpus_books_with_cuts      kept books with at least one paragraph cut (M)
  corpus_flight_eval          held-out flight paragraphs from validation books
  corpus_main_train_books, corpus_control_train_books, corpus_main_val_books, corpus_control_val_books
  corpus_main_smaller_pct     how much smaller main/train is than control/train, in characters
  corpus_main_chars_b         main/train characters, billions
  term:<lexicon term name>    matches of that term over all books that passed the other filters
  slice_main_books, slice_main_tokens, slice_control_books, slice_control_tokens
                              the capped training token files (T), tokens in millions
  slice_extra_flight_books    control slice books minus main slice books
Block
  facts                       the evaluation report's headline facts, as a bullet list
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prewright.evaluation.report import _p, _times, headline_facts  # noqa: E402

PLACEHOLDER = re.compile(r"<!-- RESULT:([A-Za-z0-9_:.\-]+) -->(.*?)<!-- /RESULT -->", re.DOTALL)
TBD = "TBD"


class Missing(Exception):
    """The data for a key is not in the results."""


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            raise Missing("/".join(path))
        d = d[k]
    return d


def _num(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        raise Missing("number")
    return f"{x:,.{nd}f}"


def _dash_range(lo: str, hi: str) -> str:
    return f"{lo}–{hi}"


def _ci_nats(ci: dict, nd: int = 2) -> str:
    return f"{ci['estimate']:.{nd}f} nats (95% CI {ci['low']:.{nd}f} to {ci['high']:.{nd}f})"


def _frac(correct: int, n: int) -> str:
    return f"{correct}/{n} ({100 * correct / n:.0f}%)" if n else TBD


def _one_line(text: str, limit: int) -> str:
    s = " ".join(text.split())
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + " …"
    # Keep markdown inert: no emphasis, tables, HTML or code from model output.
    for ch in "\\*_`|<>[]#":
        s = s.replace(ch, "\\" + ch)
    return s


class Results:
    def __init__(self, ev: dict | None, curve: dict | None, sample_chars: int, data: Path | None = None):
        self.ev, self.curve, self.sample_chars = ev or {}, curve or {}, sample_chars
        self.data = data
        self._stats: dict | None = None
        self._manifest: dict | None = None

    # -- corpus files --------------------------------------------------------
    def stats(self) -> dict:
        if self._stats is None:
            p = self.data / "corpus" / "stats.json" if self.data else None
            if not p or not p.exists():
                raise Missing(f"{p}")
            self._stats = json.loads(p.read_text(encoding="utf-8"))
        return self._stats

    def manifest(self) -> dict:
        if self._manifest is None:
            import csv
            p = self.data / "corpus" / "manifest.csv" if self.data else None
            if not p or not p.exists():
                raise Missing(f"{p}")
            cuts = title = density = 0
            with p.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["decision"] == "kept" and int(r["flight_paragraphs_removed"] or 0) > 0:
                        cuts += 1
                    elif r["decision"] == "dropped_flight_book":
                        if "title" in (r.get("reason") or "").lower():
                            title += 1
                        else:
                            density += 1
            self._manifest = {"books_with_cuts": cuts, "by_title": title, "by_density": density}
        return self._manifest

    def tokmeta(self, corpus: str) -> dict:
        p = self.data / "tokenized" / corpus / "train.meta.json" if self.data else None
        if not p or not p.exists():
            raise Missing(f"{p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def stat(self, *path: str) -> int:
        return _get(self.stats(), *path)

    # -- helpers ---------------------------------------------------------
    def paired(self, kind: str) -> dict:
        return _get(self.ev, "paired", "kinds", kind)

    def item(self, pid: str) -> dict:
        for it in _get(self.ev, "paired", "items"):
            if it["id"] == pid:
                return it
        raise Missing(f"item {pid}")

    def ppl(self, name: str) -> dict:
        return _get(self.ev, "perplexity", "sets", name)

    def know(self, kind: str) -> dict:
        return _get(self.ev, "knowledge", "kinds", kind)

    def gen_row(self, pid: str) -> dict:
        for r in _get(self.ev, "generation", "rows"):
            if r["id"] == pid:
                return r
        raise Missing(f"prompt {pid}")

    def val_loss(self, model: str) -> str:
        try:
            v = _get(self.ev, "meta", "models", model, "checkpoint_meta", "val_loss")
        except Missing:
            v = _get(self.curve, "meta", "loss", model, "last_val_loss")
        return _num(float(v), 3)

    def verdict(self) -> str:
        d = self.paired("flight")["difference"]
        cl = d.get("mean_diff_clustered", d["mean_diff"])
        vs = _get(self.ev, "paired").get("flight_vs_control_pairs", {}).get("cluster_level_diff")
        sig_cl = cl["high"] < 0
        sig_vs = vs is None or vs["high"] < 0
        if sig_cl and sig_vs:
            return "beyond the ordinary model-to-model differences measured on the non-flight control pairs"
        return "not distinguishable from the ordinary model-to-model differences measured on the non-flight " \
               "control pairs"

    def eval_mode(self) -> str:
        meta = _get(self.ev, "meta")
        fake = any(m.get("path") == "(fake)" for m in meta.get("models", {}).values())
        mode = "FAKE smoke test (not real results)" if fake else "quick run (first look only)" if meta.get(
            "quick") else "full run"
        return f"{mode}, {meta.get('timestamp', '?')}, seed {meta.get('seed', '?')}"

    def curve_first(self) -> str:
        steps = _get(self.curve, "steps")
        if not steps:
            raise Missing("curve steps")
        fs = _get(self.curve, "meta").get("first_significant_step")
        if fs is None:
            return f"no step scored (of {len(steps)}, up to step {steps[-1]['step']})"
        rec = next((r for r in steps if r["step"] == fs), {})
        tok = rec.get("tokens")
        return f"step {fs}" + (f" ({tok / 1e6:.1f}M tokens)" if tok else "")

    def curve_final(self) -> str:
        steps = _get(self.curve, "steps")
        if not steps:
            raise Missing("curve steps")
        last = steps[-1]
        return f"{_times(_get(last, 'flight', 'factor'))} at step {last['step']}"

    def audit(self, corpus: str) -> str:
        c = _get(self.ev, "corpus", corpus)
        a = _get(c, "audit")
        if c.get("max_books"):
            part = f"first {c['max_books']:,} books only"
        elif c.get("trained_books_only"):
            part = f"the {c['books']:,} books in the training token file"
        else:
            part = f"all {c['books']:,} books of {corpus}/train.jsonl"
        return f"{a['flight_hits']:,} ({part})"

    def facts(self) -> str:
        if not self.ev:
            raise Missing("eval.json")
        facts = headline_facts(self.ev)
        if not facts:
            raise Missing("facts")
        return "\n" + "\n".join(f"- {f}" for f in facts) + "\n"

    # -- the key table -----------------------------------------------------
    def resolver(self, key: str) -> Callable[[], str] | None:
        lr = lambda kind: self.paired(kind)["difference"]["likelihood_ratio"]  # noqa: E731
        fixed: dict[str, Callable[[], str]] = {
            "eval_mode": self.eval_mode,
            "val_loss_main": lambda: self.val_loss("main"),
            "val_loss_control": lambda: self.val_loss("control"),
            "flight_pairs_n": lambda: str(self.paired("flight")["difference"]["n"]),
            "control_pairs_n": lambda: str(self.paired("control")["difference"]["n"]),
            "flight_words_n": lambda: str(self.paired("flight")["difference"]["n_clusters"]),
            "surprisal_ratio": lambda: _times(lr("flight")["geometric_mean"]),
            "surprisal_ratio_ci": lambda: _dash_range(_times(lr("flight")["clustered_low"]),
                                                      _times(lr("flight")["clustered_high"])),
            "surprisal_ratio_median": lambda: _times(lr("flight")["median"]),
            "surprisal_verdict": self.verdict,
            "frac_main_more_surprised": lambda: (
                f"{100 * self.paired('flight')['difference']['frac_main_lower']:.0f}% of "
                f"{self.paired('flight')['difference']['n']}"),
            "control_pairs_ratio": lambda: _times(lr("control")["geometric_mean"]),
            "control_pairs_ratio_ci": lambda: _dash_range(_times(lr("control")["low"]), _times(lr("control")["high"])),
            "flight_vs_control": lambda: (
                _ci_nats(_get(self.ev, "paired", "flight_vs_control_pairs", "cluster_level_diff"))
                + f"; permutation p {_p(_get(self.ev, 'paired', 'flight_vs_control_pairs', 'cluster_level_permutation_p'))}"),
            "noise_floor_words": lambda: (
                f"{_get(self.ev, 'paired', 'noise_floor', 'flight_clusters_below_q025')} of "
                f"{_get(self.ev, 'paired', 'noise_floor', 'n_flight_clusters')} words "
                f"({_get(self.ev, 'paired', 'noise_floor', 'flight_pairs_below_q025')} of "
                f"{self.paired('flight')['difference']['n']} pairs)"),
            "flight_pieces_n": lambda: str(self.ppl("flight")["main"]["pieces"]),
            "flight_excess_nats": lambda: (
                _ci_nats(_get(self.ev, "perplexity", "flight_minus_ordinary_excess", "diff_book_clustered"), 3)
                .replace(" nats", " nats/token")),
            "flight_word_bits_main": lambda: _num(_get(self.ev, "perplexity", "flight_word_tokens",
                                                       "surprisal_bits_main"), 1),
            "flight_word_bits_control": lambda: _num(_get(self.ev, "perplexity", "flight_word_tokens",
                                                          "surprisal_bits_control"), 1),
            "gen_flight_main": lambda: self._gen("main"),
            "gen_flight_control": lambda: self._gen("control"),
            "tok_mean_flight": lambda: _num(_get(self.ev, "tokenizer", "mean_tokens_flight"), 1),
            "tok_mean_everyday": lambda: _num(_get(self.ev, "tokenizer", "mean_tokens_everyday"), 1),
            "tok_n_flight": lambda: str(_get(self.ev, "tokenizer", "n_flight")),
            "tok_n_everyday": lambda: str(_get(self.ev, "tokenizer", "n_everyday")),
            "audit_main_hits": lambda: self.audit("main"),
            "audit_control_hits": lambda: self.audit("control"),
            "curve_steps_n": lambda: str(len(_get(self.curve, "steps"))),
            "curve_first_significant_step": self.curve_first,
            "curve_final_factor": self.curve_final,
            "facts": self.facts,
            "corpus_books_dated": lambda: f"{self.stat('books_dated_le_1902'):,}",
            "corpus_flight_books": lambda: f"{self.stat('decisions', 'dropped_flight_book'):,}",
            "corpus_flight_books_title": lambda: f"{self.manifest()['by_title']:,}",
            "corpus_flight_books_density": lambda: f"{self.manifest()['by_density']:,}",
            "corpus_paragraphs_removed": lambda: f"{self.stat('flight_paragraphs_removed_from_kept_books'):,}",
            "corpus_hit_paragraphs": lambda: f"{self.stat('flight_hit_paragraphs'):,}",
            "corpus_books_with_cuts": lambda: f"{self.manifest()['books_with_cuts']:,}",
            "corpus_flight_eval": lambda: f"{self.stat('flight_eval_paragraphs'):,}",
            "corpus_main_train_books": lambda: f"{self.stat('books', 'main:train'):,}",
            "corpus_control_train_books": lambda: f"{self.stat('books', 'control:train'):,}",
            "corpus_main_val_books": lambda: f"{self.stat('books', 'main:val'):,}",
            "corpus_control_val_books": lambda: f"{self.stat('books', 'control:val'):,}",
            "corpus_main_smaller_pct": lambda: (
                f"{100 * (1 - self.stat('chars', 'main:train') / self.stat('chars', 'control:train')):.2f}%"),
            "corpus_main_chars_b": lambda: f"{self.stat('chars', 'main:train') / 1e9:.2f}",
            "slice_main_books": lambda: f"{self.tokmeta('main')['books']:,}",
            "slice_control_books": lambda: f"{self.tokmeta('control')['books']:,}",
            "slice_main_tokens": lambda: f"{self.tokmeta('main')['tokens'] / 1e6:.1f}M",
            "slice_control_tokens": lambda: f"{self.tokmeta('control')['tokens'] / 1e6:.1f}M",
            "slice_extra_flight_books": lambda: f"{self.tokmeta('control')['books'] - self.tokmeta('main')['books']:,}",
        }
        for setname, short in (("flight", "flight"), ("ordinary_matched", "ordinary"), ("ordinary_full", "full")):
            for model in ("main", "control"):
                fixed[f"ppl_{short}_{model}"] = (
                    lambda s=setname, m=model: _num(_get(self.ppl(s), m, "perplexity", "estimate"), 1))
            fixed[f"ppl_{short}_ratio"] = lambda s=setname: _num(_get(self.ppl(s), "perplexity_ratio", "estimate"), 3)
            fixed[f"ppl_{short}_ratio_ci"] = lambda s=setname: _dash_range(
                _num(_get(self.ppl(s), "perplexity_ratio", "low"), 3),
                _num(_get(self.ppl(s), "perplexity_ratio", "high"), 3))
        for kind in ("general", "flight"):
            fixed[f"knowledge_{kind}_n"] = lambda k=kind: str(self.know(k)["main"]["n"])
            for model in ("main", "control"):
                fixed[f"knowledge_{kind}_{model}"] = lambda k=kind, m=model: _frac(
                    self.know(k)[m]["correct"], self.know(k)[m]["n"])
        if key in fixed:
            return fixed[key]
        prefix, _, arg = key.partition(":")
        if prefix == "term" and arg:
            return lambda: f"{self.stat('top_flight_terms', arg):,}"
        if prefix == "item" and arg:
            return lambda: _times(math.exp(-self.item(arg)["diff"]))
        if prefix == "item_text" and arg:
            def item_text() -> str:
                r = math.exp(-self.item(arg)["diff"])
                return f"{_times(r)} less likely" if r >= 1 else f"{_times(1 / r)} *more* likely"
            return item_text
        if prefix in ("sample_main", "sample_control") and arg:
            model = prefix.split("_", 1)[1]
            return lambda: _one_line(_get(self.gen_row(arg), "samples", model)[0]["text"], self.sample_chars)
        return None

    def _gen(self, model: str) -> str:
        c = _get(self.ev, "generation", "summary", "categories", "flight_eliciting", model)
        return f"{c['with_flight_terms']}/{c['n']}"


def fill_text(text: str, res: Results, warnings: list[str], filled: dict[str, str], src: str) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        fn = res.resolver(key)
        if fn is None:
            warnings.append(f"{src}: unknown key {key!r}")
            value = TBD
        else:
            try:
                value = fn()
            except (Missing, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                warnings.append(f"{src}: no data for {key!r} ({type(e).__name__}: {e})")
                value = TBD
        if key == "facts" and value == TBD:
            value = "\n" + TBD + "\n"
        filled[key] = value
        return f"<!-- RESULT:{key} -->{value}<!-- /RESULT -->"

    return PLACEHOLDER.sub(sub, text)


def _load(path: Path, what: str, warnings: list[str]) -> dict | None:
    if not path.exists():
        warnings.append(f"{path} not found: every {what} key becomes {TBD}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=REPO / "evals" / "results" / "eval.json")
    ap.add_argument("--curve", type=Path, default=REPO / "evals" / "results" / "curve.json")
    ap.add_argument("--files", type=Path, nargs="+", default=[REPO / "README.md", REPO / "MODEL_CARD.md"])
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"),
                    help="data root holding corpus/stats.json, corpus/manifest.csv and tokenized/ (corpus keys)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the filled files here instead of editing them in place")
    ap.add_argument("--check", action="store_true", help="print key -> value, write nothing")
    ap.add_argument("--sample-chars", type=int, default=220, help="max characters of a quoted sample")
    ap.add_argument("--allow-fake", action="store_true",
                    help="allow numbers from a --fake evaluation in the files edited in place")
    args = ap.parse_args(argv)

    warnings: list[str] = []
    ev = _load(args.eval, "eval.json", warnings)
    curve = _load(args.curve, "curve.json", warnings)
    res = Results(ev, curve, args.sample_chars, args.data)
    if ev:
        mode = res.eval_mode()
        if mode.startswith("FAKE") and args.out_dir is None and not args.check and not args.allow_fake:
            raise SystemExit(f"{args.eval} is from a --fake evaluation; refusing to write it into the repository "
                             f"(use --out-dir for a test)")
        if "quick" in mode:
            warnings.append(f"{args.eval} is a --quick evaluation: quote only full runs")

    total_missing = 0
    for path in args.files:
        if not path.exists():
            warnings.append(f"{path} not found, skipped")
            continue
        text = path.read_text(encoding="utf-8")
        filled: dict[str, str] = {}
        n_warn = len(warnings)
        new = fill_text(text, res, warnings, filled, path.name)
        total_missing += sum(1 for v in filled.values() if v.strip() == TBD)
        if args.check:
            print(f"== {path} ({len(filled)} keys)")
            for k, v in filled.items():
                print(f"  {k}: {v.strip() if k != 'facts' else '(%d lines)' % v.count(chr(10))}")
            continue
        dst = (args.out_dir / path.name) if args.out_dir else path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if new != text or dst != path:
            dst.write_text(new, encoding="utf-8")
        print(f"{dst}: {len(filled)} keys, {len(warnings) - n_warn} warnings")
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if total_missing:
        print(f"{total_missing} placeholder(s) left as {TBD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
