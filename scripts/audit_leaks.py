#!/usr/bin/env python3
"""Red-team audit of a JSONL corpus for flight knowledge the scrub missed.

Usage
-----

    # whole main corpus vs control, and the books the model actually trained on,
    # from one streaming pass per corpus (about 15-20 min per 6.6 GB file, 1 core,
    # well under 1 GB of memory):
    python3 scripts/audit_leaks.py \\
        /home/user/data/corpus/main/train.jsonl /home/user/data/corpus/main/val.jsonl \\
        --compare /home/user/data/corpus/control/train.jsonl \\
        --with-trained --out evals/leak_audit --samples 12 --keep 200

    # only the trained subset (books whose rank is within the tokenizer's cutoff):
    python3 scripts/audit_leaks.py /home/user/data/corpus/main/train.jsonl --trained-only

    # only some patterns (name prefixes), e.g. to verify a lexicon fix:
    python3 scripts/audit_leaks.py /home/user/data/corpus/main/train.jsonl --only fix_,p1_

Input: each line of the corpus is a book {"id", "title", "year", "text"}, with
paragraphs separated by blank lines. Patterns come from a file of
"name :: regex :: anchor|anchor" lines (default evals/redteam_patterns.txt).
Each regex is matched, case-insensitively, against every paragraph after
prewright.flight.normalize(), but only on paragraphs containing one of its
anchors.

Trained subset (--trained-only / --with-trained): scripts/tokenize_corpus.py
orders train books by rank = sha1(f"{salt}:{id}") and trains on those with
rank <= cutoff. Salt and cutoff are read from --meta (default
/home/user/data/tokenized/main/train.meta.json). The filter applies to files
named train*.jsonl only; validation books are never trained on, so under
--trained-only other files are skipped, and under --with-trained they count
towards the "all books" group only. The same cutoff selects control's trained
books (control = main's books plus the dropped flight books).

Scan modes:

  * full (default): stream every book once. Each book is normalised once, a
    str.find pass over it for every anchor picks out the candidate paragraphs,
    and only the patterns whose anchors occur there are run. One book in
    memory at a time. --procs N > 1 splits the file into byte ranges for N
    worker processes (mind the memory of anything else running).
  * --rg: ripgrep (with --no-mmap) finds every anchor in the raw file with its
    byte offset; only the paragraphs around those offsets are read and
    matched. Faster when the anchors are rare; much slower than full mode
    when they are common ("wings", "flight", "in the air"), and paragraph
    counts are not available.

For each pattern it reports the number of matching paragraphs and distinct
books, how many of those paragraphs the flight lexicon would also hit
(flight.contains_flight: in a scrubbed corpus this should be 0), the most
common matched strings, a reservoir sample of contexts (--samples) and the
first --keep hits in file order. Outputs under --out: summary.md (table plus
samples), counts.json, samples_<label>.jsonl and hits_<label>.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prewright import flight  # noqa: E402

PARA_SEP_JSON = b"\\n\\n"  # a blank line, as it appears inside a JSON string
_ID = re.compile(rb'"id":\s*"((?:[^"\\]|\\.)*)"')
DEFAULT_META = Path("/home/user/data/tokenized/main/train.meta.json")


@dataclass(frozen=True)
class Pattern:
    name: str
    rx: re.Pattern[str]
    anchors: tuple[str, ...]


def load_patterns(path: Path, only: str = "") -> list[Pattern]:
    keep = tuple(p for p in only.split(",") if p)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [s.strip() for s in line.split(" :: ")]
        if len(parts) != 3:
            raise ValueError(f"expected 'name :: regex :: anchors': {line[:80]}")
        name, rx, anchors = parts
        if keep and not name.startswith(keep):
            continue
        out.append(Pattern(name, re.compile(rx, re.IGNORECASE), tuple(a.lower() for a in anchors.split("|") if a)))
    return out


# --- trained subset ------------------------------------------------------------------

@dataclass(frozen=True)
class TrainedFilter:
    salt: str
    cutoff: bytes

    @classmethod
    def from_meta(cls, path: Path) -> "TrainedFilter":
        meta = json.loads(path.read_text())
        cutoff = meta.get("cutoff") or meta.get("last_rank")
        if not cutoff:
            raise SystemExit(f"{path}: no 'cutoff' or 'last_rank' (was the train split capped?)")
        return cls(meta.get("salt", "prewright-1903"), bytes.fromhex(cutoff))

    def __call__(self, book_id: str) -> bool:
        # Same as scripts/tokenize_corpus.py rank(): sha1(f"{salt}:{id}") <= cutoff.
        return hashlib.sha1(f"{self.salt}:{book_id}".encode()).digest() <= self.cutoff


def is_train_file(path: Path) -> bool:
    return path.name.startswith("train")


# --- tallies -----------------------------------------------------------------------------

class Tally:
    """Per-pattern paragraph counts, distinct books, lexicon overlap, samples."""

    def __init__(self, k: int, seed: int = 0, keep: int = 0):
        self.k = k
        self.keep = keep
        self.rng = random.Random(seed)
        self.paras: Counter[str] = Counter()
        self.lex_overlap: Counter[str] = Counter()
        self.books: dict[str, set[str]] = defaultdict(set)
        self.samples: dict[str, list[dict]] = defaultdict(list)
        self.kept: dict[str, list[dict]] = defaultdict(list)
        self.matches: dict[str, Counter[str]] = defaultdict(Counter)
        self.n_paragraphs = 0  # 0 = not counted (rg mode)
        self.n_books = 0
        self.n_scanned = 0  # paragraphs actually decoded and matched

    def add(self, name: str, book: tuple[str, str], norm: str, match: str, lexicon_hit: bool, para: int = -1):
        self.paras[name] += 1
        self.lex_overlap[name] += lexicon_hit
        self.books[name].add(book[0])
        self.matches[name][" ".join(match.lower().split())] += 1
        rec = {"id": book[0], "title": book[1], "para": para, "match": match, "lexicon_hit": lexicon_hit,
               "context": _window(norm, match)}
        if len(self.kept[name]) < self.keep:
            self.kept[name].append(rec)
        if len(self.samples[name]) < self.k:
            self.samples[name].append(rec)
        else:
            j = self.rng.randrange(self.paras[name])
            if j < self.k:
                self.samples[name][j] = rec

    def merge(self, other: "Tally"):
        for k, v in other.samples.items():  # weighted by each side's count, approximately
            mine, theirs = self.samples[k], v
            a, b = self.paras[k], other.paras[k]
            pool = [(self.rng.random() ** (1 / max(a, 1)), r) for r in mine] + \
                   [(self.rng.random() ** (1 / max(b, 1)), r) for r in theirs]
            pool.sort(key=lambda t: -t[0])
            self.samples[k] = [r for _, r in pool[: self.k]]
        for k, v in other.kept.items():
            self.kept[k] = (self.kept[k] + v)[: self.keep]
        self.paras.update(other.paras)
        self.lex_overlap.update(other.lex_overlap)
        self.n_paragraphs += other.n_paragraphs
        self.n_books += other.n_books
        self.n_scanned += other.n_scanned
        for k, v in other.books.items():
            self.books[k] |= v
        for k, v in other.matches.items():
            self.matches[k].update(v)


def _window(norm: str, match: str, width: int = 300) -> str:
    i = norm.lower().find(match.lower())
    if i < 0:
        return norm[: 2 * width]
    a, b = max(0, i - width), min(len(norm), i + len(match) + width)
    return ("..." if a else "") + norm[a:b] + ("..." if b < len(norm) else "")


def match_paragraph(tallies: list[Tally], patterns, book: tuple[str, str], raw: str, norm: str, para: int = -1):
    """Run `patterns` (already screened by anchor) on one normalised paragraph."""
    for t in tallies:
        t.n_scanned += 1
    lex = None
    for p in patterns:
        m = p.rx.search(norm)
        if m:
            if lex is None:
                lex = flight.contains_flight(raw)
            for t in tallies:
                t.add(p.name, book, norm, m.group(0), lex, para)


def scan_paragraph(tallies: list[Tally], patterns: list[Pattern], book: tuple[str, str], para: str):
    """Anchor-screen and match one raw paragraph (rg mode)."""
    norm = flight.normalize(para)
    low = norm.lower()
    cands = [p for p in patterns if any(a in low for a in p.anchors)]
    if cands:
        match_paragraph(tallies, cands, book, para, norm)
    else:
        for t in tallies:
            t.n_scanned += 1


# --- full mode -------------------------------------------------------------------

class BookScanner:
    """Finds candidate paragraphs with one str.find pass per anchor over the whole (normalised) book."""

    def __init__(self, patterns: list[Pattern]):
        self.patterns = patterns
        by_anchor: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(patterns):
            for a in p.anchors:
                by_anchor[a].append(i)
        self.by_anchor = dict(by_anchor)

    def scan(self, tallies: list[Tally], book: tuple[str, str], paras: list[str]):
        norms = flight.normalize_paragraphs(paras)
        low = "\x00".join(norms).lower()
        starts, pos = [], 0
        for p in norms:
            starts.append(pos)
            pos += len(p) + 1
        cand: dict[int, set[int]] = {}
        for anchor, pats in self.by_anchor.items():
            i = low.find(anchor)
            while i != -1:
                k = bisect_right(starts, i) - 1
                cand.setdefault(k, set()).update(pats)
                i = low.find(anchor, i + 1)
        for k in sorted(cand):
            match_paragraph(tallies, [self.patterns[j] for j in sorted(cand[k])], book, paras[k], norms[k], k)


def _scan_range(args):
    path, start, end, patterns_path, only, k, keep, seed, trained, mode = args
    patterns = load_patterns(Path(patterns_path), only)
    scanner = BookScanner(patterns)
    all_t, tr_t = Tally(k, seed, keep), Tally(k, seed + 1, keep)
    with open(path, "rb") as f:
        f.seek(start)
        if start:
            f.readline()  # we start mid-line; the previous range owns that line
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            if mode == "trained-only":  # skip untrained books without parsing them
                m = _ID.search(line, 0, 300)
                if m and not trained(m.group(1).decode()):
                    continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line of a file still being written
            book = (str(rec.get("id")), rec.get("title", ""))
            in_tr = trained is not None and trained(book[0])
            targets = []
            if mode != "trained-only":
                targets.append(all_t)
            if in_tr:
                targets.append(tr_t)
            if not targets:
                continue
            paras = [p for p in rec.get("text", "").split("\n\n") if p.strip()]
            for t in targets:
                t.n_books += 1
                t.n_paragraphs += len(paras)
            scanner.scan(targets, book, paras)
            del rec, paras
    return all_t, tr_t


def scan_full(path: Path, patterns_path: Path, only: str, k: int, keep: int, procs: int, seed: int,
              trained, mode: str) -> tuple[Tally, Tally]:
    size = path.stat().st_size
    n = 1 if procs <= 1 else procs * 8
    bounds = [size * i // n for i in range(n + 1)]
    jobs = [(str(path), bounds[i], bounds[i + 1] - 1, str(patterns_path), only, k, keep, seed + 2 * i, trained, mode)
            for i in range(n)]
    all_t, tr_t = Tally(k, seed, keep), Tally(k, seed + 1, keep)
    if procs <= 1:
        results = map(_scan_range, jobs)
    else:
        pool = Pool(procs)
        results = pool.imap(_scan_range, jobs)  # in order, so --keep stays "first in file order"
    for a, t in results:
        all_t.merge(a)
        tr_t.merge(t)
    if procs > 1:
        pool.close()
        pool.join()
    return all_t, tr_t


# --- rg mode ------------------------------------------------------------------------

def _titles(path: Path) -> dict[int, tuple[str, str]]:
    """Line number -> (id, title), read with ripgrep from the start of each line."""
    out = subprocess.run(["rg", "--no-mmap", "-n", "-o", r'^\{"id": "[^"]*", "title": "(?:[^"\\]|\\.)*"',
                          str(path)], capture_output=True, text=True, check=False).stdout
    titles = {}
    for line in out.splitlines():
        ln, rest = line.split(":", 1)
        try:
            rec = json.loads(rest + "}")
        except json.JSONDecodeError:
            continue
        titles[int(ln)] = (rec["id"], rec["title"])
    return titles


def _rg_literals(anchors: list[str]) -> list[str]:
    """Each anchor, plus variants where any of its spaces is a raw (JSON-escaped) line break."""
    out = set()
    for a in anchors:
        parts = a.split(" ")
        if len(parts) > 4:
            out.add(a)
            continue
        for seps in itertools.product((" ", "\\n"), repeat=len(parts) - 1):
            out.add(parts[0] + "".join(s + p for s, p in zip(seps, parts[1:])))
    return sorted(out)


def _decode(fragment: bytes) -> str:
    s = fragment.decode("utf-8", errors="replace")
    try:
        return json.loads('"' + s + '"')
    except json.JSONDecodeError:
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def scan_rg(path: Path, patterns_path: Path, only: str, k: int, keep: int, seed: int,
            trained, mode: str) -> tuple[Tally, Tally]:
    patterns = load_patterns(patterns_path, only)
    all_t, tr_t = Tally(k, seed, keep), Tally(k, seed + 1, keep)
    titles = _titles(path)
    tr_lines = {ln for ln, (bid, _) in titles.items() if trained is not None and trained(bid)}
    all_t.n_books = len(titles) if mode != "trained-only" else 0
    tr_t.n_books = len(tr_lines)
    lits = _rg_literals(sorted({a for p in patterns for a in p.anchors}))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(lits) + "\n")
    proc = subprocess.Popen(["rg", "--no-mmap", "-i", "-o", "-b", "-n", "-F", "-f", tf.name, str(path)],
                            stdout=subprocess.PIPE, bufsize=1 << 20)
    cur_end = -1
    try:
        with open(path, "rb") as f:
            for raw in proc.stdout:
                try:
                    ln_s, off_s, _ = raw.split(b":", 2)
                    ln, off = int(ln_s), int(off_s)
                except ValueError:
                    continue
                if off < cur_end:
                    continue  # same paragraph as the previous anchor
                targets = [all_t] if mode != "trained-only" else []
                if ln in tr_lines:
                    targets.append(tr_t)
                if not targets:
                    continue
                a = max(0, off - 8000)
                f.seek(a)
                buf = f.read(16000)
                rel = off - a
                s = buf.rfind(PARA_SEP_JSON, 0, rel)
                nl = buf.rfind(b"\n", 0, rel)  # a real newline starts this JSON line
                s = max(s + len(PARA_SEP_JSON) if s >= 0 else 0, nl + 1 if nl >= 0 else 0)
                e = buf.find(PARA_SEP_JSON, rel)
                nl2 = buf.find(b"\n", rel)
                e = min(x for x in (e, nl2, len(buf)) if x >= 0)
                cur_end = a + e
                frag = buf[s:e]
                if frag.startswith(b'{"id"'):
                    j = frag.find(b'"text": "')
                    frag = frag[j + 9:] if j >= 0 else frag
                if frag.endswith(b'"}'):
                    frag = frag[:-2]
                scan_paragraph(targets, patterns, titles.get(ln, (f"line{ln}", "?")), _decode(frag))
    finally:
        proc.wait()
        os.unlink(tf.name)
    return all_t, tr_t


# --- report --------------------------------------------------------------------------

def _per10k(t: Tally, n: str) -> str:
    return f"{10000 * len(t.books[n]) / t.n_books:.1f}" if t.n_books else "-"


def write_report(out: Path, tallies: dict[str, Tally], names: list[str], args):
    out.mkdir(parents=True, exist_ok=True)
    labels = list(tallies)
    counts = {lab: {"n_books": t.n_books, "n_paragraphs": t.n_paragraphs, "n_scanned": t.n_scanned,
                    "patterns": {n: {"paragraphs": t.paras[n], "books": len(t.books[n]),
                                     "books_per_10k": round(10000 * len(t.books[n]) / t.n_books, 2) if t.n_books else None,
                                     "lexicon_overlap": t.lex_overlap[n],
                                     "top_matches": t.matches[n].most_common(25)} for n in names}}
              for lab, t in tallies.items()}
    (out / "counts.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False), encoding="utf-8")
    for lab, t in tallies.items():
        slug = re.sub(r"[^\w.-]+", "_", lab).strip("_")
        with (out / f"samples_{slug}.jsonl").open("w", encoding="utf-8") as f:
            for n in names:
                for rec in t.samples[n]:
                    f.write(json.dumps({"pattern": n, **rec}, ensure_ascii=False) + "\n")
        if args.keep:
            with (out / f"hits_{slug}.jsonl").open("w", encoding="utf-8") as f:
                for n in names:
                    for rec in t.kept[n]:
                        f.write(json.dumps({"pattern": n, **rec}, ensure_ascii=False) + "\n")

    first = tallies[labels[0]]
    lines = [f"# Leak audit: {' vs '.join(labels)}", "",
             f"Mode: {'ripgrep anchor pre-screen' if args.rg else 'full scan'}; patterns: `{args.patterns}`", ""]
    for lab, t in tallies.items():
        per = f", {t.n_paragraphs:,} paragraphs" if t.n_paragraphs else ""
        lines.append(f"- **{lab}**: {t.n_books:,} books{per}; {t.n_scanned:,} paragraphs matched against patterns")
    lines += ["", "| pattern | " + " | ".join(f"{lab} paras (books) [books/10k]" for lab in labels)
              + f" | {labels[0]} lexicon overlap | top {labels[0]} matches |",
              "|---|" + "---|" * len(labels) + "---|---|"]
    for n in names:
        cells = [f"{tallies[lab].paras[n]:,} ({len(tallies[lab].books[n]):,}) [{_per10k(tallies[lab], n)}]"
                 for lab in labels]
        top = ", ".join(f"{m[:40]} ×{c}" for m, c in first.matches[n].most_common(5)).replace("|", "/")
        lines.append(f"| `{n}` | " + " | ".join(cells) + f" | {first.lex_overlap[n]:,} | {top} |")
    for lab in labels[: 2 if args.with_trained else 1]:
        t = tallies[lab]
        lines += ["", f"## Sample contexts from {lab}", ""]
        for n in names:
            if not t.samples[n]:
                continue
            lines.append(f"### `{n}`")
            for rec in t.samples[n]:
                ctx = rec["context"].replace("\n", " ").replace("|", "/")
                lines.append(f"- [{rec['id']}] *{rec['title'][:70]}*: {ctx}")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, nargs="+", help="JSONL file(s) to audit (counts are pooled)")
    ap.add_argument("--compare", type=Path, nargs="*", default=[], help="a second group of files, e.g. control")
    ap.add_argument("--label", default="", help="label for the first group (default: parent dir name)")
    ap.add_argument("--compare-label", default="", help="label for the --compare group")
    ap.add_argument("--patterns", type=Path, default=ROOT / "evals" / "redteam_patterns.txt")
    ap.add_argument("--only", default="", help="comma-separated pattern-name prefixes to keep (e.g. rt_,fix_)")
    ap.add_argument("--rg", action="store_true", help="ripgrep anchor pre-screen (fast only for rare anchors)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--trained-only", action="store_true",
                     help="audit only the train books the model trained on (rank <= cutoff in --meta)")
    grp.add_argument("--with-trained", action="store_true",
                     help="report all books AND the trained subset, from the same pass")
    ap.add_argument("--meta", type=Path, default=DEFAULT_META, help="tokenizer meta.json with salt and cutoff")
    ap.add_argument("--samples", type=int, default=8, help="reservoir sample size per pattern")
    ap.add_argument("--keep", type=int, default=0, help="also keep the first N hits per pattern (hits_*.jsonl)")
    ap.add_argument("--procs", type=int, default=1, help="worker processes in full mode (1 = in-process)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "leak_audit")
    args = ap.parse_args(argv)

    names = [p.name for p in load_patterns(args.patterns, args.only)]
    mode = "trained-only" if args.trained_only else "with-trained" if args.with_trained else "all"
    trained = TrainedFilter.from_meta(args.meta) if mode != "all" else None

    def run(paths: list[Path]) -> tuple[Tally, Tally]:
        all_t, tr_t = Tally(args.samples, args.seed, args.keep), Tally(args.samples, args.seed + 1, args.keep)
        for p in paths:
            filt = trained if is_train_file(p) else None
            if mode == "trained-only" and filt is None:
                print(f"skipping {p}: not a train file, and --trained-only", file=sys.stderr)
                continue
            if args.rg:
                a, t = scan_rg(p, args.patterns, args.only, args.samples, args.keep, args.seed, filt, mode)
            else:
                a, t = scan_full(p, args.patterns, args.only, args.samples, args.keep, args.procs, args.seed,
                                 filt, mode)
            print(f"scanned {p}: {a.n_books:,} books ({t.n_books:,} trained), "
                  f"{a.n_scanned + t.n_scanned:,} paragraph matches run", file=sys.stderr, flush=True)
            all_t.merge(a)
            tr_t.merge(t)
        return all_t, tr_t

    tallies: dict[str, Tally] = {}
    groups = [(args.label or args.corpus[0].parent.name, args.corpus)]
    if args.compare:
        groups.append((args.compare_label or args.compare[0].parent.name, args.compare))
    for label, paths in groups:
        a, t = run(paths)
        if mode != "trained-only":
            tallies[label] = a
        if mode != "all":
            tallies[f"{label} (trained)"] = t
    if mode == "with-trained" and len(groups) == 2:  # main, main(trained), control, control(trained)
        order = list(tallies)
        tallies = {k: tallies[k] for k in [order[0], order[1], order[2], order[3]]}
    write_report(args.out, tallies, names, args)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
