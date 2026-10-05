#!/usr/bin/env python3
"""Red-team audit of a JSONL corpus for flight knowledge the scrub missed.

Each line of the corpus is a book: {"id", "title", "year", "text"}, with
paragraphs separated by blank lines. Patterns come from a file of
"name :: regex :: anchor|anchor" lines (default evals/redteam_patterns.txt).
Each regex is matched, case-insensitively, against every paragraph after
prewright.flight.normalize(), but only on paragraphs containing one of its
anchors.

Two scan modes:

  * --rg (recommended): ripgrep finds every anchor in the raw file with its
    byte offset, using Aho-Corasick and one core. Only the paragraphs around
    those offsets are read, decoded and matched. A full 6 GB corpus takes a
    few minutes.
  * full (default): stream every book through Python with --procs workers.
    It doesn't depend on ripgrep and it also counts paragraphs, but it is
    slower.

For each pattern it reports the number of matching paragraphs and distinct
books, how many of those paragraphs the flight lexicon would also hit
(flight.contains_flight: in a scrubbed corpus this should be 0), the most
common matched strings, and a reservoir sample of contexts. It accepts
several corpora; --compare adds another group (e.g. control) to the table.

    python scripts/audit_leaks.py --rg \\
        /home/user/data/corpus/main/train.jsonl /home/user/data/corpus/main/val.jsonl \\
        --compare /home/user/data/corpus/control/train.jsonl /home/user/data/corpus/control/val.jsonl \\
        --out evals/leak_audit --samples 8

Outputs under --out: summary.md (table plus samples), counts.json, and
samples_<label>.jsonl (every sampled context with its book id and title).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prewright import flight  # noqa: E402

PARA_SEP_JSON = b"\\n\\n"  # a blank line, as it appears inside a JSON string


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


class Tally:
    """Per-pattern paragraph counts, distinct books, lexicon overlap, reservoir samples."""

    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)
        self.paras: Counter[str] = Counter()
        self.lex_overlap: Counter[str] = Counter()
        self.books: dict[str, set[str]] = defaultdict(set)
        self.samples: dict[str, list[dict]] = defaultdict(list)
        self.matches: dict[str, Counter[str]] = defaultdict(Counter)
        self.n_paragraphs = 0  # 0 = not counted (rg mode)
        self.n_books = 0
        self.n_scanned = 0  # paragraphs actually decoded and matched

    def add(self, name: str, book: tuple[str, str], norm: str, match: str, lexicon_hit: bool):
        self.paras[name] += 1
        self.lex_overlap[name] += lexicon_hit
        self.books[name].add(book[0])
        self.matches[name][" ".join(match.lower().split())] += 1
        rec = {"id": book[0], "title": book[1], "match": match, "lexicon_hit": lexicon_hit,
               "context": _window(norm, match)}
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


def scan_paragraph(tally: Tally, patterns: list[Pattern], book: tuple[str, str], para: str):
    tally.n_scanned += 1
    norm = flight.normalize(para)
    low = norm.lower()
    lex = None
    for p in patterns:
        if not any(a in low for a in p.anchors):
            continue
        m = p.rx.search(norm)
        if m:
            if lex is None:
                lex = flight.contains_flight(para)
            tally.add(p.name, book, norm, m.group(0), lex)


# --- full mode -------------------------------------------------------------------

def _scan_chunk(args):
    path, start, end, patterns_path, only, k, seed = args
    patterns = load_patterns(Path(patterns_path), only)
    anchors = sorted({a for p in patterns for a in p.anchors})
    tally = Tally(k, seed)
    with open(path, "rb") as f:
        f.seek(start)
        if start:
            f.readline()  # we start mid-line; the previous chunk owns that line
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line of a file still being written
            book = (str(rec.get("id")), rec.get("title", ""))
            paras = [p for p in rec.get("text", "").split("\n\n") if p.strip()]
            tally.n_books += 1
            tally.n_paragraphs += len(paras)
            # Book-level anchor screen: only paragraphs containing an anchor are matched.
            low = "\x00".join(paras).lower()
            if not any(a in low for a in anchors):
                continue
            for para in paras:
                pl = para.lower()
                if any(a in pl for a in anchors) or "-\n" in para:
                    scan_paragraph(tally, patterns, book, para)
    return tally


def scan_full(path: Path, patterns_path: Path, only: str, k: int, procs: int, seed: int) -> Tally:
    size = path.stat().st_size
    n = max(1, procs * 8)
    bounds = [size * i // n for i in range(n + 1)]
    jobs = [(str(path), bounds[i], bounds[i + 1] - 1, str(patterns_path), only, k, seed + i) for i in range(n)]
    total = Tally(k, seed)
    with Pool(procs) as pool:
        for t in pool.imap_unordered(_scan_chunk, jobs):
            total.merge(t)
    return total


# --- rg mode ------------------------------------------------------------------------

def _titles(path: Path) -> dict[int, tuple[str, str]]:
    """Line number -> (id, title), read with ripgrep from the start of each line."""
    out = subprocess.run(["rg", "-n", "-o", r'^\{"id": "[^"]*", "title": "(?:[^"\\]|\\.)*"', str(path)],
                         capture_output=True, text=True, check=False).stdout
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


def scan_rg(path: Path, patterns_path: Path, only: str, k: int, seed: int) -> Tally:
    patterns = load_patterns(patterns_path, only)
    tally = Tally(k, seed)
    titles = _titles(path)
    tally.n_books = len(titles)
    lits = _rg_literals(sorted({a for p in patterns for a in p.anchors}))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(lits) + "\n")
    proc = subprocess.Popen(["rg", "-i", "-o", "-b", "-n", "-F", "-f", tf.name, str(path)],
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
                scan_paragraph(tally, patterns, titles.get(ln, (f"line{ln}", "?")), _decode(frag))
    finally:
        proc.wait()
        os.unlink(tf.name)
    return tally


# --- report --------------------------------------------------------------------------

def write_report(out: Path, tallies: dict[str, Tally], names: list[str], args):
    out.mkdir(parents=True, exist_ok=True)
    labels = list(tallies)
    counts = {lab: {"n_books": t.n_books, "n_paragraphs": t.n_paragraphs, "n_scanned": t.n_scanned,
                    "patterns": {n: {"paragraphs": t.paras[n], "books": len(t.books[n]),
                                     "lexicon_overlap": t.lex_overlap[n],
                                     "top_matches": t.matches[n].most_common(25)} for n in names}}
              for lab, t in tallies.items()}
    (out / "counts.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False), encoding="utf-8")
    for lab, t in tallies.items():
        with (out / f"samples_{re.sub(r'[^\w.-]+', '_', lab)}.jsonl").open("w", encoding="utf-8") as f:
            for n in names:
                for rec in t.samples[n]:
                    f.write(json.dumps({"pattern": n, **rec}, ensure_ascii=False) + "\n")

    first = tallies[labels[0]]
    lines = [f"# Leak audit: {' vs '.join(labels)}", "",
             f"Mode: {'ripgrep anchor pre-screen' if args.rg else 'full scan'}; patterns: `{args.patterns}`", ""]
    for lab, t in tallies.items():
        per = f", {t.n_paragraphs:,} paragraphs" if t.n_paragraphs else ""
        lines.append(f"- **{lab}**: {t.n_books:,} books{per}; {t.n_scanned:,} paragraphs matched against patterns")
    lines += ["", "| pattern | " + " | ".join(f"{lab} paras (books)" for lab in labels)
              + f" | {labels[0]} lexicon overlap | top {labels[0]} matches |",
              "|---|" + "---|" * len(labels) + "---|---|"]
    for n in names:
        cells = [f"{tallies[lab].paras[n]:,} ({len(tallies[lab].books[n]):,})" for lab in labels]
        top = ", ".join(f"{m[:40]} ×{c}" for m, c in first.matches[n].most_common(5)).replace("|", "/")
        lines.append(f"| `{n}` | " + " | ".join(cells) + f" | {first.lex_overlap[n]:,} | {top} |")
    lines += ["", f"## Sample contexts from {labels[0]}", ""]
    for n in names:
        if not first.samples[n]:
            continue
        lines.append(f"### `{n}`")
        for rec in first.samples[n]:
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
    ap.add_argument("--rg", action="store_true", help="fast ripgrep anchor pre-screen (recommended)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--procs", type=int, default=2, help="worker processes in full mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "leak_audit")
    args = ap.parse_args(argv)

    names = [p.name for p in load_patterns(args.patterns, args.only)]

    def run(paths: list[Path]) -> Tally:
        total = Tally(args.samples, args.seed)
        for p in paths:
            t = (scan_rg(p, args.patterns, args.only, args.samples, args.seed) if args.rg
                 else scan_full(p, args.patterns, args.only, args.samples, args.procs, args.seed))
            print(f"scanned {p}: {t.n_books:,} books, {t.n_scanned:,} paragraphs matched", file=sys.stderr)
            total.merge(t)
        return total

    tallies = {args.label or args.corpus[0].parent.name: run(args.corpus)}
    if args.compare:
        tallies[args.compare_label or args.compare[0].parent.name] = run(args.compare)
    write_report(args.out, tallies, names, args)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
