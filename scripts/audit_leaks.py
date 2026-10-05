#!/usr/bin/env python3
"""Red-team audit of a JSONL corpus for flight knowledge the scrub missed.

Each line of the corpus is a book: {"id", "title", "year", "text"} with
paragraphs separated by blank lines. Patterns come from a file of
"name :: regex" lines (default evals/redteam_patterns.txt); each is matched,
case-insensitively, against every paragraph after prewright.flight.normalize().

Two scan modes:

  * full (default): stream every book through Python. Exact, but slow
    (roughly 10-20 minutes per 6 GB per process).
  * --rg: a fast pre-screen. ripgrep finds literal anchors (--anchors file)
    with byte offsets; only the paragraphs around those offsets are decoded
    and matched. A pattern can only be found in a paragraph that contains one
    of the anchors, so keep the anchor list a superset of the patterns'
    literal parts. Takes about a minute per 6 GB.

For every pattern the auditor counts matching paragraphs, the number of
distinct books, and whether the paragraph would also hit the flight lexicon
(flight.contains_flight), then samples contexts. With --compare it runs a
second corpus (e.g. control) and puts both counts in one table.

    python scripts/audit_leaks.py /home/user/data/corpus/main/train.jsonl \\
        --compare /home/user/data/corpus/control/train.jsonl --rg \\
        --out evals/leak_audit --samples 8

Outputs, under --out: summary.md (tables + samples), counts.json, and
samples_<corpus>.jsonl (every sampled context with its book id and title).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prewright import flight  # noqa: E402

PARA_SEP_JSON = b"\\n\\n"  # a blank line, as it appears inside a JSON string


def load_patterns(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, rx = (s.strip() for s in line.split("::", 1))
        out.append((name, re.compile(rx, re.IGNORECASE)))
    return out


class Tally:
    """Per-pattern paragraph counts, distinct books, lexicon overlap, reservoir samples."""

    def __init__(self, names: list[str], k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)
        self.paras = Counter()
        self.lex_overlap = Counter()  # paragraphs that the flight lexicon would also hit
        self.books: dict[str, set[str]] = defaultdict(set)
        self.samples: dict[str, list[dict]] = defaultdict(list)
        self.matches: dict[str, Counter] = defaultdict(Counter)
        self.n_paragraphs = 0
        self.n_books = 0
        self.names = names

    def add(self, name: str, book: tuple[str, str], norm: str, match: str, lexicon_hit: bool):
        self.paras[name] += 1
        self.lex_overlap[name] += lexicon_hit
        self.books[name].add(book[0])
        self.matches[name][match.lower()] += 1
        n = self.paras[name]
        rec = {"id": book[0], "title": book[1], "match": match, "lexicon_hit": lexicon_hit,
               "context": _window(norm, match)}
        if len(self.samples[name]) < self.k:
            self.samples[name].append(rec)
        else:
            j = self.rng.randrange(n)
            if j < self.k:
                self.samples[name][j] = rec

    def merge(self, other: "Tally"):
        self.paras.update(other.paras)
        self.lex_overlap.update(other.lex_overlap)
        self.n_paragraphs += other.n_paragraphs
        self.n_books += other.n_books
        for k, v in other.books.items():
            self.books[k] |= v
        for k, v in other.matches.items():
            self.matches[k].update(v)
        for k, v in other.samples.items():  # approximate: keep a mix
            pool = self.samples[k] + v
            self.rng.shuffle(pool)
            self.samples[k] = pool[: self.k]


def _window(norm: str, match: str, width: int = 260) -> str:
    i = norm.lower().find(match.lower())
    if i < 0:
        return norm[: 2 * width]
    a, b = max(0, i - width), min(len(norm), i + len(match) + width)
    return ("..." if a else "") + norm[a:b] + ("..." if b < len(norm) else "")


def scan_paragraph(tally: Tally, patterns, book, para: str):
    norm = flight.normalize(para)
    lex = None
    for name, rx in patterns:
        m = rx.search(norm)
        if m:
            if lex is None:
                lex = flight.contains_flight(para)
            tally.add(name, book, norm, m.group(0), lex)


# --- full mode -----------------------------------------------------------------

def _scan_chunk(args):
    path, start, end, patterns_path, k, seed = args
    patterns = load_patterns(Path(patterns_path))
    tally = Tally([n for n, _ in patterns], k, seed)
    with open(path, "rb") as f:
        f.seek(start)
        if start:
            f.readline()  # we start mid-line; the previous chunk owns it
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line of a file still being written
            book = (str(rec.get("id")), rec.get("title", ""))
            paras = rec.get("text", "").split("\n\n")
            tally.n_books += 1
            tally.n_paragraphs += len(paras)
            for p in paras:
                if p:
                    scan_paragraph(tally, patterns, book, p)
    return tally


def scan_full(path: Path, patterns_path: Path, k: int, procs: int, seed: int) -> Tally:
    size = path.stat().st_size
    n = max(1, procs * 8)
    bounds = [size * i // n for i in range(n + 1)]
    jobs = [(str(path), bounds[i], bounds[i + 1] - 1 if i + 1 < n else size, str(patterns_path), k, seed + i)
            for i in range(n)]
    names = [n for n, _ in load_patterns(patterns_path)]
    total = Tally(names, k, seed)
    with Pool(procs) as pool:
        for t in pool.imap_unordered(_scan_chunk, jobs):
            total.merge(t)
    return total


# --- rg pre-screen mode ----------------------------------------------------------

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


def _decode(fragment: bytes) -> str:
    s = fragment.decode("utf-8", errors="replace")
    try:
        return json.loads('"' + s + '"')
    except json.JSONDecodeError:
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def scan_rg(path: Path, patterns_path: Path, anchors: Path, k: int, seed: int) -> Tally:
    patterns = load_patterns(patterns_path)
    tally = Tally([n for n, _ in patterns], k, seed)
    titles = _titles(path)
    tally.n_books = len(titles)
    proc = subprocess.Popen(["rg", "-i", "-o", "-b", "-n", "-F", "-f", str(anchors), str(path)],
                            stdout=subprocess.PIPE)
    seen: set[int] = set()
    with open(path, "rb") as f:
        for raw in proc.stdout:
            try:
                ln, off, _ = raw.split(b":", 2)
                ln, off = int(ln), int(off)
            except ValueError:
                continue
            a = max(0, off - 6000)
            f.seek(a)
            buf = f.read(12000)
            rel = off - a
            s = buf.rfind(PARA_SEP_JSON, 0, rel)
            nl = buf.rfind(b"\n", 0, rel)  # real newline = start of this JSON line
            s = max(s + len(PARA_SEP_JSON) if s >= 0 else 0, nl + 1 if nl >= 0 else 0)
            e = buf.find(PARA_SEP_JSON, rel)
            nl2 = buf.find(b"\n", rel)
            e = min(x for x in (e, nl2, len(buf)) if x >= 0)
            if a + s in seen:
                continue
            seen.add(a + s)
            frag = buf[s:e]
            if frag.startswith(b'{"id"'):
                j = frag.find(b'"text": "')
                frag = frag[j + 9:] if j >= 0 else frag
            if frag.endswith(b'"}'):
                frag = frag[:-2]
            scan_paragraph(tally, patterns, titles.get(ln, (f"line{ln}", "?")), _decode(frag))
    proc.wait()
    tally.n_paragraphs = -1  # unknown in this mode
    return tally


# --- report ------------------------------------------------------------------------

def write_report(out: Path, tallies: dict[str, Tally], names: list[str], args):
    out.mkdir(parents=True, exist_ok=True)
    labels = list(tallies)
    counts = {lab: {"n_books": t.n_books, "n_paragraphs": t.n_paragraphs,
                    "patterns": {n: {"paragraphs": t.paras[n], "books": len(t.books[n]),
                                     "lexicon_overlap": t.lex_overlap[n],
                                     "top_matches": t.matches[n].most_common(15)} for n in names}}
              for lab, t in tallies.items()}
    (out / "counts.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False))
    for lab, t in tallies.items():
        with (out / f"samples_{lab}.jsonl").open("w", encoding="utf-8") as f:
            for n in names:
                for rec in t.samples[n]:
                    f.write(json.dumps({"pattern": n, **rec}, ensure_ascii=False) + "\n")

    first = tallies[labels[0]]
    lines = [f"# Leak audit: {', '.join(labels)}", "",
             f"Mode: {'rg pre-screen' if args.rg else 'full scan'}; patterns: `{args.patterns}`", ""]
    for lab, t in tallies.items():
        per = f", {t.n_paragraphs:,} paragraphs" if t.n_paragraphs > 0 else ""
        lines.append(f"- **{lab}**: {t.n_books:,} books{per}")
    lines += ["", "| pattern | " + " | ".join(f"{lab} paras (books)" for lab in labels)
              + f" | {labels[0]} lexicon-overlap | top {labels[0]} matches |",
              "|---|" + "---|" * len(labels) + "---|---|"]
    for n in names:
        cells = [f"{tallies[lab].paras[n]:,} ({len(tallies[lab].books[n]):,})" for lab in labels]
        top = ", ".join(f"{m} ×{c}" for m, c in first.matches[n].most_common(4)).replace("|", "/")
        lines.append(f"| `{n}` | " + " | ".join(cells) + f" | {first.lex_overlap[n]:,} | {top} |")
    lines += ["", f"## Sample contexts from {labels[0]}", ""]
    for n in names:
        if not first.samples[n]:
            continue
        lines.append(f"### `{n}`")
        for rec in first.samples[n]:
            ctx = rec["context"].replace("\n", " ")
            lines.append(f"- [{rec['id']}] *{rec['title'][:70]}*: {ctx}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, nargs="+", help="JSONL file(s) to audit (counts are pooled)")
    ap.add_argument("--compare", type=Path, nargs="*", default=[], help="second corpus, e.g. control")
    ap.add_argument("--patterns", type=Path, default=ROOT / "evals" / "redteam_patterns.txt")
    ap.add_argument("--anchors", type=Path, default=ROOT / "evals" / "redteam_anchors.txt",
                    help="literal anchors for --rg (one per line)")
    ap.add_argument("--rg", action="store_true", help="fast ripgrep anchor pre-screen")
    ap.add_argument("--only", default="", help="comma-separated pattern-name prefixes to keep")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "leak_audit")
    args = ap.parse_args(argv)

    patterns_path = args.patterns
    if args.only:
        keep = tuple(args.only.split(","))
        kept = [ln for ln in patterns_path.read_text().splitlines()
                if ln.strip() and not ln.startswith("#") and ln.split("::")[0].strip().startswith(keep)]
        patterns_path = args.out / "patterns_selected.txt"
        args.out.mkdir(parents=True, exist_ok=True)
        patterns_path.write_text("\n".join(kept) + "\n")
    names = [n for n, _ in load_patterns(patterns_path)]

    def run(paths: list[Path]) -> Tally:
        total = Tally(names, args.samples, args.seed)
        for p in paths:
            t = (scan_rg(p, patterns_path, args.anchors, args.samples, args.seed) if args.rg
                 else scan_full(p, patterns_path, args.samples, args.procs, args.seed))
            total.merge(t)
        return total

    tallies = {args.corpus[0].parent.name: run(args.corpus)}
    if args.compare:
        tallies[args.compare[0].parent.name + " (compare)"] = run(args.compare)
    write_report(args.out, tallies, names, args)
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
