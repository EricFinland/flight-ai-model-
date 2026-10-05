"""Build the `main` (flight-scrubbed) and `control` corpora from raw PG-19 books.

    python scripts/build_corpus.py --raw /home/user/data/pg19_raw \\
        --metadata /home/user/data/metadata.csv --out /home/user/data/corpus [--limit N]

Per book: date check, Project Gutenberg remnants stripped, leakage filter
(REJECT and REVIEW drop the book from both corpora), English test, then the
flight scrub for `main` only. ~2% of books (by a hash of the id) go to
validation in both corpora. Outputs, all under --out:

    {main,control}/{train,val}.jsonl   one book per line: id, title, year, text
    flight_eval.jsonl                  flight paragraphs from validation books
    flight_removed_train_sample.jsonl  a sample of flight paragraphs cut from train books
    manifest.csv, stats.json           per-book decisions and totals

Re-running overwrites every output, so the build is idempotent. With 3
workers the full 17,350 books take roughly 15-25 minutes.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright import english, flight, leakage  # noqa: E402
from prewright.dates import is_before_cutoff  # noqa: E402

VAL_PERMILLE = 20  # 2% of books go to validation
MIN_BOOK_CHARS = 1000
TRAIN_SAMPLE_SIZE = 500
DECISIONS = ("kept", "dropped_flight_book", "dropped_leakage", "dropped_review", "dropped_nonenglish", "dropped_empty")

_SRC = Path(__file__).resolve().parents[1] / "src" / "prewright"
_PG_FOOTER = re.compile(r"^.{0,10}end of (?:the |this )?project gutenberg", re.I | re.M)
_PG_HEADER = re.compile(r"^.{0,10}(?:the )?project gutenberg(?:'s)? e-?(?:book|text)\b.*$", re.I | re.M)
# Volunteer credits and transcriber notes near either end of the book.
_EDGE_CREDIT = re.compile(
    r"\b(?:produced|prepared|scanned|transcribed|proofread|digitized)\s+by\b|\bhtml\s+version\b|transcriber"
    r"|\be-?text\b|\belectronic\s+(?:book|text|edition|version)|original\s+file|fulcrum\s+edition|\bthis\s+e-?(?:book|text)\b|\bvolunteers?\b.{0,40}\b(?:proofread|prepared|produced)", re.I)
_EDGE_PARAGRAPHS = 15
_PARA_SPLIT = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")


def split_of(book_id: str) -> str:
    h = int(hashlib.sha1(f"pg19:{book_id}".encode()).hexdigest()[:8], 16)
    return "val" if h % 1000 < VAL_PERMILLE else "train"


def strip_pg_remnants(text: str) -> str:
    """Cut the Gutenberg footer (and anything after it) and a residual header line."""
    for m in _PG_FOOTER.finditer(text):
        if m.start() > len(text) // 2:
            text = text[: m.start()]
            break
    head = text[:3000]
    if m := _PG_HEADER.search(head):
        text = text[m.end():]
    paras = _PARA_SPLIT.split(text.strip("\n"))
    n = len(paras)
    keep = [p for i, p in enumerate(paras)
            if not ((i < _EDGE_PARAGRAPHS or i >= n - _EDGE_PARAGRAPHS)
                    and (_EDGE_CREDIT.search(p) or not re.search(r"\w", p)))]
    return "\n\n".join(keep)


def load_metadata(path: Path) -> dict[str, tuple[str, str]]:
    """id -> (title, year). The file has no header: id, title, publication_date, url."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row[0]: (row[1], row[2]) for row in csv.reader(f) if len(row) >= 3}


# ---------------------------------------------------------------- fast leakage filter
# leakage.filter_document runs ~30 case-insensitive regexes over every book and
# every paragraph, which dominates the build. fast_filter_document returns the
# identical result: each rule is first tried, case-sensitively, on a
# length-preserving lowercased copy of the text (a superset test that is several
# times faster), and the real rule only runs where that copy matches.
# check_fast_filter() compares the two on sample books.

_LOWER = {**{c: c + 32 for c in range(65, 91)}, 0x212A: ord("k"), 0x130: ord("i"), 0x131: ord("i"), 0x17F: ord("s")}

_USSR = tuple(f"u{a}s{b}s{c}r" for a in ("", ".") for b in ("", ".") for c in ("", "."))
# Literals of which every match of the rule contains at least one, in lowercased
# text (whitespace collapsed for the whole-document rules). Keyed by the rule's
# name and a hash of its pattern: if leakage.py changes a rule, its entry stops
# applying and the slower regex gate is used instead.
_ANCHORS: dict[tuple[str, str], tuple[str, ...]] = {
    ("gutenberg", "6dcf9093ef"): ("gutenberg",),
    ("ebook", "b679c7bc96"): ("book", "text"),
    ("url", "4b2051927e"): ("http", "www.", ".com", ".org", ".net", ".edu"),
    ("email", "ebe8afa854"): ("@",),
    ("scanner", "1be99a733d"): ("digiti", "internet archive", "google books", "hathitrust", "ocr"),
    ("transcriber", "d0d618beae"): ("transcriber", "produced by"),
    ("isbn", "57a4efe4bb"): ("isbn",),
    ("modern_copyright", "e4f4d06b66"): ("copyright", "\u00a9", "(c)", "reprinted", "this edition"),
    ("relativity_theory", "10d7cadb57"): ("relativity",),
    ("quantum_physics", "acd207ec60"): ("quantum",),
    ("subatomic_coinages", "84f33a9b09"): ("proton", "neutron", "isotope", "photon", "neutrino"),
    ("nuclear", "bcf426dc81"): ("nuclear", "atom", "a-bomb"),
    ("world_wars", "9230091b31"): ("world war", "great war of 1914"),
    ("wright_flight", "83080976a8"): ("wright flyer", "powered flight", "aeroplane flight", "airplane flight", "kitty hawk"),
    ("russo_japanese_war", "f555124906"): ("russo-japanese war", "battle of tsushima", "treaty of portsmouth"),
    ("titanic", "a4f4c0790f"): ("titanic",),
    ("twentieth_century_politics", "88aa89f21c"): ("hitler", "nazi", "soviet union", "league of nations", "lenin", "stalin", *_USSR),
    ("medicine", "4439a30a46"): ("vitamin", "penicillin", "insulin", "antibiotic"),
    ("technology", "61df2c1793"): ("radar", "laser", "transistor", "internet", "television set", "radio broadcast",
                                   "model t ford", "jet engine", "jet plane", "jet aircraft"),
    ("music", "7828cf627a"): ("jazz",),
    ("airplane", "5daca98944"): ("airplane", "air-plane"),
    ("einstein", "fa5b61109c"): ("einstein",),
    ("kitty_hawk", "14039c5bb3"): ("kitty hawk",),
    ("world_war", "4d4a492e5e"): ("world war",),
    ("bolshevik", "11ed0c1c05"): ("bolshevi",),
    ("atomic_energy", "64a1cf598a"): ("atomic energy",),
}


def _lower(text: str) -> str:
    """Lowercase without changing length (str.lower can, for a few non-ASCII letters)."""
    return text.lower() if text.isascii() else text.translate(_LOWER)


def _anchors(rule: leakage.Rule) -> tuple[str, ...] | None:
    key = (rule.name, hashlib.sha1(rule.pattern.pattern.encode()).hexdigest()[:10])
    return _ANCHORS.get(key)


def _gate(rule: leakage.Rule) -> re.Pattern[str] | None:
    pat = rule.pattern
    if not pat.flags & re.IGNORECASE or re.search(r"\\[A-Z]", pat.pattern):
        return None  # case-sensitive already, or has escapes that lowercasing would change
    return re.compile(pat.pattern.lower(), pat.flags & ~re.IGNORECASE)


_ALL_RULES = (*leakage.BOILERPLATE_RULES, *leakage.HARD_RULES, *leakage.SOFT_RULES)
_GATES = {r.name: _gate(r) for r in _ALL_RULES}
_RULE_ANCHORS = {r.name: _anchors(r) for r in _ALL_RULES}


def _candidate_paragraphs(rule: leakage.Rule, lowered: str, starts: list[int], n: int):
    anchors, gate = _RULE_ANCHORS[rule.name], _GATES[rule.name]
    if anchors is None and gate is None:
        return range(n)
    idx: set[int] = set()
    if anchors is not None:
        for anchor in anchors:
            i = lowered.find(anchor)
            while i != -1:
                a = bisect.bisect_right(starts, i) - 1
                b = bisect.bisect_right(starts, i + len(anchor) - 1) - 1
                idx.update(range(a, b + 1))
                i = lowered.find(anchor, i + 1)
    else:
        for m in gate.finditer(lowered):
            a = bisect.bisect_right(starts, m.start()) - 1
            b = bisect.bisect_right(starts, max(m.start(), m.end() - 1)) - 1
            idx.update(range(a, b + 1))
    return idx


def _fast_strip_boilerplate(text: str) -> tuple[str, int]:
    paras = leakage._PARAGRAPH_SPLIT.split(leakage.strip_gutenberg(text))
    starts, pos = [], 0
    for p in paras:
        starts.append(pos)
        pos += len(p) + 2
    lowered = _lower("\n\n".join(paras))  # offsets match `starts`
    candidates: dict[int, list[leakage.Rule]] = {}
    for rule in leakage.BOILERPLATE_RULES:
        for i in _candidate_paragraphs(rule, lowered, starts, len(paras)):
            candidates.setdefault(i, []).append(rule)
    kept, removed = [], 0
    for i, para in enumerate(paras):
        if any(rule.pattern.search(para) for rule in candidates.get(i, ())):
            removed += 1
        else:
            kept.append(para)
    return "\n\n".join(kept).strip(), removed


def fast_filter_document(text: str, review_threshold: float = leakage.DEFAULT_REVIEW_THRESHOLD) -> leakage.FilterResult:
    cleaned, removed = _fast_strip_boilerplate(text)
    lowered = _lower(cleaned)
    collapsed = " ".join(lowered.split())

    def may_match(rule: leakage.Rule) -> bool:
        anchors = _RULE_ANCHORS[rule.name]
        if anchors is not None:
            return any(a in collapsed for a in anchors)
        gate = _GATES[rule.name]
        return gate is None or gate.search(lowered) is not None

    hard_hits = [leakage.Hit(rule.name, m.group(0)) for rule in leakage.HARD_RULES
                 if may_match(rule) and (m := rule.pattern.search(cleaned))]
    soft_hits: list[leakage.Hit] = []
    score = 0.0
    for rule in leakage.SOFT_RULES:
        if not may_match(rule):
            continue
        for match in sorted({m.group(0).lower() for m in rule.pattern.finditer(cleaned)}):
            soft_hits.append(leakage.Hit(rule.name, match))
            score += rule.weight
    if hard_hits:
        decision = leakage.Decision.REJECT
    elif score >= review_threshold:
        decision = leakage.Decision.REVIEW
    else:
        decision = leakage.Decision.KEEP
    return leakage.FilterResult(decision, cleaned, hard_hits, soft_hits, score, removed)


def check_fast_filter(paths: list[Path]) -> int:
    """Number of books on which fast_filter_document differs from leakage.filter_document."""
    bad = 0
    for path in paths:
        text = strip_pg_remnants(path.read_text(encoding="utf-8", errors="replace"))
        if fast_filter_document(text) != leakage.filter_document(text):
            bad += 1
            print(f"fast filter differs on {path}", file=sys.stderr)
    return bad


# ---------------------------------------------------------------- per book

def stage1(raw_path: Path) -> dict:
    """Flight-independent filtering: Gutenberg remnants, leakage filter, English score."""
    text = strip_pg_remnants(raw_path.read_text(encoding="utf-8", errors="replace"))
    result = fast_filter_document(text)
    lang = english.score_english(result.text)
    return {
        "leakage": result.decision.value,
        "leakage_reasons": result.reasons()[:10],
        "lang": lang,
        "text": result.text if result.decision is leakage.Decision.KEEP else "",
    }


# ---------------------------------------------------------------- stage 2

_VERSE_LINE_MAX = 55


def unwrap(paragraph: str) -> str:
    """Join hard-wrapped lines with spaces; keep line breaks in verse-like blocks."""
    lines = [" ".join(line.split()) for line in paragraph.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) >= 3 and max(map(len, lines)) <= _VERSE_LINE_MAX:
        return "\n".join(lines)
    return " ".join(lines)


def process(task: tuple[str, str, str, str]) -> dict:
    book_id, title, year, raw = task
    out = {"id": book_id, "title": title, "year": year, "split": split_of(book_id),
           "decision": "kept", "reason": "", "removed": 0, "main": "", "control": "",
           "hits": {}, "hit_paragraphs": [], "train_removed": []}
    rec = stage1(Path(raw))
    if rec["leakage"] == "reject":
        out.update(decision="dropped_leakage", reason="; ".join(rec["leakage_reasons"]))
        return out
    if rec["leakage"] == "review":
        out.update(decision="dropped_review", reason="; ".join(rec["leakage_reasons"]))
        return out
    if len(rec["text"]) < MIN_BOOK_CHARS:
        out.update(decision="dropped_empty", reason=f"{len(rec['text'])} chars after cleaning")
        return out
    lang = rec["lang"]
    if not lang.is_english:
        out.update(decision="dropped_nonenglish", reason=lang.reason())
        return out

    paragraphs = flight.split_paragraphs(rec["text"])
    control = [u for u in map(unwrap, paragraphs) if u]
    out["control"] = "\n\n".join(control)
    scrub = flight.scrub_paragraphs(paragraphs, title)
    out["hits"] = dict(scrub.hits)
    out["removed"] = len(scrub.removed_paragraphs)
    out["hit_paragraphs"] = [unwrap(p) for p in scrub.hit_paragraphs]
    if scrub.dropped_book:
        out.update(decision="dropped_flight_book", reason=scrub.reason)
    else:
        out["main"] = "\n\n".join(u for u in map(unwrap, scrub.text.split("\n\n")) if u) if scrub.text else ""
    return out


def _tasks(args, meta):
    ids = []
    for book_id, (title, year) in meta.items():
        if not year.strip().isdigit() or int(year) > 1902 or not is_before_cutoff(year):
            continue
        ids.append(book_id)
    ids.sort(key=lambda i: (len(i), i))
    present = [i for i in ids if (args.raw / f"{i}.txt").exists()]
    missing = len(ids) - len(present)
    if args.limit:
        present = present[: args.limit]
    tasks = [(i, meta[i][0], meta[i][1], str(args.raw / f"{i}.txt")) for i in present]
    return tasks, len(ids), missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("/home/user/data/pg19_raw"))
    parser.add_argument("--metadata", type=Path, default=Path("/home/user/data/metadata.csv"))
    parser.add_argument("--out", type=Path, default=Path("/home/user/data/corpus"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N eligible books")
    args = parser.parse_args(argv)

    meta = load_metadata(args.metadata)
    tasks, n_eligible, missing = _tasks(args, meta)
    print(f"{n_eligible} books dated <= 1902; {missing} not downloaded; processing {len(tasks)}", file=sys.stderr)

    out = args.out  # written in place: a second copy would not fit on this disk
    for sub in ("main", "control"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shards = {(c, s): (out / c / f"{s}.jsonl").open("w", encoding="utf-8")
              for c in ("main", "control") for s in ("train", "val")}
    eval_f = (out / "flight_eval.jsonl").open("w", encoding="utf-8")
    manifest_f = (out / "manifest.csv").open("w", newline="", encoding="utf-8")
    manifest = csv.writer(manifest_f)
    manifest.writerow(["id", "title", "year", "split", "decision", "flight_paragraphs_removed",
                       "chars_main", "chars_control", "reason"])

    rng = random.Random(1903)
    train_sample: list[dict] = []
    seen_train = 0
    decisions: Counter[str] = Counter()
    decisions_by_split: Counter[str] = Counter()
    term_hits: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    books: Counter[str] = Counter()
    removed_total = hit_total = eval_n = 0
    start = time.time()

    with Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap(process, tasks, chunksize=2), 1):
            decisions[r["decision"]] += 1
            decisions_by_split[f"{r['split']}:{r['decision']}"] += 1
            term_hits.update(r["hits"])
            if r["control"]:
                removed_total += r["removed"] if r["decision"] == "kept" else 0
                hit_total += len(r["hit_paragraphs"])
                for corpus in ("main", "control"):
                    if r[corpus]:
                        rec = {"id": r["id"], "title": r["title"], "year": int(r["year"]), "text": r[corpus]}
                        shards[(corpus, r["split"])].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        chars[f"{corpus}:{r['split']}"] += len(r[corpus])
                        books[f"{corpus}:{r['split']}"] += 1
                source = "flight_book" if r["decision"] == "dropped_flight_book" else "passage"
                for k, para in enumerate(r["hit_paragraphs"]):
                    item = {"id": r["id"], "title": r["title"], "source": source, "paragraph": k, "text": para}
                    if r["split"] == "val":
                        eval_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        eval_n += 1
                    else:  # reservoir sample
                        seen_train += 1
                        if len(train_sample) < TRAIN_SAMPLE_SIZE:
                            train_sample.append(item)
                        elif (j := rng.randrange(seen_train)) < TRAIN_SAMPLE_SIZE:
                            train_sample[j] = item
            manifest.writerow([r["id"], r["title"], r["year"], r["split"], r["decision"],
                               r["removed"] if r["decision"] == "kept" else 0,
                               len(r["main"]), len(r["control"]), r["reason"][:300]])
            if n % 250 == 0 or n == len(tasks):
                rate = n / (time.time() - start)
                print(f"[{n}/{len(tasks)}] {rate:.1f} books/s, eta {(len(tasks) - n) / rate / 60:.1f} min; "
                      f"{dict(decisions)}", file=sys.stderr, flush=True)

    for f in (*shards.values(), eval_f, manifest_f):
        f.close()
    with (out / "flight_removed_train_sample.jsonl").open("w", encoding="utf-8") as f:
        for item in train_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "books_dated_le_1902": n_eligible,
        "books_missing": missing,
        "books_processed": len(tasks),
        "decisions": dict(decisions),
        "decisions_by_split": dict(sorted(decisions_by_split.items())),
        "books": dict(books),
        "chars": dict(chars),
        "est_tokens_chars_div_4": {k: v // 4 for k, v in chars.items()},
        "flight_paragraphs_removed_from_kept_books": removed_total,
        "flight_hit_paragraphs": hit_total,
        "flight_eval_paragraphs": eval_n,
        "flight_removed_train_sample": len(train_sample),
        "top_flight_terms": dict(term_hits.most_common(40)),
        "flight_policy": {"neighbours": flight.NEIGHBOURS, "bridge": flight.BRIDGE,
                          "book_min_hit_paragraphs": flight.BOOK_MIN_HIT_PARAGRAPHS,
                          "book_hit_fraction": flight.BOOK_HIT_FRACTION},
        "seconds": round(time.time() - start),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({k: stats[k] for k in ("decisions", "books", "chars", "flight_eval_paragraphs")}, indent=2),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
