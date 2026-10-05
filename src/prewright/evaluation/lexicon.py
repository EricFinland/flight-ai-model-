"""Flight-word facts: how the tokenizer splits them, and how often the corpora contain them.

The tokenizer was trained on the scrubbed ``main`` corpus only, so a word that
never occurs there gets no merges of its own and is spelled out in several
pieces, while an everyday word of similar general frequency is usually a single
token. The corpus counts make the "similar frequency" claim checkable: the
control corpus shows how common each flight word and its comparison words are
in unscrubbed 19th-century books.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence

from prewright.evaluation.interface import Codec

# (flight word, everyday words of similar shape and comparable frequency in ordinary 19th-century prose)
WORD_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("balloon", ("saloon", "baboon", "lagoon")),
    ("balloons", ("saloons", "lagoons", "cartoons")),
    ("ballooning", ("bloodhound", "ballasting")),
    ("aeronaut", ("argonaut", "astronomer")),
    ("aeronauts", ("argonauts", "astronomers")),
    ("aeronautics", ("hydraulics", "acoustics")),
    ("aerostat", ("acrobat", "barometer")),
    ("aerostatics", ("hydrostatics", "electrostatics")),
    ("airship", ("warship", "hardship")),
    ("airships", ("warships", "hardships")),
    ("aeroplane", ("aqueduct", "amphibian")),
    ("parachute", ("parasol", "paradise")),
    ("dirigible", ("eligible", "corrigible")),
    ("glider", ("slider", "spider")),
    ("aviation", ("navigation", "salvation")),
    ("aviator", ("navigator", "alligator")),
    ("zeppelin", ("javelin", "mandolin")),
    ("montgolfier", ("montgomery", "mountaineer")),
    ("aerodrome", ("hippodrome", "palindrome")),
)

# Multi-word phrases counted in the corpora (not tokenizer facts).
PHRASES: tuple[str, ...] = ("flying machine", "flying machines", "aerial navigation", "conquest of the air",
                            "steam navigation", "sewing machine", "steam engine", "railway train")


def tokenization_facts(codec: Codec, pairs: Sequence[tuple[str, Sequence[str]]] = WORD_PAIRS) -> dict:
    """Token counts and pieces of each word, as it appears mid-sentence (" word") and capitalised."""
    def info(word: str) -> dict:
        out = {}
        for form in (" " + word, " " + word.capitalize(), word):
            ids = codec.encode(form)
            pieces = codec.token_pieces(ids) if hasattr(codec, "token_pieces") else [codec.decode([i]) for i in ids]
            out[repr(form)] = {"n_tokens": len(ids), "pieces": pieces}
        out["n_tokens"] = out[repr(" " + word)]["n_tokens"]
        return out

    rows = []
    for flight, everyday in pairs:
        rows.append({"flight": flight, "flight_info": info(flight),
                     "everyday": [{"word": w, **info(w)} for w in everyday]})
    f = [r["flight_info"]["n_tokens"] for r in rows]
    e = [x["n_tokens"] for r in rows for x in r["everyday"]]
    return {
        "rows": rows,
        "mean_tokens_flight": sum(f) / len(f) if f else float("nan"),
        "mean_tokens_everyday": sum(e) / len(e) if e else float("nan"),
        "single_token_flight": sum(1 for x in f if x == 1),
        "single_token_everyday": sum(1 for x in e if x == 1),
        "n_flight": len(f), "n_everyday": len(e),
    }


def _all_words(pairs: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    words = []
    for f, e in pairs:
        words += [f, *e]
    return sorted(set(words))


class WordCounter:
    """Exact whole-word, case-insensitive counts of a fixed word/phrase list.

    Faster than one big alternation: the text is lower-cased once, each word is
    looked for only if its first word occurs as a substring, and the pattern
    starts with a literal so the regex engine can skip ahead. "Whole word"
    means not preceded or followed by a letter, so ``_balloon_`` (Gutenberg
    italics) counts. Inside phrases, spaces match any run of spaces or hyphens
    ("flying-machine").
    """

    def __init__(self, words: Iterable[str], phrases: Iterable[str] = PHRASES):
        self.items = sorted({w.lower() for w in (*words, *phrases)})
        self.patterns = {w: re.compile(re.escape(w).replace(r"\ ", r"[\s-]+") + r"(?![^\W\d_])")
                         for w in self.items}

    def update(self, counts: Counter[str], text: str) -> None:
        low = text.lower()
        for w in self.items:
            if w.split(" ", 1)[0] not in low:
                continue
            for m in self.patterns[w].finditer(low):
                s = m.start()
                if s == 0 or not low[s - 1].isalpha():
                    counts[w] += 1


_BOOK_ID = re.compile(r'"id":\s*"((?:[^"\\]|\\.)*)"')


def trained_book_filter(meta_path: str | Path) -> tuple[Callable[[str], bool], dict] | None:
    """Predicate on JSONL lines: is this book inside the token file the model was trained on?

    ``scripts/tokenize_corpus.py`` ranks books by sha1(salt:id) and caps the
    training token file at a rank ``cutoff`` (about 1,400 of 16,000 books in the
    first build), recorded in ``train.meta.json``. Counting the whole
    ``train.jsonl`` would describe text neither model saw. Returns None when the
    meta file is missing or there was no cap (every book was used).
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if not meta.get("cutoff"):
        return None
    salt, cutoff = meta["salt"], bytes.fromhex(meta["cutoff"])

    def keep(line: str) -> bool:
        m = _BOOK_ID.search(line, 0, 300)  # the same id lookup as tokenize_corpus.index_jsonl
        book_id = m.group(1) if m else json.loads(line).get("id")
        return book_id is not None and hashlib.sha1(f"{salt}:{book_id}".encode()).digest() <= cutoff

    return keep, {"meta": str(meta_path), "cutoff": meta["cutoff"], "salt": salt, "books": meta.get("books")}


def iter_book_lines(path: str | Path, max_books: int | None = None,
                    keep: Callable[[str], bool] | None = None) -> Iterable[str]:
    """Non-empty JSONL lines; with ``keep``, only the lines it accepts (``max_books`` counts kept lines)."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_books is not None and n >= max_books:
                return
            if line.strip() and (keep is None or keep(line)):
                n += 1
                yield line


def iter_book_texts(path: str | Path, max_books: int | None = None) -> Iterable[str]:
    for line in iter_book_lines(path, max_books):
        yield json.loads(line).get("text", "")


_WORKER: dict = {}


def _count_one(args: tuple[str, bool]) -> tuple[int, Counter[str], list]:
    """Count one book (a JSONL line). Returns (chars, word counts, [(term, matched text), ...])."""
    line, audit = args
    if "counter" not in _WORKER:
        _WORKER["counter"] = WordCounter(_all_words(WORD_PAIRS))
    text = json.loads(line).get("text", "")
    counts: Counter[str] = Counter()
    _WORKER["counter"].update(counts, text)
    hits = []
    if audit:
        from prewright.flight import find_flight_hits
        hits = [(h.term, h.text) for h in find_flight_hits(text)]
    return len(text), counts, hits


def count_corpus(path: str | Path, max_books: int | None = None, audit: bool = True, workers: int = 1,
                 keep: Callable[[str], bool] | None = None) -> dict:
    """Exact (case-insensitive, whole-word) counts of the ``WORD_PAIRS`` words and ``PHRASES``, plus an audit.

    The audit runs ``prewright.flight.find_flight_hits`` on every book:
    ``flight_hits`` counts lexicon matches (main should have none),
    ``books_with_hits`` the books containing at least one. Reads about 6 MB/s
    per worker. ``keep`` (see ``trained_book_filter``) restricts the pass to the
    books that are in the training token file.
    """
    counts: Counter[str] = Counter()
    hit_terms: Counter[str] = Counter()
    books = chars = books_with_hits = 0
    examples: list[str] = []
    jobs = ((line, audit) for line in iter_book_lines(path, max_books, keep))
    if workers > 1:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap(_count_one, jobs, chunksize=8)
    else:
        pool, results = None, map(_count_one, jobs)
    try:
        for n_chars, c, hits in results:
            books += 1
            chars += n_chars
            counts.update(c)
            if hits:
                books_with_hits += 1
                hit_terms.update(t for t, _ in hits)
                if len(examples) < 10:
                    examples.append(hits[0][1])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    out = {"books": books, "chars": chars, "counts": dict(sorted(counts.items())), "max_books": max_books}
    if audit:
        out["audit"] = {"flight_hits": sum(hit_terms.values()), "books_with_hits": books_with_hits,
                        "by_term": dict(hit_terms.most_common()), "examples": examples}
    return out
