"""Flight-word facts: how the tokenizer splits them, and how often the corpora contain them.

The tokenizer was trained on the scrubbed ``main`` corpus only, so a word that
never occurs there gets no merges of its own and is spelled out in several
pieces, while an everyday word of similar general frequency is usually a single
token. The corpus counts make the "similar frequency" claim checkable: the
control corpus shows how common each flight word and its comparison words are
in unscrubbed 19th-century books.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

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


def build_counter_pattern(words: Iterable[str], phrases: Iterable[str] = PHRASES) -> re.Pattern[str]:
    alts = sorted({re.escape(w).replace(r"\ ", r"[\s-]+") for w in (*words, *phrases)}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


def _canonical(match: str) -> str:
    return re.sub(r"[\s-]+", " ", match.lower())


def iter_book_texts(path: str | Path, max_books: int | None = None) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if max_books is not None and k >= max_books:
                return
            if line.strip():
                yield json.loads(line).get("text", "")


def count_corpus(path: str | Path, pairs: Sequence[tuple[str, Sequence[str]]] = WORD_PAIRS,
                 max_books: int | None = None, audit: bool = True) -> dict:
    """Exact (case-insensitive, whole-word) counts of the comparison words, plus a flight-lexicon audit.

    The audit runs ``prewright.flight`` on every book: ``flight_hits`` counts
    lexicon matches (main should have none), ``books_with_hits`` the books
    containing at least one.
    """
    pattern = build_counter_pattern(_all_words(pairs))
    counts: Counter[str] = Counter()
    hit_terms: Counter[str] = Counter()
    books = chars = books_with_hits = 0
    examples: list[str] = []
    find_hits = None
    if audit:
        try:
            from prewright.flight import find_flight_hits as find_hits
        except ImportError:  # pragma: no cover
            audit = False
    for text in iter_book_texts(path, max_books):
        books += 1
        chars += len(text)
        for m in pattern.finditer(text):
            counts[_canonical(m.group(0))] += 1
        if audit:
            hits = find_hits(text)
            if hits:
                books_with_hits += 1
                hit_terms.update(h.term for h in hits)
                if len(examples) < 10:
                    examples.append(hits[0].text)
    out = {"books": books, "chars": chars, "counts": dict(sorted(counts.items())), "max_books": max_books}
    if audit:
        out["audit"] = {"flight_hits": sum(hit_terms.values()), "books_with_hits": books_with_hits,
                        "by_term": dict(hit_terms.most_common()), "examples": examples}
    return out
