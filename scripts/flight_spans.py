"""Locate flight-lexicon hits as character spans of the original text.

`prewright.flight.find_flight_hits` matches on a normalised copy of each
paragraph (accents and ligatures folded, underscores turned into spaces,
whitespace collapsed) and reports only the matched words. To draw a
redaction bar over a word we need its position in the text as printed, so
this module builds the same normalised string while remembering, for every
character, where it came from, runs the scrub's own TERMS on it with the same
context rules, and maps each match back.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prewright import flight  # noqa: E402


# A word broken at a line end ("bal-\nloon"); in unwrapped paragraphs the newline is a space.
_BROKEN_WORD = re.compile(r"(?<=[^\W\d_])[-\u2010\u2011][ \t]*\n?\s*(?=[^\W\d_])")


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    src: list[int] = []
    skip = set()
    for m in _BROKEN_WORD.finditer(text):
        if m.end() - m.start() > 1:  # hyphen plus a break, not an ordinary compound like "air-ship"
            skip.update(range(m.start(), m.end()))
    for i, ch in enumerate(text):
        if i in skip:
            continue
        rep = ch.translate(flight._FOLD) if not ch.isascii() else ch
        if rep == "_":
            rep = " "
        for r in rep:
            if r.isspace():
                if not out or out[-1] == " ":
                    continue
                r = " "
            out.append(r)
            src.append(i)
    while out and out[-1] == " ":
        out.pop()
        src.pop()
    return "".join(out), src


def flight_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, term) for every lexicon hit in one paragraph, in the original text, merged."""
    norm, src = _normalise_with_map(text)
    found = []
    for term in flight.TERMS:
        matches = [m for m in term.pattern.finditer(norm) if m.end() > m.start()]
        if matches and (term.context is None or term.context.search(norm)):
            found += [(src[m.start()], src[m.end() - 1] + 1, term.name) for m in matches]
    found.sort()
    merged: list[tuple[int, int, str]] = []
    for s, e, t in found:
        if merged and s <= merged[-1][1]:
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, max(pe, e), pt)
        else:
            merged.append((s, e, t))
    return merged


def segments(text: str) -> list[tuple[str, bool]]:
    """The paragraph as (text, is_flight) pieces, in order."""
    out, pos = [], 0
    for s, e, _ in flight_spans(text):
        if s > pos:
            out.append((text[pos:s], False))
        out.append((text[s:e], True))
        pos = e
    if pos < len(text):
        out.append((text[pos:], False))
    return out
