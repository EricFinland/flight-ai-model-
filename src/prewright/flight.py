"""Scrub all knowledge of human flight from a book.

The policy (docs/PLAN.md, docs/flight-scrub.md) removes real or plausible
human flight and its technology: balloons of every kind, airships, flying
machines, gliders, parachutes, aerial navigation, and the names of aeronauts
and flight experimenters. It keeps what ordinary English needs: birds, bats and
insects flying, *flight* as fleeing, flights of stairs, flying colours, and
myth and fairy tale (Icarus, angels, witches, flying carpets).

Matching runs on a normalised copy of each paragraph: accents and ligatures
folded ("aérostat", "æronaut"), soft hyphens dropped, words re-joined across
hyphenated line breaks ("bal-\\nloon"), and whitespace collapsed. A hit drops
its paragraph plus one on each side; hits a few paragraphs apart drop the whole
passage between them. A book whose title matches, or which has many hits, is
dropped whole: cutting it would leave a skeleton that still implies flight.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Term:
    name: str
    pattern: re.Pattern[str]
    # When set, the term counts only if this also matches the same paragraph
    # (for common surnames and words with an innocent main sense).
    context: re.Pattern[str] | None = None
    note: str = ""


def _rx(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_SEP = r"[- ]?"  # "air-ship", "air ship", "airship" (line-break hyphens are already joined)

# Always a hit, wherever they appear.
TERMS: tuple[Term, ...] = (
    # Balloons of every kind, including toys, sails and sleeves: the word itself invites the inference.
    Term("balloon", _rx(r"balloon\w*|\bbal+o+ns?\b|ballon(?:s|e?ts?|fahrt\w*)?\b|\bbaloon\w*"),
         note="covers air-/fire-/hydrogen/toy balloons, ballooning, balloonist, French/German 'ballon'"),
    Term("aerostat", _rx(r"aerostat\w*|\bcharlieres?\b|montgolfiere?s?\b")),
    Term("aeronaut", _rx(r"aeronaut\w*|\baero[- ]?(?:nef|plane|drome|dynamic|motor|club|curve|car)\w*")),
    Term("airship", _rx(rf"\bair{_SEP}ships?\b|\bsky{_SEP}ships?\b|\bluft{_SEP}(?:schiff|ballon)\w*")),
    Term("airplane", _rx(rf"\bair{_SEP}planes?\b")),
    Term("airman", _rx(r"\bair-?m[ae]n\b")),
    Term("dirigible", _rx(r"\bdirigibles?\b(?!\s+(?:torpedo|torpedoes|mines?|spirits?)\b)|\bdirigeables?\b"),
         note="'dirigible torpedo' (Whitehead) is naval, not flight"),
    Term("zeppelin", _rx(r"\bzeppelin\w*")),
    Term("flying_machine", _rx(rf"\bflying{_SEP}(?:machines?|ships?|apparatus(?:es)?|contrivances?|models?)\b")),
    Term("glider", _rx(r"(?<!mr )(?<!mr\. )(?<!mrs )(?<!mrs\. )(?<!miss )\bgliders?\b|\bgliding\s+(?:machines?|apparatus|experiments?|flights?)\b")),
    Term("soaring_machine", _rx(r"\bsoaring\s+(?:machines?|apparatus|experiments?)\b")),
    Term("ornithopter", _rx(r"\bornithopters?\b"), note="not the butterfly genus Ornithoptera"),
    Term("helicopter", _rx(r"\bhelicopt\w*")),
    Term("parachute", _rx(r"parachut\w*")),
    Term("aviation", _rx(r"\baviat(?:ion|ors?|rix|ress|ic)\b")),
    Term("man_lifting_kite", _rx(
        r"\bman-?lifting\b|\bman\s+lifting\s+kites?\b|\bkites?\b(?:\s+\w+){0,5}?\s+(?:lift|lifted|lifting|raise[sd]?|raising|carr(?:y|ies|ied|ying)|bore|bear)"
        r"\s+(?:up\s+)?(?:a\s+|the\s+|an\s+)?(?:man|men|observer|person|passenger|soldier|aeronaut)s?\b")),
    Term("aerial_craft", _rx(
        r"\baerial\s+(?:navigat\w*|ships?|vessels?|machines?|cars?|voyag\w*|journey\w*|fleets?|navies|navy|"
        r"(?:steam[- ])?carriages?|steamers?|ascents?|ascensions?|excursions?|trips?|warfare|torpedo\w*|"
        r"squadrons?|cruisers?|locomotion|travel\w*|caravans?|screws?|propell\w*|tours?|passages?|vehicles?|boats?|"
        r"craft|crafts|machinery|contrivances?)\b")),
    Term("navigation_of_the_air", _rx(
        r"\bnavigat\w*\s+(?:of\s+|through\s+|in\s+)?the\s+(?:air|atmosphere|aerial\s+ocean)\b|\bair[- ]navigat\w*|"
        r"\bnavigable\s+balloon|\bair[- ]?voyag\w*|\bvoyag\w*\s+(?:through|in|across)\s+the\s+air\b")),
    Term("conquest_of_the_air", _rx(r"\b(?:conquest|mastery|subjugation)\s+of\s+the\s+air\b")),
    Term("heavier_than_air", _rx(
        r"\b(?:heavier|lighter)-than-air\b|\b(?:heavier|lighter)\s+than\s+(?:the\s+)?air\s+(?:machines?|flying|apparatus|craft|ships?|"
        r"vessels?|flight|navigation|type|system|principle|class)\b"),
        note="'lighter/heavier than air' as plain physics of gases is kept"),
    Term("art_of_flying", _rx(
        r"\b(?:art|science|problem|possibility|secret)\s+of\s+(?:aerial\s+|mechanical\s+|artificial\s+|human\s+)?"
        r"fl(?:ight|ying)\b(?!\s+(?:hawks?|kites?|falcons?|of\s+(?:birds|the\s+bird|insects|bats)))|"
        r"\b(?:mechanical|artificial|human|man-?)\s*flight\b|\bflight\s+of\s+man\b|"
        r"\b(?:artificial|mechanical)\s+wings\b")),
    Term("airy_navies", _rx(
        r"\b(?:airy|aerial|air)\s+nav(?:y|ies)\b|argosies\s+of\s+magic\s+sails|pilots\s+of\s+the\s+purple\s+twilight"),
        note="Tennyson, Locksley Hall (1842)"),
    Term("gas_bag", _rx(r"\bgas[- ]?bags?\b"),
         _rx(r"inflat\w*|ascen[dt]\w*|\bhydrogen\b|\bvalve|\benvelope\b|\baerial\b|\bcar\b|\bnetting\b"),
         note="also slang for a windbag"),
    # Names that are unambiguous.
    Term("montgolfier", _rx(r"montgolfier\w*")),
    Term("lunardi", _rx(r"\blunardi\b")),
    Term("garnerin", _rx(r"\bgarnerin\b")),
    Term("santos_dumont", _rx(r"\bsantos[- ]?dumont\b")),
    Term("lilienthal", _rx(r"\blilienthal\b"),
         _rx(r"\bfly\w*|\bflight|\bflew\b|\bglid\w*|\bsoar\w*|\bwings?\b|\baerial\b|\bmachine|\bapparatus\b"),
         note="also Samuel Lilienthal (homoeopath) and Max Lilienthal (rabbi)"),
    Term("chanute", _rx(r"\bchanute\b")),
    Term("kitty_hawk", _rx(r"\bkitty\s*hawk\b")),
    Term("pilatre_de_rozier", _rx(r"\bpilatre\b|\bde\s+rozier\b")),
    Term("zambeccari", _rx(r"\bzambeccari\b")),
    Term("tissandier", _rx(r"\btissandier\b")),
    Term("gusmao", _rx(r"\bgusmao\b|\bpassarola\b")),
    Term("wright_brothers", _rx(r"\b(?:wilbur|orville)\s+wright\b|\bwright\s+brothers\b|\bbrothers\s+wright\b")),
    Term("lana_terzi", _rx(r"\b(?:francesco|father|padre)\s+(?:de\s+)?lana\b|\blana[- ](?:de\s+)?terzi\b")),
    Term("flying_automata", _rx(
        r"\b(?:pigeon|dove)\s+of\s+archytas|\barchytas(?:'s)?\s+(?:wooden\s+)?(?:pigeon|dove)|"
        r"\b(?:eagle|fly)\s+of\s+regiomontanus|\bregiomontanus(?:'s)?\s+(?:iron\s+)?(?:eagle|fly)"),
        note="legendary flying automata, listed as early flying machines"),
    Term("malmesbury_flier", _rx(r"\b(?:oliver|elmer|eilmer)\s+(?:the\s+monk\s+)?of\s+malmesbury\b")),
    # Names that need flight context in the same paragraph.
    Term("blanchard", _rx(r"\bblanchard\b"),
         _rx(r"ascen[dt]\w*|\baerial\b|\bjeff(?:e)?ries\b|crossed\s+the\s+channel|crossing\s+of\s+the\s+channel|\bthrough\s+the\s+air\b"),
         note="common surname (Laman Blanchard, Blanchard Jerrold)"),
    Term("langley", _rx(r"\blangley\b"),
         _rx(r"\bfly\w*|\bflight|\bflew\b|\bsoar\w*|\baerial\b|\bwings?\b|\bmachine|\bplanes?\b|\bpotomac\b|\bmodel\b"),
         note="also an astronomer, and an English place-name"),
    Term("maxim", _rx(r"\bMaxim(?:'s)?\b", 0),
         _rx(r"\baerial\b|\blifting\s+(?:power|force)\b|\bsoar\w*|\bpropell\w*|\bplanes?\b|\baeroplane"),
         note="the word 'maxim' and the Maxim gun are kept"),
    Term("glaisher", _rx(r"\bglaisher\b|\bcoxwell\b"),
         _rx(r"ascen[dt]\w*|\baerial\b|\baltitude|\bfeet\s+high\b|\bmiles\s+high\b|glaisher.{0,40}coxwell|coxwell.{0,40}glaisher|\bheight\s+of\s+[\d,]+\s+feet"),
         note="J. W. L. Glaisher is also a mathematician"),
    Term("cayley", _rx(r"\bcayley\b"), _rx(r"\baerial\b|\bfly\w*|\bflight|\bglid\w*|\bwings?\b")),
    Term("flight_experimenters", _rx(
        r"\b(?:henson|stringfellow|wenham|pilcher|ader|hargrave|giffard|renard|krebs|nadar|andree|besnier|degen|bacqueville)\b"),
        _rx(r"\bfly\w*|\bflight|\bflew\b|\baerial\b|\bglid\w*|\bsoar\w*|\bwings?\b|ascen[dt]\w*|\bkites?\b|\bmachine"),
        note="surnames that are only flight-related in such context"),
)

UNCONDITIONAL_TERMS = tuple(t for t in TERMS if t.context is None)
CONTEXT_TERMS = tuple(t for t in TERMS if t.context is not None)

# One pattern for every unconditional term, for audits and quick screening.
# It runs on normalised text (see normalize); case-sensitive terms are folded.
FLIGHT_PATTERN: re.Pattern[str] = _rx("|".join(f"(?:{t.pattern.pattern})" for t in UNCONDITIONAL_TERMS))

# Every match of a term contains one of its anchors (lowercase literals, after
# normalize). Scanning for them with str.find is far faster than running the
# lexicon over a whole book, so each term's pattern only runs on paragraphs
# that contain one of its anchors. tests/test_flight.py checks the coverage.
TERM_ANCHORS: dict[str, tuple[str, ...]] = {
    "balloon": ("balloon", "ballon", "baloon", "balon"),
    "aerostat": ("aerostat", "charlier", "montgolf"),
    "aeronaut": ("aero",),
    "airship": ("airship", "air-ship", "air ship", "skyship", "sky-ship", "sky ship", "luft"),
    "airplane": ("airplane", "air-plane", "air plane"),
    "airman": ("airm", "air-m"),
    "dirigible": ("dirig",),
    "zeppelin": ("zeppelin",),
    "flying_machine": ("flying",),
    "glider": ("glider", "gliding"),
    "soaring_machine": ("soaring",),
    "ornithopter": ("ornithopter",),
    "helicopter": ("helicopt",),
    "parachute": ("parachut",),
    "aviation": ("aviat",),
    "man_lifting_kite": ("lifting", "kite"),
    "aerial_craft": ("aerial",),
    "navigation_of_the_air": ("navigat", "navigable", "voyag"),
    "conquest_of_the_air": ("of the air",),
    "heavier_than_air": ("than-air", "than air", "than the air"),
    "art_of_flying": ("of flight", "of flying", "flight of man", "flight", "wings"),
    "airy_navies": ("navy", "navies", "argosies", "pilots of the purple"),
    "gas_bag": ("gas-bag", "gas bag", "gasbag"),
    "montgolfier": ("montgolfier",),
    "lunardi": ("lunardi",),
    "garnerin": ("garnerin",),
    "santos_dumont": ("dumont",),
    "lilienthal": ("lilienthal",),
    "chanute": ("chanute",),
    "kitty_hawk": ("kitty",),
    "pilatre_de_rozier": ("pilatre", "rozier"),
    "zambeccari": ("zambeccari",),
    "tissandier": ("tissandier",),
    "gusmao": ("gusmao", "passarola"),
    "wright_brothers": ("wright",),
    "lana_terzi": ("lana",),
    "malmesbury_flier": ("malmesbury",),
    "flying_automata": ("archytas", "regiomontanus"),
    "blanchard": ("blanchard",),
    "langley": ("langley",),
    "maxim": ("maxim",),
    "glaisher": ("glaisher", "coxwell"),
    "cayley": ("cayley",),
    "flight_experimenters": ("henson", "stringfellow", "wenham", "pilcher", "ader", "hargrave", "giffard",
                             "renard", "krebs", "nadar", "andree", "besnier", "degen", "bacqueville"),
}
ANCHORS: tuple[str, ...] = tuple(sorted({a for anchors in TERM_ANCHORS.values() for a in anchors}))
_TERMS_BY_ANCHOR: dict[str, tuple[Term, ...]] = {
    a: tuple(t for t in TERMS if a in TERM_ANCHORS[t.name]) for a in ANCHORS
}

# Granularity. A hit drops its paragraph and NEIGHBOURS on each side; two hits
# at most BRIDGE paragraphs apart drop everything between them too.
NEIGHBOURS = 1
BRIDGE = 4
# A book is about flight (and dropped whole) if more than BOOK_HIT_FRACTION of
# its paragraphs hit (with at least BOOK_FRACTION_MIN_HITS of them), or if it
# has BOOK_MIN_HIT_PARAGRAPHS hits however long it is. Tuned on PG-19: big
# compendia (collected works, dictionaries of quotations, science reports)
# with a dozen scattered mentions fall under 0.1% and are scrubbed instead.
BOOK_MIN_HIT_PARAGRAPHS = 30
BOOK_HIT_FRACTION = 0.005
BOOK_FRACTION_MIN_HITS = 3



def _fold_table() -> dict[int, str]:
    table = {ord("æ"): "ae", ord("Æ"): "Ae", ord("œ"): "oe", ord("Œ"): "Oe", 0xAD: "", ord("ſ"): "s"}
    for code in range(0xC0, 0x250):  # Latin-1 Supplement and Latin Extended-A/B
        base = "".join(c for c in unicodedata.normalize("NFKD", chr(code)) if not unicodedata.combining(c))
        if code not in table and base != chr(code) and base.isascii():
            table[code] = base
    return table


_FOLD = _fold_table()
_LINE_BREAK_HYPHEN = re.compile(r"[-\u2010\u2011](?<=[^\W\d_].)[ \t]*\n\s*(?=[^\W\d_])")
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
# Paragraph-less books (no blank lines) are split at indented lines or every
# few lines, so a hit cannot take a whole chapter with it.
_MAX_PARAGRAPH_CHARS = 4000
_INDENTED_LINE = re.compile(r"\n(?=[ \t]+\S)")


def normalize(text: str) -> str:
    """Fold accents and ligatures, re-join hyphenated line breaks, collapse whitespace."""
    if not text.isascii():
        text = text.translate(_FOLD)
    if "\n" in text:
        text = _LINE_BREAK_HYPHEN.sub("", text)
    return " ".join(text.split())


def _split_long(para: str) -> list[str]:
    parts = _INDENTED_LINE.split(para)
    if len(parts) == 1:
        lines = para.split("\n")
        parts = ["\n".join(lines[i:i + 12]) for i in range(0, len(lines), 12)]
    return [p for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; break up over-long blocks. Paragraph text is unchanged."""
    out = []
    for para in _PARAGRAPH_SPLIT.split(text):
        if not para.strip():
            continue
        if len(para) > _MAX_PARAGRAPH_CHARS:
            out.extend(_split_long(para))
        else:
            out.append(para)
    return out


@dataclass(frozen=True)
class Hit:
    term: str
    text: str
    paragraph: int = 0


def _paragraph_hits(norm: str, index: int = 0, terms: tuple[Term, ...] = TERMS) -> list[Hit]:
    hits = []
    for term in terms:
        matches = list(term.pattern.finditer(norm))
        if matches and (term.context is None or term.context.search(norm)):
            hits.extend(Hit(term.name, m.group(0).lower(), index) for m in matches)
    return hits


def normalize_paragraphs(paragraphs: list[str]) -> list[str]:
    """normalize() applied to each paragraph, in one pass over the whole book."""
    if not paragraphs:
        return []
    return [p.strip() for p in normalize("\x00".join(paragraphs)).split("\x00")]


def _candidate_terms(norms: list[str]) -> dict[int, set[str]]:
    """Paragraph index -> names of the terms whose anchors it contains."""
    lowered = "\x00".join(norms).lower()
    starts, pos = [], 0
    for p in norms:
        starts.append(pos)
        pos += len(p) + 1
    found: dict[int, set[str]] = {}
    for anchor in ANCHORS:
        i = lowered.find(anchor)
        while i != -1:
            k = bisect_right(starts, i) - 1
            found.setdefault(k, set()).update(t.name for t in _TERMS_BY_ANCHOR[anchor])
            nxt = starts[k + 1] if k + 1 < len(starts) else len(lowered)
            i = lowered.find(anchor, nxt)
    return found


def _hits_by_paragraph(paragraphs: list[str]) -> list[Hit]:
    norms = normalize_paragraphs(paragraphs)
    hits: list[Hit] = []
    for i, names in sorted(_candidate_terms(norms).items()):
        hits.extend(_paragraph_hits(norms[i], i, tuple(t for t in TERMS if t.name in names)))
    return hits


def find_flight_hits(text: str) -> list[Hit]:
    """All lexicon hits in `text`, with the index of the paragraph they fall in."""
    return _hits_by_paragraph(split_paragraphs(text))


def contains_flight(text: str) -> bool:
    """True if any part of `text` (one paragraph, a title, a whole book) matches the lexicon."""
    return any(_paragraph_hits(normalize(p)) for p in split_paragraphs(text) or [text])


@dataclass
class ScrubResult:
    text: str  # paragraphs kept, joined by blank lines ("" if the book is dropped)
    dropped_book: bool = False
    reason: str = ""
    removed_paragraphs: list[str] = field(default_factory=list)  # every paragraph cut, in order
    hit_paragraphs: list[str] = field(default_factory=list)  # the subset containing a hit
    hits: Counter[str] = field(default_factory=Counter)  # term name -> matches
    total_paragraphs: int = 0


def is_flight_book(n_hit_paragraphs: int, n_paragraphs: int) -> bool:
    if n_hit_paragraphs >= BOOK_MIN_HIT_PARAGRAPHS:
        return True
    return n_hit_paragraphs >= BOOK_FRACTION_MIN_HITS and n_hit_paragraphs > BOOK_HIT_FRACTION * n_paragraphs


def removal_mask(hit_indices: list[int], n: int, neighbours: int = NEIGHBOURS, bridge: int = BRIDGE) -> list[bool]:
    """Paragraphs to drop: each hit +/- `neighbours`, plus gaps of <= `bridge` between hits."""
    drop = [False] * n
    previous = None
    for i in sorted(set(hit_indices)):
        start = previous + 1 if previous is not None and i - previous <= bridge + 1 else max(0, i - neighbours)
        for j in range(start, min(n, i + neighbours + 1)):
            drop[j] = True
        previous = i
    return drop


def scrub_paragraphs(paragraphs: list[str], title: str = "") -> ScrubResult:
    """Scrub an already-split book. See scrub_book."""
    hits = _hits_by_paragraph(paragraphs)
    counts = Counter(h.term for h in hits)
    hit_idx = sorted({h.paragraph for h in hits})
    n = len(paragraphs)
    hit_paragraphs = [paragraphs[i] for i in hit_idx]

    reason = ""
    if title and contains_flight(title):
        reason = f"title matches flight lexicon: {title!r}"
    elif is_flight_book(len(hit_idx), n):
        reason = f"{len(hit_idx)} of {n} paragraphs hit the flight lexicon"
    if reason:
        return ScrubResult("", True, reason, list(paragraphs), hit_paragraphs, counts, n)

    drop = removal_mask(hit_idx, n)
    kept = [p for p, d in zip(paragraphs, drop) if not d]
    removed = [p for p, d in zip(paragraphs, drop) if d]
    return ScrubResult("\n\n".join(kept), False, "", removed, hit_paragraphs, counts, n)


def scrub_book(text: str, title: str = "") -> ScrubResult:
    """Remove flight passages from a book, or drop the book if it is about flight."""
    return scrub_paragraphs(split_paragraphs(text), title)
