"""Detect and remove text that leaks knowledge from after the cutoff.

Old documents reach us wrapped in modern material: digitization boilerplate,
editors' introductions, translators' notes, reprint copyright pages. Catalogue
dates are also sometimes simply wrong. This module works in three passes:

1. Strip digitization boilerplate (Project Gutenberg headers, scanner notices,
   URLs). It is local and harmless to cut, so we remove the paragraph rather
   than discard the document.
2. Look for *hard* anachronisms: facts or coinages that cannot appear in
   genuine pre-1904 text. Any hit rejects the whole document, because an
   editor's note naming Einstein usually means more modern apparatus nearby.
3. Score *soft* signals (words with rare pre-1904 senses, mentions of later
   years). A high score sends the document to human review, not training.

Rules aim for precision on hard hits; recall comes from layering this with
metadata checks and the evaluation probes in evals/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    KEEP = "keep"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    weight: float = 0.0  # soft rules only
    note: str = ""


def _rx(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Years after the cutoff year, as they would appear in running text.
_POST_YEAR = r"(?:190[4-9]|19[1-9]\d|20\d\d)"

# Paragraphs matching these are digitization artefacts and are cut out.
BOILERPLATE_RULES: tuple[Rule, ...] = (
    Rule("gutenberg", _rx(r"project\s+gutenberg|gutenberg\.org|gutenberg-tm")),
    Rule("ebook", _rx(r"\be-?books?\b|\be-?texts?\b|\betext\d*\b")),
    Rule("url", _rx(r"https?://|\bwww\.\w|\.(?:com|org|net|edu)\b/?")),
    Rule("email", _rx(r"\b[\w.+-]+@[\w-]+\.(?:com|org|net|edu)\b")),
    Rule("scanner", _rx(r"digiti[sz]ed by|internet archive|google books|hathitrust|\bOCR\b")),
    Rule("transcriber", _rx(r"transcriber['’]?s?\s+note|produced by .{0,80}(?:proofread|distributed)")),
    Rule("isbn", _rx(r"\bISBN\b", 0)),
    Rule("modern_copyright", _rx(rf"(?:copyright|©|\(c\)|reprinted|this edition)[^\n]{{0,60}}?\b{_POST_YEAR}\b")),
)

# Any match rejects the document.
HARD_RULES: tuple[Rule, ...] = (
    Rule("relativity_theory", _rx(r"\b(?:special|general)\s+(?:theory\s+of\s+)?relativity|theory\s+of\s+relativity|relativity\s+theory"),
         note="Einstein 1905/1915; 'relativity' alone is an older philosophical term"),
    Rule("quantum_physics", _rx(r"quantum\s+(?:mechanics|physics|theory|numbers?)"),
         note="Planck's 1900 hypothesis is fine; the theory's names are later"),
    Rule("subatomic_coinages", _rx(r"\b(?:protons?(?!\s+pseudos)|neutrons?|isotopes?|photons?|neutrinos?)\b"),
         note="coined 1913-1930; 'electron' (1891) is a period term; 'proton pseudos' is old logic jargon"),
    Rule("nuclear", _rx(r"\bnuclear\s+(?:physics|energy|power|weapons?|reactors?|fission|fusion|bomb)|\batom(?:ic)?\s+bomb|\bA-bomb\b"),
         note="'nuclear' alone is a period biology term"),
    Rule("world_wars", _rx(r"\b(?:first|second)\s+world\s+war|world\s+war\s+(?:I{1,2}|one|two|1|2)\b|great\s+war\s+of\s+1914"),
         note="'the Great War' alone was used of earlier wars"),
    Rule("wright_flight", _rx(r"wright\s+flyer|first\s+(?:powered|aeroplane|airplane)\s+flight|wrights?['’]?\s+(?:first\s+)?(?:powered\s+)?flights?\s+(?:at|near)\s+kitty\s+hawk"),
         note="the Wrights' 1900-02 gliding was published and is allowed"),
    Rule("russo_japanese_war", _rx(r"russo-japanese\s+war|battle\s+of\s+tsushima|treaty\s+of\s+portsmouth"),
         note="February 1904 onward"),
    Rule("titanic", _rx(r"(?:R\.?M\.?S\.?|steamship|liner|sinking\s+of\s+the)\s+titanic|titanic\s+disaster"),
         note="'titanic' alone is an ordinary adjective"),
    Rule("twentieth_century_politics", _rx(r"\b(?:hitler|nazis?|soviet\s+union|U\.?S\.?S\.?R\.?|league\s+of\s+nations|lenin(?:ism|ist)|stalin(?:ism|ist)?)\b"),
         note="'Lenin' as a pen name dates from 1901, so only the -isms"),
    Rule("medicine", _rx(r"\b(?:vitamins?|penicillin|insulin|antibiotics?)\b"),
         note="coined 1909-1940s"),
    Rule("technology", _rx(r"\b(?:radar|laser|transistors?|internet|television\s+set|radio\s+broadcast(?:ing)?|model\s+T\s+ford|jet\s+(?:engine|plane|aircraft))\b"),
         note="'television' itself was coined in 1900, so only later compounds"),
    Rule("music", _rx(r"\bjazz\b"), note="first attested c. 1912"),
)

# Each distinct match adds `weight`; past the review threshold the document is held back.
SOFT_RULES: tuple[Rule, ...] = (
    Rule("airplane", _rx(r"\bair-?planes?\b"), 2.0,
         note="American coinage after 1903; 'aeroplane' is older and allowed"),
    Rule("einstein", _rx(r"\beinstein\b"), 2.0,
         note="Einstein published on capillarity in 1901, so not a hard rule"),
    Rule("kitty_hawk", _rx(r"\bkitty\s+hawk\b"), 1.0,
         note="Wright gliding camps 1900-02 are allowed; 1903 flight is not"),
    Rule("world_war", _rx(r"\bworld\s+war\b"), 1.5),
    Rule("bolshevik", _rx(r"\bbolshevi(?:k|ks|sm|st)\b"), 1.5,
         note="the 1903 party split was named late in the year"),
    Rule("atomic_energy", _rx(r"\batomic\s+energy\b"), 1.0,
         note="Soddy and others speculated about radium energy c. 1903"),
    Rule("post_cutoff_year", _rx(rf"\b{_POST_YEAR}\b", 0), 0.5,
         note="noisy: bond maturities and fiction set in the future mention later years"),
)

DEFAULT_REVIEW_THRESHOLD = 3.0


@dataclass(frozen=True)
class Hit:
    rule: str
    text: str


@dataclass
class FilterResult:
    decision: Decision
    text: str  # with boilerplate removed
    hard_hits: list[Hit] = field(default_factory=list)
    soft_hits: list[Hit] = field(default_factory=list)
    soft_score: float = 0.0
    removed_paragraphs: int = 0

    def reasons(self) -> list[str]:
        return [f"{h.rule}: {h.text!r}" for h in self.hard_hits + self.soft_hits]


_GUTENBERG_START = _rx(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_GUTENBERG_END = _rx(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def strip_gutenberg(text: str) -> str:
    """Keep only the text between Project Gutenberg's START and END markers."""
    if start := _GUTENBERG_START.search(text):
        text = text[start.end():]
    if end := _GUTENBERG_END.search(text):
        text = text[: end.start()]
    return text.strip()


def strip_boilerplate(text: str) -> tuple[str, int]:
    """Remove paragraphs containing digitization artefacts. Returns (text, removed)."""
    kept, removed = [], 0
    for para in _PARAGRAPH_SPLIT.split(strip_gutenberg(text)):
        if any(rule.pattern.search(para) for rule in BOILERPLATE_RULES):
            removed += 1
        else:
            kept.append(para)
    return "\n\n".join(kept).strip(), removed


def filter_document(text: str, review_threshold: float = DEFAULT_REVIEW_THRESHOLD) -> FilterResult:
    cleaned, removed = strip_boilerplate(text)

    hard_hits = [
        Hit(rule.name, m.group(0))
        for rule in HARD_RULES
        if (m := rule.pattern.search(cleaned))
    ]

    soft_hits: list[Hit] = []
    score = 0.0
    for rule in SOFT_RULES:
        distinct = {m.group(0).lower() for m in rule.pattern.finditer(cleaned)}
        for match in sorted(distinct):
            soft_hits.append(Hit(rule.name, match))
            score += rule.weight

    if hard_hits:
        decision = Decision.REJECT
    elif score >= review_threshold:
        decision = Decision.REVIEW
    else:
        decision = Decision.KEEP
    return FilterResult(decision, cleaned, hard_hits, soft_hits, score, removed)
