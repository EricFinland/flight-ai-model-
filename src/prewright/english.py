"""A cheap test of whether a book is written in English.

English prose is about half function words: in a sample of running text, the
share of word tokens found in a list of ~150 common English stopwords is
typically 0.45-0.60. French, German, Latin and the rest score under 0.10, and
their own stopwords score high instead. Word lists, indexes, catalogues and
sheet music also score low; they are not useful prose, so they fail too.

The sample is a few windows spread through the book, so a long English preface
to a foreign text (or the reverse) does not decide the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ENGLISH_STOPWORDS = frozenset("""
the of and to a in that is was he for it with as his on be at by i this had not are but from or have an they
which one you were her all she there would their we him been has when who will more no if out so said what up
its about into than them can only other some could time these two may then do first any my now such like our
over man me even most made after also did many before must through back where much your way well down should
because each just those how too little very make still own see men work long here both between life being
under never day same another know while last might us great old off come since against go came right take
three upon shall thee thou thy hath unto nor yet himself herself myself whom whose these those ye
""".split())

# Stopwords of the languages most often found in old book collections. A word
# that is also English ("a", "in", "is") is left out of these lists.
OTHER_STOPWORDS: dict[str, frozenset[str]] = {
    "french": frozenset("le la les de des du et est une que qui dans pour pas au sur ne se ce il elle sont avec par plus mais nous vous je".split()),
    "german": frozenset("der die das und ist nicht ein eine zu den von mit sich des auf dem ich er sie es auch als wie wir aus noch".split()),
    "latin": frozenset("et est non ad cum quod qui quae sed ut esse enim per sunt aut atque ab ex nec etiam quam".split()),
    "spanish": frozenset("el los las y en que del se por con para una es lo su al como pero sus".split()),
    "italian": frozenset("il di che e la per un non una sono del della le con si da nel alla ma come".split()),
    "dutch": frozenset("de het een en van ik te dat die niet zijn op aan met als voor".split()),
    "portuguese": frozenset("o os as e do da dos das em que um uma para com não se por mais ao".split()),
}

MIN_ENGLISH_RATIO = 0.25  # English prose rarely falls below 0.35
MIN_MARGIN = 1.5  # English share must beat every other language's share by this factor
MIN_WORDS = 50

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_WINDOW = 20_000
_WINDOW_STARTS = (0.10, 0.50, 0.85)


@dataclass
class LanguageScore:
    is_english: bool
    english_ratio: float
    words: int
    other: dict[str, float] = field(default_factory=dict)

    def reason(self) -> str:
        lang, ratio = max(self.other.items(), key=lambda kv: kv[1]) if self.other else ("none", 0.0)
        return f"english stopword ratio {self.english_ratio:.3f} over {self.words} words; best other {lang} {ratio:.3f}"


def sample(text: str, window: int = _WINDOW) -> str:
    """Up to three windows of `window` characters spread through the text."""
    if len(text) <= window * len(_WINDOW_STARTS):
        return text
    return "\n".join(text[int(len(text) * s): int(len(text) * s) + window] for s in _WINDOW_STARTS)


def score_english(text: str) -> LanguageScore:
    words = _WORD.findall(sample(text).lower())
    n = len(words)
    if n < MIN_WORDS:
        return LanguageScore(False, 0.0, n)
    english = sum(w in ENGLISH_STOPWORDS for w in words) / n
    other = {lang: sum(w in sw for w in words) / n for lang, sw in OTHER_STOPWORDS.items()}
    return LanguageScore(decide(english, other), english, n, other)


def decide(english_ratio: float, other: dict[str, float]) -> bool:
    """The English verdict from stopword shares (kept separate so thresholds can be re-tuned on stored scores)."""
    return english_ratio >= MIN_ENGLISH_RATIO and english_ratio >= MIN_MARGIN * max(other.values(), default=0.0)


def is_english(text: str) -> bool:
    return score_english(text).is_english
