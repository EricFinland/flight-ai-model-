"""Mechanical extraction of period question/answer pairs for chat fine-tuning.

Everything here is rule-based. No modern instruction data and no text written
by a modern language model goes into the pairs (see docs/leakage.md, section
6): the questions and answers are both lifted verbatim (apart from whitespace
and markup clean-up) from books in the *filtered main corpus*.

Sources, in order of preference:

``catechism``
    "Q. ... A. ...", "Question. ... Answer. ...", "_Q._--... _A._--..." and
    "Ques. ... Ans. ..." layouts: science and history catechisms, examination
    books, evidence taken before committees.
``correspondence``
    "Answers to correspondents" and "Notes and Queries": a query paragraph
    containing a question followed by the editor's bracketed ``[...]`` reply
    or an ``Answer.`` paragraph.
``drama``
    Plays and Socratic dialogues: a speech by X whose last sentence is a
    question, immediately followed by a speech by a different speaker Y.
    Speaker labels are detected per book by frequency (``MENO: ...``,
    ``Mrs. Alving. ...``, ``_Adr._ ...``, ``CAROLINE.`` on its own line).
``novel``
    Novel dialogue: a paragraph that is a quoted question (with at most a
    short speech tag such as "said he") followed by a paragraph that is a
    quoted reply. Strict, because novel dialogue is the noisiest source.

Every pair is re-checked against the flight lexicon (``prewright.flight``
when available, plus a stricter built-in pattern that also rejects any
mention of flying at all, even by birds) and against
``prewright.leakage.filter_document``. A pair with any hit is dropped.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

SOURCES = ("catechism", "correspondence", "drama", "novel")

# --------------------------------------------------------------------------
# Text clean-up
# --------------------------------------------------------------------------

_PARA_SPLIT = re.compile(r"\n[ \t]*\n")
_VERSE_NUM = re.compile(r"[ \t]{2,}\d{1,4}[ \t]*$", re.M)
_PAGE_NUM = re.compile(r"\{\d{1,4}\}")
_FOOTNOTE_REF = re.compile(r"\[(?:\d{1,3}|[a-z]|\*)\]")
_ITALIC = re.compile(r"(?<![A-Za-z0-9])_([^_\n]{1,200}?)_(?![A-Za-z0-9])")
_WS = re.compile(r"\s+")


def raw_paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping the line breaks inside each paragraph."""
    return [p for p in (q.strip("\n") for q in _PARA_SPLIT.split(text)) if p.strip()]


def unwrap(para: str) -> str:
    """Join hard-wrapped lines, drop verse line numbers and page markers."""
    para = _VERSE_NUM.sub("", para)
    para = _PAGE_NUM.sub("", para)
    para = _FOOTNOTE_REF.sub("", para)
    return _WS.sub(" ", para).strip()


def strip_italics(s: str) -> str:
    """``_word_`` -> ``word`` (Gutenberg's italic markup)."""
    return _ITALIC.sub(r"\1", s).replace("_", "")


def normalise_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))


def tidy(s: str) -> str:
    """Final clean-up applied to every question and answer."""
    s = strip_italics(s)
    s = s.replace("—", "--").replace(" -- ", "--")
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)
    s = _WS.sub(" ", s).strip()
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)
    s = s.strip(" -")
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


_ABBREV = "".join(rf"(?<!\b{re.escape(w)})" for w in ("Mr.", "Mrs.", "Dr.", "St.", "Messrs.", "viz.", "i.e.",
                                                     "e.g.", "No.", "vol.", "Mme.", "Mlle.", "Rev."))
_SENT_SPLIT = re.compile(r"(?<=[.!?])" + _ABBREV + r"([\"')\]]*)\s+(?=[\"'(\[]?[A-Z])")


def sentences(s: str) -> list[str]:
    parts = _SENT_SPLIT.split(s)  # [sentence, closing quotes, sentence, ...]
    out = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    return [x for x in out if x.strip()]


def truncate_sentences(s: str, max_chars: int) -> str:
    """Longest prefix of whole sentences within max_chars ('' if none)."""
    if len(s) <= max_chars:
        return s
    out = ""
    for sent in sentences(s):
        cand = f"{out} {sent}".strip()
        if len(cand) > max_chars:
            break
        out = cand
    return out


def last_sentences(s: str, max_chars: int) -> str:
    """Longest suffix of whole sentences within max_chars ('' if none)."""
    if len(s) <= max_chars:
        return s
    out = ""
    for sent in reversed(sentences(s)):
        cand = f"{sent} {out}".strip()
        if len(cand) > max_chars:
            break
        out = cand
    return out


# --------------------------------------------------------------------------
# Pairs and quality filters
# --------------------------------------------------------------------------

@dataclass
class Pair:
    source: str
    book_id: str
    question: str
    answer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Limits:
    q_min: int = 8
    q_max: int = 400
    a_min: int = 20
    a_max: int = 1200


LIMITS = {
    "catechism": Limits(q_max=400, a_max=1200),
    "correspondence": Limits(q_max=900, a_max=1200),
    "drama": Limits(q_min=15, q_max=300, a_max=800),
    "novel": Limits(q_min=15, q_max=300, a_max=600),
}

_IMPERATIVE = re.compile(
    r"^(?:give|name|describe|explain|state|define|mention|tell|enumerate|show|"
    r"distinguish|point out|repeat|say|relate|recite|prove|illustrate|compare)\b",
    re.I,
)
_BAD_SUBSTR = re.compile(
    r"gutenberg|e-?text|transcriber|footnote|illustration|sidenote|http|www\.|"
    r"\bpage \d|\bp\. \d|\bibid\b|\bop\. cit|copyright|\bchapter [ivxlc\d]+\b",
    re.I,
)
_ALLOWED_PUNCT = set(" .,;:!?'\"()-&")
_YES_NO = re.compile(r"^(?:yes|no|nay|ay|aye|certainly|surely|true|indeed|"
                     r"i do|i did|i am|i have|it is|very true|of course)[\s,.!;]*"
                     r"(?:sir|madam|socrates)?[\s,.!;]*$", re.I)
_DIALOGUE_STOP_ANSWER = re.compile(r"^(?:what\??|eh\??|well\??|ah!?|oh!?)$", re.I)


def _junk_ratio(s: str) -> float:
    bad = sum(1 for c in s if not (c.isalpha() or c.isdigit() or c in _ALLOWED_PUNCT))
    return bad / max(1, len(s))


def quality_problem(q: str, a: str, source: str) -> str | None:
    """Return the reason a pair is rejected, or None if it passes."""
    lim = LIMITS[source]
    if _YES_NO.match(a) or _DIALOGUE_STOP_ANSWER.match(a):
        return "a_bare_yes_no"
    if not (lim.q_min <= len(q) <= lim.q_max):
        return "q_length"
    if not (lim.a_min <= len(a) <= lim.a_max):
        return "a_length"
    if "?" not in q and not (source == "catechism" and _IMPERATIVE.match(q)):
        return "q_not_question"
    if source != "catechism" and a.rstrip("\"' )").endswith("?"):
        return "a_is_question"
    if a.rstrip("\"')]").strip()[-1:] not in (".", "!", "?") and not a.endswith("--"):
        return "a_unfinished"
    for s in (q, a):
        if _BAD_SUBSTR.search(s):
            return "boilerplate"
        if _junk_ratio(s) > 0.01:
            return "junk_chars"
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 0.6 * len(s):
            return "too_few_letters"
        if sum(c.isupper() for c in letters) > 0.3 * len(letters):
            return "too_many_capitals"
        if sum(c.isdigit() for c in s) > 0.1 * len(s):
            return "too_many_digits"
        if s.count('"') % 2 or s.count("(") != s.count(")"):
            return "unbalanced"
    if _norm(q) == _norm(a):
        return "q_equals_a"
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]+", "", s.lower())


def dedup_key(pair: Pair) -> str:
    """Pairs with the same answer are duplicates (editions, reprints, refrains)."""
    return hashlib.md5(_norm(pair.answer)[:300].encode()).hexdigest()


# --------------------------------------------------------------------------
# Safety: flight lexicon + leakage filter
# --------------------------------------------------------------------------

# Stricter than the corpus scrub on purpose: the chat data must not teach the
# model anything about flying, in either direction, because the evaluation
# asks it questions about flight. So even birds flying, wings and "the air"
# as a medium of travel are out.
_SFT_FLIGHT = re.compile(
    r"\b(?:fl(?:y|ies|ying|yer|yers|ier|iers|ew|own|ight|ights|ighted|edge|edged|utter\w*)|"
    r"wing(?:s|ed|less)?|soar\w*|aloft|airborne|hover\w*|glid(?:e|er|ers|ing)|"
    r"balloon\w*|aero\w*|aeronaut\w*|air-?ships?|dirigible\w*|zeppelin\w*|"
    r"parachut\w*|aviat\w*|montgolfier\w*|lilienthal|chanute|langley|"
    r"santos.dumont|kitty hawk|wright|icarus|daedalus|pegasus|kite\w*|"
    r"ornithopter\w*|helicopter\w*|levitat\w*)\b",
    re.I,
)


def _load_flight_checker() -> Callable[[str], bool] | None:
    """Find the corpus flight scrub's 'does this text mention flight' function."""
    try:
        from prewright import flight  # type: ignore
    except Exception:
        return None
    for name in ("contains_flight", "has_flight", "is_flight", "mentions_flight"):
        fn = getattr(flight, name, None)
        if callable(fn):
            return lambda s, fn=fn: bool(fn(s))
    for name in ("find_hits", "flight_hits", "hits", "scan", "find_flight"):
        fn = getattr(flight, name, None)
        if callable(fn):
            return lambda s, fn=fn: bool(fn(s))
    return None


_flight_fn: Callable[[str], bool] | None | bool = False  # False = not loaded yet


def flight_hit(text: str) -> bool:
    global _flight_fn
    if _SFT_FLIGHT.search(text):
        return True
    if _flight_fn is False:
        _flight_fn = _load_flight_checker()
    if _flight_fn:
        try:
            return _flight_fn(text)
        except Exception:
            return True  # fail closed
    return False


def leakage_hit(text: str) -> bool:
    from prewright.leakage import Decision, filter_document

    res = filter_document(text)
    return res.decision != Decision.KEEP or bool(res.hard_hits or res.soft_hits) or res.removed_paragraphs > 0


def safety_problem(q: str, a: str) -> str | None:
    text = f"{q}\n\n{a}"
    if flight_hit(text):
        return "flight"
    if leakage_hit(text):
        return "leakage"
    return None


# --------------------------------------------------------------------------
# Source 1+4a: catechisms ("Q. ... A. ...") and "Answer." replies
# --------------------------------------------------------------------------

_LEAD = r"(?:\A|(?<=\n\n)|(?<=[.?!;)\"'] ))[ \t]*(?:\d{1,4}\.[ \t]*)?"
_Q_WORD = r"(?:Q|Qu|Ques|Quest|Question|QUES|QUEST|QUESTION)"
_A_WORD = r"(?:A|Ans|Answ|Answer|ANS|ANSW|ANSWER)"
_MARK_TAIL = r"\s*(?:--|—|-)?\s*(?=[A-Z\"'(“‘_])"
_MARK = re.compile(
    _LEAD + r"_?(?:(?P<q>" + _Q_WORD + r")|(?P<a>" + _A_WORD + r"))(?:\s*[.:]_?|_?\s*[.:])" + _MARK_TAIL,
)
_SIGNATURE_PREFIX = re.compile(r"^_[^_]{1,60}_\.?\s*(?:--)?\s*")
_TITLE_PREFIX = re.compile(r"^[^.?!]{2,80}?\.?\s*--\s*")


def extract_catechism(paras: list[str]) -> list[tuple[str, str, str]]:
    """(source, question, answer) from Q./A. marker layouts.

    ``paras`` are unwrapped paragraphs. Questions may sit in their own
    paragraph or inline; an answer runs to the next marker or the end of its
    paragraph. An ``Answer.`` paragraph without a ``Q.`` before it is paired
    with the preceding paragraph if that asks a question (answers to
    correspondents).
    """
    text = "\n\n".join(paras)
    marks = [(m.start(), m.end(), "q" if m.group("q") else "a") for m in _MARK.finditer(text)]
    out: list[tuple[str, str, str]] = []
    for i, (s, e, kind) in enumerate(marks):
        if kind != "a":
            continue
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        answer = text[e:nxt].split("\n\n")[0]
        answer = re.sub(r"\s\d{1,4}\.?\s*$", "", answer).strip()
        prev = marks[i - 1] if i > 0 else None
        if prev and prev[2] == "q":
            question = text[prev[1]:s]
            if question.count("\n\n") > 1:
                continue
            out.append(("catechism", question.strip(), answer))
        else:
            # "Answer." paragraph replying to a correspondent's query. Only the
            # spelled-out marker counts: a bare "A." starting a paragraph is
            # usually an initial or a list item.
            if not re.match(r"\s*(?:\d{1,4}\.\s*)?_?an", text[s:e], re.I):
                continue
            before = text[:s]
            if not before.endswith("\n\n") and before:
                continue  # inline A. with no Q.: not a reply
            pstart = before.rstrip("\n").rfind("\n\n")
            question = before[pstart + 2 if pstart >= 0 else 0:].strip()
            if _MARK.match(question) or "?" not in question or question.startswith(("\"", "'")):
                continue
            question = _SIGNATURE_PREFIX.sub("", question)
            out.append(("correspondence", question, answer))
    return out


# --------------------------------------------------------------------------
# Source 4b: Notes and Queries style bracketed editorial replies
# --------------------------------------------------------------------------

_QUERY_BOOK = re.compile(r"queries with answers|answers to correspondents|notes and queries", re.I)


def extract_bracket_replies(paras: list[str]) -> list[tuple[str, str, str]]:
    """Editorial ``[...]`` replies under a correspondent's query.

    Callers should only use this on books that have such columns (see
    ``_QUERY_BOOK``); elsewhere bracketed paragraphs are mostly footnotes.
    """
    out = []
    for i, p in enumerate(paras):
        if not (p.startswith("[") and p.endswith("]") and len(p) >= 60):
            continue
        j = i - 1
        skipped = 0
        while j >= 0 and len(paras[j]) < 60 and "?" not in paras[j] and skipped < 2:
            j -= 1
            skipped += 1
        if j < 0 or paras[j].startswith(("[", "\"", "'")) or "?" not in paras[j] or len(paras[j]) < 40:
            continue
        q = _SIGNATURE_PREFIX.sub("", paras[j])
        q = _TITLE_PREFIX.sub("", strip_italics(q)) if "--" in q[:90] else q
        out.append(("correspondence", q, p[1:-1].strip()))
    return out


# --------------------------------------------------------------------------
# Source 2: plays and dialogues
# --------------------------------------------------------------------------

_LABEL_TOKEN = r"(?:[A-Z][A-Za-z'\-]*\.?|of|de|the|van|von|la|le|du|da)"
_LABEL_BODY = re.compile(r"^" + _LABEL_TOKEN + r"(?:\s+" + _LABEL_TOKEN + r"){0,3}$")
_LABEL_END = re.compile(r"[.:]_?\s+|_\s+")
_LABEL_STOP = {
    # Function words and interjections that start ordinary sentences.
    "a", "an", "and", "ah", "alas", "all", "as", "at", "aye", "ay", "but", "by", "come", "do",
    "dear", "eh", "enter", "even", "exit", "exeunt", "for", "go", "good", "ha", "he", "her",
    "here", "hm", "how", "hush", "i", "if", "in", "is", "it", "let", "look", "my", "nay", "no",
    "nor", "not", "now", "o", "of", "oh", "ok", "on", "or", "pooh", "she", "sir", "so", "stay",
    "stop", "that", "the", "then", "there", "they", "this", "thus", "to", "true", "very",
    "we", "well", "what", "when", "where", "who", "why", "yes", "yet", "you", "chapter",
    "book", "part", "act", "scene", "fig", "vol", "no", "note", "notes", "see", "mr", "mrs",
    "dr", "st", "ms", "messrs", "mme", "mlle", "mons", "q", "ans", "question", "answer",
    "preface", "introduction", "contents", "index", "finis", "the end", "appendix", "madam",
    "thank", "thanks", "indeed", "certainly", "never", "nothing", "perhaps", "quite",
    "really", "right", "silence", "sure", "wait", "hark", "lo", "farewell", "adieu",
    "hello", "hallo", "halloa", "nonsense", "pshaw", "bah", "fie", "amen", "listen",
    "soft", "peace", "away", "hence", "again", "still", "only", "both", "one", "two",
}
_STAGE = re.compile(r"\[[^\]]{0,300}\]|\((?=[^)]{0,300}\))[^)]*\)")


def _label_candidates(para: str) -> list[tuple[str, str]]:
    """Possible (label, rest-of-paragraph) splits for a raw paragraph."""
    cands = []
    first, _, rest_lines = para.partition("\n")
    fl = first.strip()
    # Label alone on the first line: "CAROLINE." / "MRS. B." / "HAMLET".
    lab = fl.strip("_").rstrip(".:").strip("_").strip()
    if 0 < len(lab) <= 30 and _LABEL_BODY.match(lab) and (fl.rstrip("_").endswith((".", ":")) or lab.isupper()):
        cands.append((lab, rest_lines))
    # Label leading the first line: "MENO: Can you..." / "Mrs. Alving. I am..."
    head = fl[:60]
    # Optional stage direction right after the label: "Mrs. Alving (walking...)."
    m = re.match(r"^_?([A-Z][^()\[\]]{0,30}?)_?\s*[(\[][^)\]]{0,200}[)\]]\s*[.:]?_?\s*", fl)
    if m and _LABEL_BODY.match(m.group(1).strip().rstrip(".")):
        cands.append((m.group(1).strip().rstrip("."), fl[m.end():] + ("\n" + rest_lines if rest_lines else "")))
    for m in _LABEL_END.finditer(head):
        lab = head[: m.start()].strip().strip("_").rstrip(".").strip()
        if lab and len(lab) <= 30 and _LABEL_BODY.match(lab):
            cands.append((lab, fl[m.end():] + ("\n" + rest_lines if rest_lines else "")))
    return cands


def _label_key(lab: str) -> str:
    return _WS.sub(" ", lab.upper().replace(".", ". ")).strip(" .")


def detect_speakers(paras: list[str], min_count: int = 3) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for p in paras:
        seen = {_label_key(lab) for lab, _ in _label_candidates(p)}
        counts.update(seen)
    return {k: c for k, c in counts.items() if c >= min_count and k.lower() not in _LABEL_STOP
            and not any(w.lower().strip(".") in {"chapter", "book", "part", "act", "scene"} for w in k.split())}


def _vocative_stripper(names: Iterable[str]) -> Callable[[str], str]:
    words = sorted({n.title() for n in names if len(n) >= 3 and n.replace(" ", "").replace(".", "").isalpha()},
                   key=len, reverse=True)
    if not words:
        return lambda s: s
    alt = "|".join(re.escape(w) for w in words)
    mid = re.compile(r",\s*(?:O |my dear |my good |dear |good )?(?:" + alt + r")(?=\s*[,.!?;:])", re.I)
    lead = re.compile(r"^(?:O |My dear |My good |Dear |Good )?(?:" + alt + r")\s*,\s*", re.I)

    def strip(s: str) -> str:
        s = lead.sub("", mid.sub("", s))
        return s[:1].upper() + s[1:] if s else s

    return strip


def extract_drama(raw_paras: list[str]) -> list[tuple[str, str, str]]:
    speakers = detect_speakers(raw_paras)
    if len(speakers) < 2:
        return []
    # Each paragraph: (speaker, speech) or None for anything else.
    units: list[tuple[str, str] | None] = []
    labelled = 0
    i = 0
    while i < len(raw_paras):
        p = raw_paras[i]
        best = None
        for lab, rest in _label_candidates(p):
            k = _label_key(lab)
            if k in speakers and (best is None or len(k) > len(best[0])):
                best = (k, rest)
        if best is None:
            units.append(None)
            i += 1
            continue
        labelled += 1
        speaker, speech = best
        speech = unwrap(speech)
        if not speech:
            # Label alone in its paragraph: the speech is the next paragraph(s).
            parts = []
            while i + 1 < len(raw_paras) and len(parts) < 3:
                nxt = raw_paras[i + 1]
                if any(_label_key(lab) in speakers for lab, _ in _label_candidates(nxt)):
                    break
                parts.append(unwrap(nxt))
                i += 1
            speech = " ".join(parts)
        speech = _WS.sub(" ", _STAGE.sub(" ", strip_italics(speech))).strip()
        units.append((speaker, speech) if speech else None)
        i += 1
    if labelled < 30 or labelled < 0.2 * len(raw_paras):
        return []
    strip_voc = _vocative_stripper(speakers)
    out = []
    for a, b in zip(units, units[1:]):
        if not a or not b or a[0] == b[0]:
            continue
        q, ans = a[1], b[1]
        if not q.rstrip("\"' ").endswith("?"):
            continue
        q = last_sentences(q, LIMITS["drama"].q_max)
        ans = truncate_sentences(ans, LIMITS["drama"].a_max)
        q, ans = strip_voc(q), strip_voc(ans)
        if q and ans:
            out.append(("drama", q, ans))
    return out


# --------------------------------------------------------------------------
# Source 3: novel dialogue
# --------------------------------------------------------------------------

_SPEECH_VERB = re.compile(
    r"\b(?:said|says|say|asked|asks|ask|cried|cries|replied|replies|answered|answers|inquired|"
    r"enquired|returned|exclaimed|continued|added|demanded|observed|remarked|rejoined|"
    r"murmured|whispered|began|resumed|repeated|retorted|responded|shouted|called|"
    r"pursued|went on|interposed|interrupted|suggested|urged|protested|cried out|"
    r"answered he|quoth)\b",
    re.I,
)
_SINGLE_Q = re.compile(r"(?:^|(?<=[\s(]))'(.+?[.,!?;:\-])'(?=[\s,.;:)\-]|$)")


def _split_quoted(p: str, single: bool) -> tuple[list[str], list[str]] | None:
    """(quoted segments, narration segments), or None if the paragraph is not dialogue."""
    if single:
        if not p.startswith("'"):
            return None
        quoted = [m.group(1) for m in _SINGLE_Q.finditer(p)]
        narration = _SINGLE_Q.sub("\x00", p).split("\x00")
        if not quoted or "'" in "".join(narration).replace("'s ", "").replace("n't", ""):
            return None
    else:
        if not p.startswith('"') or p.count('"') % 2:
            return None
        parts = p.split('"')
        quoted, narration = parts[1::2], parts[0::2]
    return quoted, [n for n in narration if n.strip()]


def _utterance(p: str, single: bool) -> str | None:
    sq = _split_quoted(p, single)
    if not sq:
        return None
    quoted, narration = sq
    words = sum(len(n.split()) for n in narration)
    if narration and (words > 12 or not _SPEECH_VERB.search(" ".join(narration))):
        return None
    segs = [q.strip() for q in quoted if q.strip()]
    for k, seg in enumerate(segs):
        # '"Well," said he, "I think..."' -> 'Well, I think...' but
        # '"It is so," said he. "Not before June."' -> 'It is so. Not before June.'
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if seg.endswith(",") and (nxt is None or (nxt[:1].isupper() and not nxt.startswith("I "))):
            segs[k] = seg[:-1] + "."
    return _WS.sub(" ", " ".join(segs)).strip() or None


def extract_novel(paras: list[str]) -> list[tuple[str, str, str]]:
    starts_d = sum(p.startswith('"') for p in paras)
    starts_s = sum(p.startswith("'") for p in paras)
    single = starts_s > starts_d
    if max(starts_d, starts_s) < 20:
        return []
    out = []
    for a, b in zip(paras, paras[1:]):
        if not a.endswith(('"', "'")) and "?" not in a[-40:]:
            continue
        if len(a) > 600 or len(b) > 1500:
            continue
        q = _utterance(a, single)
        if not q or not q.endswith("?"):
            continue
        ans = _utterance(b, single)
        if not ans:
            continue
        q = last_sentences(q, LIMITS["novel"].q_max)
        ans = truncate_sentences(ans, LIMITS["novel"].a_max)
        if q and ans:
            out.append(("novel", q, ans))
    return out


# --------------------------------------------------------------------------
# Per-book driver and dataset assembly
# --------------------------------------------------------------------------

PER_BOOK_CAP = {"catechism": 200, "correspondence": 200, "drama": 60, "novel": 25}


def extract_book(book_id: str, text: str, stats: Counter | None = None,
                 rng: random.Random | None = None) -> list[Pair]:
    """All pairs from one book that pass the quality and safety filters."""
    stats = stats if stats is not None else Counter()
    rng = rng or random.Random(hash(book_id) & 0xFFFFFFFF)
    raw = raw_paragraphs(normalise_quotes(text))
    paras = [unwrap(p) for p in raw]
    found: list[tuple[str, str, str]] = []
    if _MARK.search(text):
        found += extract_catechism(paras)
    if "[" in text and _QUERY_BOOK.search(text):
        found += extract_bracket_replies(paras)
    drama = extract_drama(raw)
    found += drama
    if not drama:
        found += extract_novel(paras)

    by_source: dict[str, list[Pair]] = defaultdict(list)
    for source, q, a in found:
        stats[f"raw/{source}"] += 1
        q, a = tidy(q), tidy(a)
        why = quality_problem(q, a, source) or safety_problem(q, a)
        if why:
            stats[f"drop/{why}"] += 1
            continue
        by_source[source].append(Pair(source, str(book_id), q, a))
    out: list[Pair] = []
    for source, pairs in by_source.items():
        cap = PER_BOOK_CAP[source]
        if len(pairs) > cap:
            stats["drop/book_cap"] += len(pairs) - cap
            pairs = rng.sample(pairs, cap)
        out += pairs
    return out


def balance(pairs: list[Pair], max_pairs: int, max_share: dict[str, float],
            rng: random.Random) -> list[Pair]:
    """Keep every catechism/correspondence pair (up to max_pairs) and cap each
    dialogue source at ``max_share[source]`` of the final set."""
    by: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by[p.source].append(p)
    for v in by.values():
        rng.shuffle(v)
    preferred = [p for s in ("catechism", "correspondence") for p in by.get(s, [])][:max_pairs]
    pools = {s: by.get(s, []) for s in ("drama", "novel")}

    def size(total: int) -> int:
        return min(max_pairs, len(preferred) + sum(
            min(len(v), int(max_share.get(s, 1.0) * total)) for s, v in pools.items()))

    # Largest total T with T <= size(T): iterate downwards from max_pairs.
    total = max_pairs
    for _ in range(100):
        nxt = size(total)
        if nxt >= total:
            break
        total = nxt
    chosen = list(preferred)
    caps = {s: min(len(v), int(max_share.get(s, 1.0) * total)) for s, v in pools.items()}
    room = max(0, total - len(preferred))
    if sum(caps.values()) > room:  # max_pairs binds: share the room in proportion to the caps
        scale = room / sum(caps.values())
        caps = {s: int(c * scale) for s, c in caps.items()}
    for s, v in pools.items():
        chosen += v[: caps[s]]
    rng.shuffle(chosen)
    return chosen


def split_val(pairs: list[Pair], val_frac: float) -> tuple[list[Pair], list[Pair]]:
    """Book-level split: a book's pairs are all train or all val."""
    def is_val(book_id: str) -> bool:
        h = int(hashlib.md5(f"sft-val:{book_id}".encode()).hexdigest()[:8], 16)
        return h < val_frac * 0xFFFFFFFF

    train, val = [], []
    for p in pairs:
        (val if is_val(p.book_id) else train).append(p)
    return train, val


def load_seed(path) -> list[dict]:
    """Hand-written seed conversations: each line has either question/answer or messages."""
    import json

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
