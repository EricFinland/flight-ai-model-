"""Automatic concept matcher for the discovery experiment (docs/discovery-experiment.md).

The flight lexicon (``prewright.flight``) finds the *words* of human flight:
balloon, aeronaut, flying machine. A model that has never read them can only
arrive at the *idea* in other words: "a man might be carried up into the air",
"a great bag of silk filled with heated air", "a screw turning against the
air". This module looks for such ideas and for the principles a workable
proposal needs.

Two groups of concepts:

* ``idea``: a person, or a made vehicle, leaving the ground or travelling
  aloft (``human_aloft``, ``carry_human``, ``leave_ground``,
  ``vehicle_aloft``, ``wings_for_human``).
* ``principle``: the four pre-registered principles (``fixed_wing``: a
  surface held at an angle to the air; ``propulsion``: a screw, fan or vane
  driven against the air, separate from the lifting surface;
  ``flight_control``: rudder, tail, wing warping, weight shifting, balance in
  the air; ``lighter_than_air``: heated air or a light gas in an envelope that
  rises), plus ``powered_flapping`` (an engine working wings), recorded but
  not one of the four.

Every pattern works inside one sentence (the word gaps cannot cross ``.``,
``!``, ``?`` or ``;``). A concept may also *require* a context word in the same
sentence, or be *vetoed* by one (angels, witches and Icarus fly by magic or
myth, which the scrub keeps on purpose). The matcher is deliberately
conservative: it is a reproducible screen, not a judge. A model can state an
idea in words nobody anticipated; the blinded rating (see
``discovery.export_blinded``) is there for that, and the summary reports how
well the two agree.

Matching runs on ``prewright.flight.normalize``-d text, the same folding the
scrub uses. ``score_text(text, prefix=...)`` scores a completion together with
the end of its prompt, so a sentence the model finishes counts, but a match
must reach into the model's own text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from prewright.flight import find_flight_hits, normalize

# --- building blocks -------------------------------------------------------

# Separator between two words of one sentence: anything but a word character or sentence punctuation.
SEP = r"[^\w.!?;]+"


def _words(*ws: str) -> str:
    return "(?:" + "|".join(ws) + ")"


def gap(k: int, exclude: Sequence[str] = ()) -> str:
    """Up to ``k`` intervening words (none of them in ``exclude``), then a separator."""
    word = rf"(?!{_words(*exclude)}\b)\w+" if exclude else r"\w+"
    return rf"(?:{SEP}{word}){{0,{k}}}?{SEP}"


NEG = ("not", "never", "no", "nor", "neither", "cannot", "unable", "incapable", "impossible", "nothing", "none")
# Things people throw, raise or wave into the air, and things that fly or float by themselves.
OBJECTS = ("kites?", "stones?", "balls?", "hats?", "caps?", "eyes?", "hands?", "arms?", "heads?", "voices?", "glass",
           "glasses", "cups?", "smoke", "dust", "sparks?", "flags?", "colou?rs", "banners?", "pennants?", "spray",
           "steam", "birds?", "pigeons?", "doves?", "hawks?", "falcons?", "larks?", "arrows?", "spears?", "shots?",
           "guns?", "rockets?", "fireworks?", "handkerchiefs?", "feathers?", "leaves", "snow", "rain", "fumes",
           "vapou?rs?", "clouds?", "mist", "foam", "sword", "swords", "whip", "fist", "fists", "finger", "fingers")

HUMAN = (r"\b(?:man|men|mankind|humanity|a\s+person|persons|people|passengers?|travell?ers?|voyagers?|"
         r"human\s+beings?|the\s+human\s+(?:race|body|frame)|we|us|ourselves|himself|themselves|myself|he|him|i|"
         r"woman|women|boy|boys|child|children|"
         r"an?\s+(?:engineer|inventor|operator|navigator|explorer|soldier|observer|rider|driver|traveller|traveler)|"
         r"the\s+(?:engineer|inventor|operator|navigator|explorer|soldier|observer|rider|driver|traveller|traveler))\b")
HUMAN_OBJ = (r"\b(?:a\s+man|men|a\s+person|persons|people|passengers?|travell?ers?|a\s+rider|riders|a\s+boy|a\s+child|"
             r"a\s+human\s+being|human\s+beings|(?:the|its|his)\s+(?:weight|body)\s+of\s+(?:a\s+)?(?:man|men|person)|"
             r"him|us|themselves|himself|ourselves|one\s+or\s+(?:two|more)\s+(?:men|persons)|"
             r"the\s+(?:engineer|inventor|operator|navigator|observer|rider|driver)|"
             r"an?\s+(?:car|basket|seat|cabin|cage|carriage)\s+(?:with|for|containing|holding|bearing)\s+"
             r"(?:a\s+man|men|passengers?|travell?ers?|persons|people))\b")
AIR = r"(?:the\s+)?(?:upper\s+|open\s+|thin\s+|free\s+)?(?:air|sky|skies|clouds|heavens|atmosphere|ether|firmament|empyrean)(?![\w-])"
PREP = r"(?:into|through|in|above|over|across|among|amid|amidst|up\s+to|upon|on|to|towards?|beyond)"
ALOFT = (r"(?:aloft|skywards?|heavenwards?|on\s+high|in\s+mid-?air|"
         r"above\s+the\s+(?:earth|ground|trees|tree-?tops|house-?tops|roofs|hills|mountains|clouds|sea))\b")
BIRDS = (r"(?:birds?|eagles?|hawks?|swallows?|larks?|gulls?|sea-?gulls?|storks?|pigeons?|doves?|albatross(?:es)?|"
         r"vultures?|condors?|kites?|bats?|cranes?|geese|wild\s+geese|frigate-?birds?)")
LIKE_BIRD = rf"(?:like|as)\s+(?:a\s+|an\s+|the\s+)?(?:great\s+|large\s+|wild\s+)?{BIRDS}\b"
VERB_ALOFT = (r"(?:rise|rises|rising|rose|risen|ascend\w*|mount|mounts|mounted|mounting|soar\w*|float\w*|fly|flies|"
              r"flew|flown|flying|glid\w*|hover\w*|sail|sails|sailed|sailing|travel\w*|journey\w*|voyag\w*|swim|swims|"
              r"swam|swimming|drift\w*|pass|passes|passed|passing|mov(?:e|es|ed|ing)|go|goes|went|gone|going|wing\w*|"
              r"(?:be|been|being|is|was|were|are|am)\s+(?:lifted|raised|carried|borne|conveyed|transported|wafted|"
              r"hoisted|supported|sustained|buoyed|propelled|driven|kept\s+up|held\s+up|drawn\s+up))")
VERB_LIKE_BIRD = r"(?:fly|flies|flew|flown|flying|soar\w*|glid\w*|float\w*|hover\w*|rise|rises|rising|rose|mount\w*|wing\w*)"
MODAL = r"(?:might|may|could|would|shall|will|can|should|must|to|ought\s+to)"
VEHICLE = (r"\b(?:machines?|vessels?|ships?|cars?|boats?|carriages?|chariots?|apparatus|contrivances?|engines?|crafts?|"
           r"structures?|frames?|globes?|bags?|rafts?|platforms?|baskets?|cages?|cabins?)\b")
CARRY = (r"\b(?:carry|carries|carried|carrying|bear|bears|bore|borne|bearing|lift|lifts|lifted|lifting|raise|raises|"
         r"raised|raising|convey\w*|transport\w*|support\w*|sustain\w*|hold\w*\s+up|held\s+up|buoy\w*|hoist\w*|"
         r"waft\w*|draw\w*\s+up|drew\s+up|take|takes|took|taking)\b")
LIFT = (r"\b(?:ris(?:e|es|en|ing)|rose|ascend\w*|float\w*|lift\w*|buoy\w*|bear\w*\s+up|bore\s+up|borne\s+up|"
        r"rais(?:e|es|ed|ing)|carr(?:y|ies|ied|ying)|aloft|upwards?|mount\w*|soar\w*|support\w*|sustain\w*)\b")
AIR_CONTEXT = (r"\b(?:air|aloft|sky|skies|clouds|heavens|kites?|wings?|winged|globes?|bags?|gas|hydrogen|fly|flies|"
               r"flying|flew|soar\w*|float\w*|glid\w*)\b")
MYTH = (r"\b(?:angels?|archangels?|cherub\w*|seraph\w*|witch\w*|wizards?|fair(?:y|ies)|genii|genie|spirits?|souls?|"
        r"ghosts?|phantoms?|demons?|devils?|satan|gods?|goddess\w*|icarus|daedalus|pegasus|cupid|"
        r"magic\w*|enchant\w*|sorcer\w*|broomsticks?|carpets?)\b")


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# --- concepts --------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One way of expressing a concept.

    ``pattern`` must match inside one sentence. ``requires`` (optional) must
    also match somewhere in that sentence; ``veto`` (optional) must not.
    ``all_of`` makes the rule a co-occurrence rule: every regex in it must
    match the same sentence (``pattern`` is then unused).
    """

    pattern: re.Pattern[str] | None
    requires: re.Pattern[str] | None = None
    veto: re.Pattern[str] | None = None
    all_of: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class Concept:
    name: str
    group: str  # "idea" | "principle" | "other"
    description: str
    rules: tuple[Rule, ...] = field(default_factory=tuple)


def R(pattern: str | None = None, requires: str | None = None, veto: str | None = None,
      all_of: Sequence[str] = ()) -> Rule:
    return Rule(_rx(pattern) if pattern else None, _rx(requires) if requires else None,
                _rx(veto) if veto else None, tuple(_rx(p) for p in all_of))


_NO_OBJ = NEG + OBJECTS
WINDMILL = r"\b(?:wind-?mills?|mills?|weather-?cocks?|vanes?\s+of\s+the\s+(?:wind-?)?mill)\b"
# Ways of holding something up in the air that are not propulsion ("he held the fan in the air").
HELD = (r"rais\w*", "held", r"hold\w*", r"wav\w*", r"toss\w*", r"lift\w*", r"flourish\w*", "thrown")
DEATH = r"\b(?:die|dies|died|death|dead|grave|buried|burial|heaven)\b"
HEIGHT_CONTEXT = (r"\b(?:air|aloft|wings?|birds?|soar\w*|fly|flying|flew|flight|heights?|clouds|"
                  r"feet\s+(?:high|above)|yards\s+(?:high|above))\b")

CONCEPTS: tuple[Concept, ...] = (
    # ---- the idea: a person or a made thing leaving the ground ----
    Concept("human_aloft", "idea", "a person rises, travels or is carried in the air, or flies like a bird", (
        R(rf"{HUMAN}{gap(4, _NO_OBJ)}{VERB_ALOFT}{gap(3, OBJECTS)}{PREP}\s+{AIR}", veto=MYTH),
        R(rf"{HUMAN}{gap(4, _NO_OBJ)}{VERB_ALOFT}{gap(2, OBJECTS)}{ALOFT}", veto=MYTH),
        R(rf"{HUMAN}{gap(4, _NO_OBJ)}{VERB_LIKE_BIRD}{gap(1)}{LIKE_BIRD}", veto=MYTH),
    )),
    Concept("carry_human", "idea", "something lifts or carries a person into the air", (
        R(rf"{CARRY}{gap(3, NEG)}(?:up\s+)?{HUMAN_OBJ}{gap(5, OBJECTS)}(?:(?:{PREP}\s+{AIR})|{ALOFT}|"
          r"(?:off|from)\s+the\s+(?:ground|earth))", veto=MYTH),
        R(rf"{CARRY}\s+(?:up\s+)?{HUMAN_OBJ}", requires=AIR_CONTEXT, veto=MYTH),
        R(rf"\b(?:enough|sufficient|power|force|strength|able|capable){gap(2)}(?:to|of){gap(1)}{CARRY}"
          rf"{gap(3)}{HUMAN_OBJ}", requires=AIR_CONTEXT, veto=MYTH),
    )),
    Concept("leave_ground", "idea", "a person might leave the ground (rise from the earth)", (
        R(rf"{HUMAN}{gap(3, NEG)}{MODAL}{gap(2, NEG)}(?:rise|raise\s+(?:himself|themselves|ourselves)|"
          r"lift\s+(?:himself|themselves|ourselves)|leave|quit|forsake|depart\s+from|get\s+(?:clear\s+)?(?:off|of))"
          r"(?:\s+\w+)?\s+(?:clear\s+)?(?:of\s+|off\s+|from\s+)?the\s+(?:ground|earth|solid\s+earth|"
          r"surface\s+of\s+the\s+earth)\b", requires=HEIGHT_CONTEXT, veto=MYTH + "|" + DEATH),
    )),
    Concept("vehicle_aloft", "idea", "a made thing (machine, vessel, car, globe, frame) rises or travels in the air", (
        R(rf"{VEHICLE}{gap(4, _NO_OBJ)}(?:{VERB_ALOFT}|be\s+(?:propelled|driven|steered|navigated|guided))"
          rf"{gap(3, OBJECTS)}(?:{PREP}\s+{AIR}|{ALOFT})", veto=MYTH),
        R(rf"{VEHICLE}{gap(4, _NO_OBJ)}{VERB_LIKE_BIRD}{gap(1)}{LIKE_BIRD}", veto=MYTH),
    )),
    Concept("wings_for_human", "idea", "wings made for, or fastened to, a person", (
        R(rf"\bwings?{gap(3)}(?:for|fastened\s+to|fixed\s+to|attached\s+to|strapped\s+to|bound\s+to|tied\s+to|"
          r"fitted\s+to|upon|on)(?:\s+\w+){0,2}?\s+(?:a\s+man|men|man|his\s+(?:arms|shoulders|back|body)|"
          r"the\s+(?:arms|shoulders|back)\s+of|human|himself|ourselves|us|the\s+human)\b", veto=MYTH),
        R(rf"{HUMAN}{gap(3, NEG)}(?:furnish\w*|provid\w*|equip\w*|fit|fits|fitted|fitting|make|makes|made|"
          r"construct\w*|build\w*|built|contriv\w*|fashion\w*|put\s+on|wear\w*|wore|strap\w*|fasten\w*)\s+"
          r"(?:himself\s+|themselves\s+|ourselves\s+|him\s+|them\s+|us\s+)?(?:with\s+)?(?:a\s+)?(?:pair\s+of\s+|set\s+of\s+)?"
          r"(?:large\s+|great\s+|broad\s+|light\s+|strong\s+)?wings\b", veto=MYTH),
        R(rf"\b(?:a\s+man|men|man|mankind){gap(1, NEG)}with{gap(1)}wings\b", veto=MYTH),
    )),
    # ---- the four pre-registered principles ----
    Concept("fixed_wing", "principle", "a surface held fixed and inclined to the air bears a weight (lift without flapping)", (
        R(r"\b(?:fixed|rigid|motionless|immovable|unmoving|stationary|outspread|outstretched|extended|steady)\s+"
          r"(?:and\s+\w+\s+)?(?:wings?|planes?|surfaces?)\b", requires=r"\b(?:air|wind|breeze|glid\w*|soar\w*|bear|support\w*|sustain\w*)\b"),
        R(rf"\b(?:wings?|planes?|surfaces?){gap(1)}(?:held|kept|fixed)\s+(?:quite\s+)?(?:fixed|rigid|motionless|still|"
          r"immovable|steady|outspread|extended|at\s+an\s+angle|obliquely|aslant|inclined)\b"),
        R(r"\bwithout\s+(?:a\s+|one\s+|any\s+)?(?:flapping|beating|moving|beat|stroke|flap|motion)\s+(?:of\s+)?"
          r"(?:its\s+|their\s+|the\s+|his\s+)?wings\b"),
        R(r"\b(?:inclined|slanting|sloping|oblique|tilted|canted|slightly\s+inclined)\s+(?:planes?|surfaces?|wings?|"
          r"sails?|boards?|sheets?|faces?|blades?|vanes?|plates?)\b", requires=r"\b(?:air|wind|breeze|current)\b"),
        R(r"\bangle\s+of\s+(?:incidence|inclination|attack)\b", requires=r"\b(?:air|wind|breeze)\b"),
        R(r"\b(?:meet\w*|strik\w*|struck|receiv\w*|cut|cuts|cutting|tak(?:e|es|ing)|press\w*\s+(?:upon|against))\s+"
          r"(?:the\s+)?(?:air|wind)\s+(?:obliquely|aslant|edgeways|edgewise|at\s+an?\s+(?:small\s+|slight\s+|acute\s+|"
          r"low\s+|gentle\s+)?angle)\b"),
        R(r"\b(?:air|wind)(?:\s+\w+){0,2}?\s+(?:press\w*|act\w*|bear\w*|push\w*|strik\w*|blow\w*)(?:\s+\w+){0,2}?\s+"
          r"(?:upon|on|against|beneath|under|underneath)\s+(?:the\s+|its\s+|their\s+|a\s+|an\s+)?"
          r"(?:under[- ]?(?:side|surface)s?|lower\s+(?:side|surface)s?|inclined|slanting|oblique|wings?|planes?)\b",
          requires=LIFT),
    )),
    Concept("propulsion", "principle", "a screw, fan or vane driven against the air (thrust separate from lift)", (
        R(rf"\b(?:screws?|propell[eo]rs?|fans?|vanes?|paddles?|blades?|wheels?\s+of\s+vanes){gap(6, HELD)}"
          r"(?:in|against|upon|on|through|into|by\s+means\s+of)\s+(?:the\s+)?air(?![\w-])", veto=WINDMILL),
        R(r"\bair[- ]?(?:screws?|propell[eo]rs?)\b"),
        R(r"\b(?:propell\w*|driv(?:e|es|en|ing)|drove|drawn|pulled|push\w*)\s+(?:\w+\s+){0,2}?(?:through|against|in)\s+"
          r"the\s+air\b(?:\s+\w+){0,4}?\s+(?:screws?|propell[eo]rs?|fans?|vanes?)\b", veto=WINDMILL),
        R(rf"\b(?:engines?|motors?|steam|springs?|clockwork|electricity|petroleum|benzine|naphtha|gas[- ]engines?|"
          rf"oil[- ]engines?|machinery){gap(6)}(?:drive|drives|driving|driven|drove|turn\w*|work\w*|revolv\w*|spin\w*|"
          rf"rotat\w*|actuat\w*|move|moves|moving|moved|ply|plies|plying){gap(4)}(?:screws?|propell[eo]rs?|fans?|"
          r"vanes?|paddles?|blades?)\b", requires=r"\b(?:air|aloft|sky|clouds|fly|flying|soar\w*|float\w*|wings?)\b",
          veto=WINDMILL),
    )),
    Concept("flight_control", "principle", "steering and balance in the air: rudder or tail, warped or tilted wings, shifted weight", (
        R(r"\b(?:rudders?|tails?|helms?)\b(?:\s+\w+){0,8}?\s+(?:in|of|upon|against|through)\s+(?:the\s+)?(?:air|wind)\b",
          requires=r"\b(?:steer\w*|guid\w*|direct\w*|turn\w*|balanc\w*|stead\w*|govern\w*|control\w*|manag\w*)\b"),
        R(r"\b(?:horizontal|vertical|second|double|upright)\s+rudders?\b", requires=AIR_CONTEXT),
        R(r"\b(?:warp\w*|twist\w*|tilt\w*|bend\w*|dip\w*|lower\w*|inclin\w*)\s+(?:the\s+|its\s+|his\s+|their\s+|one\s+)?"
          r"(?:\w+\s+)?(?:wings?|wing-?tips?|tips\s+of\s+(?:the|its|his|their)\s+wings|planes?|surfaces?)\b",
          requires=r"\b(?:balanc\w*|steer\w*|guid\w*|equilibri\w*|stead\w*|roll\w*|lateral\w*|turn\w*|control\w*|upset\w*|overturn\w*)\b"),
        R(r"\b(?:shift\w*|mov(?:e|es|ed|ing)|throw\w*|threw|chang\w*|swing\w*|sway\w*)\s+(?:of\s+)?(?:his|the|their|its|our)\s+"
          r"(?:own\s+)?(?:weight|body|centre\s+of\s+gravity|center\s+of\s+gravity|ballast)\b",
          requires=r"\b(?:air|aloft|wings?|sky|fly\w*|soar\w*|glid\w*|float\w*)\b"),
        R(r"\b(?:balance|equilibrium|stability|steadiness|steer\w*|guid\w*|control)\b(?:\s+\w+){0,4}?\s+"
          r"(?:in\s+the\s+air|aloft|in\s+mid-?air)\b"),
    )),
    Concept("lighter_than_air", "principle", "heated air or a light gas, held in a bag or globe, rises and lifts", (
        R(all_of=(r"\b(?:(?:hot|heated|warm|warmed|rarefied|rarified|expanded)\s+air|hydrogen|coal[- ]gas|"
                  r"(?:inflammable|light|lightest)\s+(?:air|gas)|gas\s+lighter\s+than|lighter\s+than\s+(?:the\s+|common\s+|"
                  r"ordinary\s+)?air|helium|smoke)\b",
                  r"\b(?:bags?|globes?|spheres?|envelopes?|bladders?|vessels?|cases?|cylinders?|sacks?|bubbles?|skins?|"
                  r"tents?|canopy|balls?|cover(?:ing)?s?|receptacles?|reservoirs?|silk|silken|chambers?|bod(?:y|ies))\b",
                  LIFT)),
        R(r"\bdisplac\w*(?:\s+\w+){0,4}?\s+(?:air|(?:its\s+own|a\s+greater|an\s+equal)\s+(?:weight|bulk|volume)\s+of\s+air)\b",
          requires=r"\b(?:ris\w*|rose|float\w*|buoy\w*|lift\w*|bear\w*\s+up|ascend\w*|upward\w*)\b"),
    )),
    Concept("powered_flapping", "other", "an engine or motor works flapping wings (not one of the four principles)", (
        R(rf"\b(?:engines?|motors?|steam|springs?|clockwork|electricity|machinery){gap(6)}(?:drive|drives|driving|"
          rf"driven|drove|work\w*|beat\w*|flap\w*|mov(?:e|es|ed|ing)|actuat\w*){gap(3)}wings\b", veto=MYTH),
    )),
)

CONCEPTS_BY_NAME = {c.name: c for c in CONCEPTS}
IDEA_CONCEPTS = tuple(c.name for c in CONCEPTS if c.group == "idea")
PRINCIPLES = tuple(c.name for c in CONCEPTS if c.group == "principle")

_SENTENCE_END = re.compile(r"[.!?;]+(?=\s|$)|\n\s*\n")


@dataclass(frozen=True)
class ConceptHit:
    concept: str
    start: int
    end: int
    text: str  # the matched span (normalised text)


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _rule_hits(rule: Rule, sentence: str, offset: int) -> list[tuple[int, int]]:
    if rule.veto is not None and rule.veto.search(sentence):
        return []
    if rule.requires is not None and not rule.requires.search(sentence):
        return []
    if rule.all_of:
        found = [list(p.finditer(sentence)) for p in rule.all_of]
        if not all(found):
            return []
        # Every component that matches is part of the evidence; the span covers all of them.
        ms = [m for f in found for m in f]
        return [(offset + min(m.start() for m in ms), offset + max(m.end() for m in ms))]
    assert rule.pattern is not None
    return [(offset + m.start(), offset + m.end()) for m in rule.pattern.finditer(sentence)]


def find_concepts(text: str, *, normalized: bool = False, min_end: int = 0) -> list[ConceptHit]:
    """Every concept hit in ``text``; a hit must end after character ``min_end`` (of the normalised text)."""
    norm = text if normalized else normalize(text)
    hits: list[ConceptHit] = []
    for s0, s1 in _sentences(norm):
        if s1 <= min_end:
            continue
        sentence = norm[s0:s1]
        for concept in CONCEPTS:
            for rule in concept.rules:
                for a, b in _rule_hits(rule, sentence, s0):
                    if b > min_end:
                        hits.append(ConceptHit(concept.name, a, b, norm[a:b]))
    return sorted(hits, key=lambda h: (h.start, h.concept))


def concept_set(text: str) -> set[str]:
    return {h.concept for h in find_concepts(text)}


def _join(prefix: str, text: str) -> tuple[str, int]:
    """Normalised prefix + text, and the offset where ``text`` starts."""
    p = normalize(prefix).rstrip() if prefix else ""
    t = normalize(text).strip()
    if not p:
        return t, 0
    glue = "" if text[:1] and not text[:1].isspace() else " "
    return p + glue + t, len(p) + len(glue)


def score_text(text: str, *, prefix: str = "", prompt_concepts: Iterable[str] = ()) -> dict:
    """Score one model output.

    ``prefix`` is the end of a completion prompt the output continues (so
    ``"... the screw would" + " carry a man aloft"`` is read as one sentence);
    matches must reach into the output itself. ``prompt_concepts`` are the
    concepts the prompt already states (``concept_set`` of all user turns): a
    concept found in the output is *novel* only if it is not among them.

    Returns the concepts found, the novel ones, whether the output states the
    idea of human flight (``idea``: an idea concept or a flight-lexicon word),
    the novel principles, and *applied* principles: novel principles in an
    output that also states the idea (the pre-registered H3 outcome).
    """
    joined, boundary = _join(prefix, text)
    hits = find_concepts(joined, normalized=True, min_end=boundary)
    found = sorted({h.concept for h in hits})
    in_prompt = set(prompt_concepts)
    novel = [c for c in found if c not in in_prompt]
    lexicon = sorted({h.text for h in find_flight_hits(text)})
    idea_concepts = [c for c in found if c in IDEA_CONCEPTS]
    novel_principles = [c for c in novel if c in PRINCIPLES]
    idea = bool(idea_concepts or lexicon)
    novel_idea = bool([c for c in novel if c in IDEA_CONCEPTS] or lexicon)
    return {
        "concepts": found,
        "novel": novel,
        "idea_concepts": idea_concepts,
        "flight_lexicon": lexicon,
        "flight_like": idea,
        "flight_like_novel": novel_idea,
        "principles_novel": novel_principles,
        "principles_applied": novel_principles if idea else [],
        "spans": [{"concept": h.concept, "text": h.text} for h in hits],
    }
