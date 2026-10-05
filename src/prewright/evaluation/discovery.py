"""The discovery experiment: can a model that never read about human flight reason its way to it?

Protocol: docs/discovery-experiment.md. Prompts: evals/discovery_prompts.jsonl.
Command line: scripts/discovery_probe.py.

This module holds everything except model loading, so it is tested with the
fake model (``prewright.evaluation.fake``):

* prompts: loading, validation (flight-free, well-formed), word-shuffled
  baselines;
* sampling: N completions per prompt per model, single-turn prompts in one
  batch, Socratic chains turn by turn with the model's own replies fed back;
  chat models use the chat template (``prewright.chat_format``), base models a
  period catechism layout (``Q. ...`` / ``A.``);
* scoring: ``prewright.evaluation.concepts`` on every sample;
* statistics: per-family rates with prompt-clustered bootstrap CIs, the
  pre-registered tests with a Holm correction, ICCs for power planning;
* blinding: a shuffled CSV of transcripts for judges, with model identities in
  a separate key file, and merging the judges' ratings back.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from prewright.evaluation import stats
from prewright.evaluation.concepts import CONCEPTS, PRINCIPLES, concept_set, score_text
from prewright.evaluation.generation import prompt_seed
from prewright.evaluation.interface import Codec, LanguageModel, with_bos
from prewright.evaluation.probes import read_jsonl
from prewright.flight import contains_flight, find_flight_hits

FAMILIES = ("neutral", "leading", "socratic", "goal_stated", "control_topic", "shuffled")
FORMATS = ("chat", "completion")
TEMPLATES = ("chat", "catechism")

# ---------------------------------------------------------------------------
# prompts


@dataclass(frozen=True)
class DiscoveryPrompt:
    id: str
    family: str
    format: str  # "chat": turns are user messages; "completion": turns == [text prefix]
    turns: tuple[str, ...]
    note: str = ""
    source: str = ""  # for "shuffled": the id of the prompt whose words were shuffled
    # Concepts the prompt itself states, declared by its author (the matcher may miss a paraphrase).
    # Together with what the matcher finds in the prompt, they are never credited as novel.
    given: tuple[str, ...] = ()

    @property
    def multi_turn(self) -> bool:
        return len(self.turns) > 1

    @property
    def text(self) -> str:
        return "\n".join(self.turns)

    def to_dict(self) -> dict:
        d = {"id": self.id, "family": self.family, "format": self.format, "turns": list(self.turns)}
        if self.source:
            d["source"] = self.source
        if self.given:
            d["given"] = list(self.given)
        if self.note:
            d["note"] = self.note
        return d


def load_prompts(path: str | Path) -> list[DiscoveryPrompt]:
    return [DiscoveryPrompt(r["id"], r["family"], r.get("format", "chat"), tuple(r["turns"]), r.get("note", ""),
                            r.get("source", ""), tuple(r.get("given", ()))) for r in read_jsonl(path)]


def validate_prompts(prompts: Sequence[DiscoveryPrompt]) -> list[str]:
    """Problems with a prompt set; an empty list means it is usable."""
    problems = []
    ids = [p.id for p in prompts]
    by_id = {p.id: p for p in prompts}
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate id {dup}")
    for p in prompts:
        if p.family not in FAMILIES:
            problems.append(f"{p.id}: unknown family {p.family!r}")
        if p.format not in FORMATS:
            problems.append(f"{p.id}: unknown format {p.format!r}")
        if not p.turns or any(not t.strip() for t in p.turns):
            problems.append(f"{p.id}: empty turn")
        unknown = set(p.given) - {c.name for c in CONCEPTS}
        if unknown:
            problems.append(f"{p.id}: unknown given concept(s) {sorted(unknown)}")
        if p.format == "completion" and len(p.turns) != 1:
            problems.append(f"{p.id}: a completion prompt has exactly one turn")
        for k, t in enumerate(p.turns):
            hits = find_flight_hits(t)
            if hits or contains_flight(t):
                problems.append(f"{p.id} turn {k + 1}: flight-lexicon hit {sorted({h.text for h in hits})}")
        if p.family == "shuffled":
            src = by_id.get(p.source)
            if src is None:
                problems.append(f"{p.id}: shuffled prompt without a valid source")
            elif [sorted(_tokens(t)) for t in src.turns] != [sorted(_tokens(t)) for t in p.turns]:
                problems.append(f"{p.id}: not a word permutation of {p.source}")
    return problems


def _tokens(text: str) -> list[str]:
    return text.split()


def shuffle_words(text: str, seed: int) -> str:
    """The words of ``text`` (whitespace-separated, punctuation attached) in a random order."""
    words = _tokens(text)
    random.Random(seed).shuffle(words)
    return " ".join(words)


def make_shuffled(prompt: DiscoveryPrompt, seed: int, max_tries: int = 100) -> DiscoveryPrompt:
    """A word-shuffled copy of ``prompt`` (each turn shuffled on its own) that is still flight-free.

    Shuffling keeps the vocabulary (what a model could echo) and destroys the
    argument (what it could reason from). A shuffle that happens to create a
    lexicon phrase is redrawn.
    """
    for attempt in range(max_tries):
        turns = tuple(shuffle_words(t, prompt_seed(seed + attempt, f"{prompt.id}/{k}")) for k, t in enumerate(prompt.turns))
        if not any(contains_flight(t) for t in turns):
            return DiscoveryPrompt(f"x-{prompt.id}", "shuffled", prompt.format, turns,
                                   f"word-shuffled copy of {prompt.id}", prompt.id, prompt.given)
    raise ValueError(f"no flight-free shuffle of {prompt.id} in {max_tries} tries")


# ---------------------------------------------------------------------------
# sampling


@dataclass(frozen=True)
class SamplingSettings:
    samples: int = 8
    max_new_tokens: int = 80  # per turn
    temperature: float = 0.8
    top_p: float | None = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# The defaults of scripts/generate.py (base models) and scripts/chat.py (chat models).
DEFAULT_SETTINGS = {
    "catechism": SamplingSettings(temperature=0.8, top_p=0.95, repetition_penalty=1.0),
    "chat": SamplingSettings(temperature=0.7, top_p=0.9, repetition_penalty=1.1),
}

_CATECHISM_NEXT_Q = re.compile(r"\n\s*(?:Q\.|Q:|Ques\.|Question[.:])")
_SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>")


def _chat_tokenizer(codec: Codec):
    """What ``prewright.chat_format`` should wrap: the project tokenizer inside a TokenizerCodec, else the codec."""
    return getattr(codec, "tok", None) or codec


def encode_turn(codec: Codec, prompt: DiscoveryPrompt, history: Sequence[dict], template: str,
                context_length: int, reserve: int) -> list[int]:
    """Token ids to sample the reply to the last user message in ``history``.

    ``history`` alternates {"role": "user"|"assistant", "content": ...} and ends
    with a user turn. Completion prompts ignore ``template``.
    """
    budget = max(8, context_length - reserve)
    if prompt.format == "completion":
        ids = with_bos(codec, codec.encode(prompt.turns[0]))
        return ids if len(ids) <= budget else ids[:1] + ids[-(budget - 1):]
    if template == "chat":
        from prewright.chat_format import build_prompt
        return build_prompt(_chat_tokenizer(codec), list(history), context_length, reserve=reserve)
    if template == "catechism":
        text = ""
        for m in history:
            text += f"Q. {m['content'].strip()}\n" if m["role"] == "user" else f"A. {m['content'].strip()}\n\n"
        ids = with_bos(codec, codec.encode(text + "A."))
        return ids if len(ids) <= budget else ids[:1] + ids[-(budget - 1):]
    raise ValueError(f"unknown template {template!r}")


def clean_reply(text: str, template: str, fmt: str) -> str:
    for s in _SPECIALS:
        text = text.replace(s, "")
    if fmt == "chat" and template == "catechism":
        m = _CATECHISM_NEXT_Q.search(text)
        if m:
            text = text[:m.start()]
        return text.strip()
    return text if fmt == "completion" else text.strip()


def _sample(lm: LanguageModel, ids: list[int], n: int, s: SamplingSettings, seed: int, stop: list[int]) -> list[list[int]]:
    extra = {"repetition_penalty": s.repetition_penalty} if s.repetition_penalty != 1.0 else {}
    return lm.sample(ids, n, max_new_tokens=s.max_new_tokens, temperature=s.temperature, top_k=s.top_k,
                     top_p=s.top_p, seed=seed, stop_ids=stop, **extra)


def sample_prompt(lm: LanguageModel, codec: Codec, prompt: DiscoveryPrompt, template: str,
                  settings: SamplingSettings, seed: int) -> list[dict]:
    """``settings.samples`` independent samples of ``prompt``; each is {sample, seeds, replies}.

    Seeds depend only on (seed, prompt id, sample, turn), never on the model,
    so main and control are sampled with the same random numbers.
    """
    stop = [t for t in (getattr(codec, "eot_id", None), getattr(codec, "user_id", None)) if t is not None]
    reserve = min(settings.max_new_tokens, lm.context_length // 2)
    if not prompt.multi_turn:
        s = prompt_seed(seed, prompt.id)
        ids = encode_turn(codec, prompt, [{"role": "user", "content": prompt.turns[0]}], template,
                          lm.context_length, reserve)
        outs = _sample(lm, ids, settings.samples, settings, s, stop)
        return [{"sample": j, "seeds": [s], "replies": [clean_reply(codec.decode(o), template, prompt.format)]}
                for j, o in enumerate(outs)]
    results = []
    for j in range(settings.samples):
        history: list[dict] = []
        seeds, replies = [], []
        for k, turn in enumerate(prompt.turns):
            history.append({"role": "user", "content": turn})
            s = prompt_seed(seed, f"{prompt.id}#{j}#{k}")
            ids = encode_turn(codec, prompt, history, template, lm.context_length, reserve)
            (out,) = _sample(lm, ids, 1, settings, s, stop)
            reply = clean_reply(codec.decode(out), template, prompt.format)
            history.append({"role": "assistant", "content": reply})
            seeds.append(s)
            replies.append(reply)
        results.append({"sample": j, "seeds": seeds, "replies": replies})
    return results


def prompt_concepts(prompt: DiscoveryPrompt) -> set[str]:
    """What the prompt already states: the matcher's concepts in its text plus the author-declared ``given``."""
    return concept_set(prompt.text) | set(prompt.given)


def score_sample(prompt: DiscoveryPrompt, replies: Sequence[str]) -> dict:
    """Concept scores of one sample: a completion is read with the end of its prompt; chain replies together."""
    given = prompt_concepts(prompt)
    if prompt.format == "completion":
        return score_text(replies[0], prefix=prompt.turns[0][-400:], prompt_concepts=given)
    return score_text("\n\n".join(r for r in replies), prompt_concepts=given)


def settings_fingerprint(model: str, template: str, settings: SamplingSettings, seed: int) -> str:
    blob = json.dumps({"model": model, "template": template, "settings": settings.to_dict(), "seed": seed},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def run_discovery(models: dict[str, tuple[LanguageModel, str, SamplingSettings]], codec: Codec,
                  prompts: Sequence[DiscoveryPrompt], seed: int, out_jsonl: str | Path | None = None,
                  done: Iterable[tuple[str, str, str]] = (), log: Callable[[str], None] = lambda m: None) -> list[dict]:
    """Sample and score every prompt for every model.

    ``models`` maps a name to (model, template, settings). Records are appended
    to ``out_jsonl`` as each (model, prompt) finishes, so an interrupted run
    keeps its work; ``done`` lists (model, prompt id, fingerprint) to skip.
    """
    done = set(done)
    records: list[dict] = []
    fh = open(out_jsonl, "a", encoding="utf-8") if out_jsonl else None
    try:
        for name, (lm, template, settings) in models.items():
            fp = settings_fingerprint(name, template, settings, seed)
            for i, p in enumerate(prompts):
                if (name, p.id, fp) in done:
                    continue
                log(f"[{name}] {i + 1}/{len(prompts)} {p.id} ({p.family}, {len(p.turns)} turn(s))")
                for smp in sample_prompt(lm, codec, p, template, settings, seed):
                    rec = {"model": name, "template": template, "prompt_id": p.id, "family": p.family,
                           "format": p.format, "source": p.source, "multi_turn": p.multi_turn,
                           "sample": smp["sample"], "seeds": smp["seeds"], "replies": smp["replies"],
                           "score": score_sample(p, smp["replies"]), "fingerprint": fp}
                    records.append(rec)
                    if fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if fh:
                    fh.flush()
    finally:
        if fh:
            fh.close()
    return records


def read_records(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def rescore(records: Sequence[dict], prompts: Sequence[DiscoveryPrompt]) -> list[dict]:
    """Re-apply the concept matcher (e.g. after editing concepts.py) without sampling again."""
    by_id = {p.id: p for p in prompts}
    out = []
    for r in records:
        r = dict(r)
        r["score"] = score_sample(by_id[r["prompt_id"]], r["replies"])
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# outcomes and statistics


def _judge_flight(r: dict) -> float:
    j = r.get("judge")
    return float(j["flight_idea"] >= 2) if j and j.get("flight_idea") is not None else math.nan


def _judge_principles(r: dict) -> float:
    j = r.get("judge")
    if not j or j.get("flight_idea") is None:
        return math.nan
    if j["flight_idea"] < 2:
        return 0.0
    return float(sum(1 for c in PRINCIPLES if (j.get(c) or 0) >= 0.5))


OUTCOMES: dict[str, Callable[[dict], float]] = {
    "flight_like": lambda r: float(r["score"]["flight_like"]),
    "flight_like_novel": lambda r: float(r["score"]["flight_like_novel"]),
    "idea_concept": lambda r: float(bool(r["score"]["idea_concepts"])),
    "flight_lexicon": lambda r: float(bool(r["score"]["flight_lexicon"])),
    "principles_applied": lambda r: float(len(r["score"]["principles_applied"])),
    "any_principle_applied": lambda r: float(bool(r["score"]["principles_applied"])),
    "principles_novel": lambda r: float(len(r["score"]["principles_novel"])),
    "judge_flight": _judge_flight,
    "judge_principles": _judge_principles,
}
for _c in CONCEPTS:
    OUTCOMES[f"concept:{_c.name}"] = (lambda name: lambda r: float(name in r["score"]["novel"]))(_c.name)


def _select(records: Sequence[dict], model: str, families: Sequence[str], turns: str = "any") -> list[dict]:
    return [r for r in records if r["model"] == model and r["family"] in families
            and (turns == "any" or (turns == "single") != bool(r.get("multi_turn")))]


def prompt_means(records: Sequence[dict], outcome: str) -> dict[str, float]:
    """Mean outcome per prompt (samples of one prompt are one cluster). NaN outcomes are skipped."""
    f = OUTCOMES[outcome]
    acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = f(r)
        if not math.isnan(v):
            acc[r["prompt_id"]].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def icc_anova(groups: Sequence[Sequence[float]]) -> float:
    """One-way ANOVA estimate of the intra-class correlation (samples within prompts); NaN if undefined."""
    groups = [list(g) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return math.nan
    grand = sum(sum(g) for g in groups) / n
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else math.nan


def family_rates(records: Sequence[dict], models: Sequence[str], outcome: str, seed: int = 0) -> dict:
    """{model: {family: {mean, low, high, n_prompts, n_samples, icc}}} with a prompt-clustered bootstrap CI."""
    f = OUTCOMES[outcome]
    out: dict = {}
    for m in models:
        out[m] = {}
        for fam in FAMILIES:
            rs = [r for r in records if r["model"] == m and r["family"] == fam]
            vals = [(f(r), r["prompt_id"]) for r in rs]
            vals = [(v, c) for v, c in vals if not math.isnan(v)]
            if not vals:
                continue
            ci = stats.cluster_bootstrap_mean([v for v, _ in vals], [c for _, c in vals], seed=seed)
            groups: dict[str, list[float]] = defaultdict(list)
            for v, c in vals:
                groups[c].append(v)
            out[m][fam] = {"mean": ci.estimate, "low": ci.low, "high": ci.high, "n_prompts": len(groups),
                           "n_samples": len(vals), "icc": icc_anova(list(groups.values()))}
    return out


@dataclass(frozen=True)
class TestSpec:
    id: str
    hypothesis: str
    description: str
    outcome: str
    kind: str  # "families" (A vs B, one model), "shuffled" (A vs its shuffles), "models" (control - main), "did"
    family_a: tuple[str, ...]
    family_b: tuple[str, ...] = ()
    model: str = "main"  # role: "main" or "control"
    turns: str = "any"  # "any" | "single" | "multi"
    expected: int = 1  # expected sign of the estimate
    primary: bool = False


PREREGISTERED: tuple[TestSpec, ...] = (
    TestSpec("H2", "H2", "flight-naive model: flight-like outputs, leading vs topic-matched control prompts (single-turn)",
             "flight_like", "families", ("leading",), ("control_topic",), turns="single", primary=True),
    TestSpec("H3", "H3", "flight-naive model: principles applied in a proposal, leading prompts vs their word-shuffled copies",
             "principles_applied", "shuffled", ("leading",), ("shuffled",), primary=True),
    TestSpec("H1g", "H1", "generative H1: flight-like outputs, control minus flight-naive, on leading and Socratic prompts",
             "flight_like", "models", ("leading", "socratic"), primary=True),
    # secondary
    TestSpec("H2-shuf", "H2", "flight-naive: flight-like, leading vs word-shuffled copies (paired)",
             "flight_like", "shuffled", ("leading",), ("shuffled",)),
    TestSpec("H2-neutral", "H2", "flight-naive: flight-like, leading vs neutral", "flight_like", "families",
             ("leading",), ("neutral",)),
    TestSpec("H2-socratic", "H2", "flight-naive: flight-like, Socratic chains vs control-topic chains",
             "flight_like", "families", ("socratic",), ("control_topic",), turns="multi"),
    TestSpec("H2-novel", "H2", "flight-naive: flight-like and not already in the prompt, leading vs control-topic",
             "flight_like_novel", "families", ("leading",), ("control_topic",), turns="single"),
    TestSpec("H2-ctl", "H2", "control model: flight-like, leading vs control-topic (single-turn)",
             "flight_like", "families", ("leading",), ("control_topic",), model="control", turns="single"),
    TestSpec("DiD", "H1/H2", "how much more the leading prompts lift the control than the flight-naive model "
             "((control-main) on leading minus (control-main) on control-topic)",
             "flight_like", "did", ("leading", "socratic"), ("control_topic",)),
    TestSpec("H3-goal", "H3", "flight-naive: principles applied, goal-stated vs control-topic prompts",
             "principles_applied", "families", ("goal_stated",), ("control_topic",)),
    TestSpec("H3-any", "H3", "flight-naive: any novel principle (with or without a proposal), leading vs shuffled",
             "principles_novel", "shuffled", ("leading",), ("shuffled",)),
    TestSpec("lexicon", "leak", "flight-lexicon words in outputs, control minus flight-naive, all prompts",
             "flight_lexicon", "models", FAMILIES),
)
JUDGE_TESTS: tuple[TestSpec, ...] = (
    TestSpec("H2-judge", "H2", "judge: flight proposal (rating >= 2), leading vs control-topic (single-turn)",
             "judge_flight", "families", ("leading",), ("control_topic",), turns="single", primary=True),
    TestSpec("H3-judge", "H3", "judge: principles in a proposal, leading vs shuffled",
             "judge_principles", "shuffled", ("leading",), ("shuffled",), primary=True),
    TestSpec("H1g-judge", "H1", "judge: flight proposals, control minus flight-naive, leading and Socratic",
             "judge_flight", "models", ("leading", "socratic"), primary=True),
)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are left out of the family and stay NaN)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adj = [math.nan] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def _empty(spec: TestSpec, why: str) -> dict:
    return {**asdict(spec), "estimate": math.nan, "low": math.nan, "high": math.nan, "p": math.nan,
            "n_a": 0, "n_b": 0, "note": why}


def run_test(records: Sequence[dict], spec: TestSpec, roles: dict[str, str], seed: int = 0) -> dict:
    """Compute one pre-registered comparison. Every test works on per-prompt means (the prompt is the unit)."""
    model = roles.get(spec.model, spec.model)
    if spec.kind == "families":
        a = prompt_means(_select(records, model, spec.family_a, spec.turns), spec.outcome)
        b = prompt_means(_select(records, model, spec.family_b, spec.turns), spec.outcome)
        if len(a) < 2 or len(b) < 2:
            return _empty(spec, "fewer than 2 prompts in a group")
        x, y = list(a.values()), list(b.values())
        ci = stats.bootstrap_diff_of_means(x, y, seed=seed)
        p = stats.two_sample_permutation_test(x, y, seed=seed)
        return {**asdict(spec), "estimate": ci.estimate, "low": ci.low, "high": ci.high, "p": p,
                "n_a": len(x), "n_b": len(y), "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)), "note": ""}
    if spec.kind == "shuffled":
        a = prompt_means(_select(records, model, spec.family_a, spec.turns), spec.outcome)
        sh = _select(records, model, spec.family_b, spec.turns)
        b = prompt_means(sh, spec.outcome)
        source = {r["prompt_id"]: r.get("source", "") for r in sh}
        pairs = [(a[source[s]], b[s]) for s in b if source.get(s) in a]
        if len(pairs) < 2:
            return _empty(spec, "fewer than 2 prompt/shuffle pairs")
        diffs = [u - v for u, v in pairs]
        ci = stats.bootstrap_mean(diffs, seed=seed)
        p = stats.paired_permutation_test(diffs, seed=seed)
        return {**asdict(spec), "estimate": ci.estimate, "low": ci.low, "high": ci.high, "p": p,
                "n_a": len(pairs), "n_b": len(pairs), "mean_a": float(np.mean([u for u, _ in pairs])),
                "mean_b": float(np.mean([v for _, v in pairs])), "note": ""}
    main, control = roles.get("main", "main"), roles.get("control", "control")
    if spec.kind == "models":
        m = prompt_means(_select(records, main, spec.family_a, spec.turns), spec.outcome)
        c = prompt_means(_select(records, control, spec.family_a, spec.turns), spec.outcome)
        common = sorted(set(m) & set(c))
        if len(common) < 2:
            return _empty(spec, "fewer than 2 prompts sampled for both models")
        diffs = [c[k] - m[k] for k in common]
        ci = stats.bootstrap_mean(diffs, seed=seed)
        p = stats.paired_permutation_test(diffs, seed=seed)
        return {**asdict(spec), "estimate": ci.estimate, "low": ci.low, "high": ci.high, "p": p,
                "n_a": len(common), "n_b": len(common), "mean_a": float(np.mean([c[k] for k in common])),
                "mean_b": float(np.mean([m[k] for k in common])), "note": ""}
    if spec.kind == "did":
        def deltas(fams: Sequence[str]) -> list[float]:
            m = prompt_means(_select(records, main, fams, spec.turns), spec.outcome)
            c = prompt_means(_select(records, control, fams, spec.turns), spec.outcome)
            return [c[k] - m[k] for k in sorted(set(m) & set(c))]
        x, y = deltas(spec.family_a), deltas(spec.family_b)
        if len(x) < 2 or len(y) < 2:
            return _empty(spec, "fewer than 2 prompts in a group")
        ci = stats.bootstrap_diff_of_means(x, y, seed=seed)
        p = stats.two_sample_permutation_test(x, y, seed=seed)
        return {**asdict(spec), "estimate": ci.estimate, "low": ci.low, "high": ci.high, "p": p,
                "n_a": len(x), "n_b": len(y), "note": ""}
    raise ValueError(f"unknown test kind {spec.kind!r}")


def run_tests(records: Sequence[dict], roles: dict[str, str], specs: Sequence[TestSpec] = PREREGISTERED,
              seed: int = 0) -> list[dict]:
    """All comparisons; Holm over the primary ones (and, separately, over the secondary ones)."""
    results = [run_test(records, s, roles, seed) for s in specs]
    for flag in (True, False):
        idx = [i for i, r in enumerate(results) if r["primary"] == flag]
        adj = holm([results[i]["p"] for i in idx])
        for i, a in zip(idx, adj):
            results[i]["p_holm"] = a
    for r in results:
        r["supported"] = bool(not math.isnan(r["p_holm"]) and r["p_holm"] < 0.05
                              and np.sign(r["estimate"]) == r["expected"])
    return results


def matcher_judge_agreement(records: Sequence[dict]) -> dict | None:
    """Agreement of the automatic flight-like flag with the judges' flight-proposal rating (>= 2)."""
    pairs = [(OUTCOMES["flight_like"](r), OUTCOMES["judge_flight"](r)) for r in records]
    pairs = [(a, j) for a, j in pairs if not math.isnan(j)]
    if not pairs:
        return None
    tp = sum(1 for a, j in pairs if a and j)
    fp = sum(1 for a, j in pairs if a and not j)
    fn = sum(1 for a, j in pairs if not a and j)
    tn = len(pairs) - tp - fp - fn
    n = len(pairs)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else math.nan
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan, "kappa": kappa}


# ---------------------------------------------------------------------------
# power


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cluster_power(p0: float, p1: float, prompts_a: int, prompts_b: int, samples: int, icc: float,
                  alpha: float = 0.05) -> float:
    """Approximate power of a two-sided comparison of two rates, with ``samples`` per prompt.

    Normal approximation with the design effect 1 + (m - 1) * icc for
    samples clustered within prompts; the effective sample size of a family
    can never exceed prompts / icc, however many samples are drawn.
    """
    de = 1 + (samples - 1) * icc
    v = p1 * (1 - p1) * de / (prompts_a * samples) + p0 * (1 - p0) * de / (prompts_b * samples)
    if v <= 0:
        return 1.0
    z = abs(p1 - p0) / math.sqrt(v)
    zc = -_ndtri(alpha / 2)
    return _phi(z - zc) + _phi(-z - zc)


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF (bisection; plenty for power tables)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# blinding


PERSON_TERMS = {"aeronaut", "airman", "aviation", "montgolfier", "lunardi", "garnerin", "santos_dumont", "lilienthal",
                "chanute", "pilatre_de_rozier", "zambeccari", "tissandier", "wright_brothers", "lana_terzi",
                "malmesbury_flier", "blanchard", "langley", "maxim", "glaisher", "cayley", "flight_experimenters"}
JUDGE_COLUMNS = ("flight_idea", "fixed_wing", "propulsion", "flight_control", "lighter_than_air", "powered_flapping",
                 "coherence", "source_guess", "notes")


def redact_flight_terms(text: str, max_passes: int = 3) -> tuple[str, int]:
    """Replace flight-lexicon words with neutral ones ("contrivance", "traveller"); returns (text, n replaced).

    The control model has read the flight words and the flight-naive one has
    not, so a judge who sees "balloon" knows which model wrote the text. The
    replacement is a plain word, not a marker, so redaction itself does not
    give the model away. The idea around the word stays for the judge to rate.
    """
    n = 0
    for _ in range(max_passes):
        hits = find_flight_hits(text)
        if not hits:
            break
        for h in sorted(hits, key=lambda h: -len(h.text)):
            word = "traveller" if h.term in PERSON_TERMS or h.text.startswith(("aeronaut", "aviator")) else "contrivance"
            parts = [re.escape(w) for w in re.split(r"[\s-]+", h.text) if w]
            rx = re.compile(r"[\s-]+".join(parts), re.IGNORECASE)

            def repl(m: re.Match) -> str:
                return word.capitalize() if m.group(0)[:1].isupper() else word
            text, k = rx.subn(repl, text)
            n += k
    return text, n


def transcript(prompt: DiscoveryPrompt, replies: Sequence[str]) -> str:
    """What a judge reads: the prompt and the model's text, clearly separated."""
    if prompt.format == "completion":
        return f"PROMPT: {prompt.turns[0]}\nCONTINUATION: {replies[0].strip()}"
    return "\n".join(f"QUESTION: {q.strip()}\nANSWER: {a.strip()}" for q, a in zip(prompt.turns, replies))


def export_blinded(records: Sequence[dict], prompts: Sequence[DiscoveryPrompt], items_csv: str | Path,
                   key_csv: str | Path, seed: int = 0, redact: bool = True,
                   calibration: Sequence[dict] = ()) -> int:
    """Write the judges' CSV (shuffled transcripts, opaque ids, empty rating columns) and the key, separately.

    The items file holds no model name, prompt id, family, sample number or
    seed. ``calibration`` items ({"id", "text"} with known labels kept in the
    key) are mixed in to check the judges. Returns the number of items.
    """
    by_id = {p.id: p for p in prompts}
    entries = []
    for r in records:
        p = by_id[r["prompt_id"]]
        replies = list(r["replies"])
        n_red = 0
        if redact:
            red = [redact_flight_terms(x) for x in replies]
            replies, n_red = [x for x, _ in red], sum(k for _, k in red)
        entries.append(({"transcript": transcript(p, replies)},
                        {"model": r["model"], "template": r["template"], "prompt_id": r["prompt_id"],
                         "family": r["family"], "sample": r["sample"], "redactions": n_red}))
    for c in calibration:
        entries.append(({"transcript": c["text"]},
                        {"model": "calibration", "template": "", "prompt_id": c["id"], "family": "calibration",
                         "sample": 0, "redactions": 0, "label": json.dumps(c.get("label", {}))}))
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    Path(items_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(key_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(items_csv, "w", newline="", encoding="utf-8") as fi, open(key_csv, "w", newline="", encoding="utf-8") as fk:
        wi = csv.writer(fi)
        wk = csv.writer(fk)
        wi.writerow(("item_id", "transcript") + JUDGE_COLUMNS)
        wk.writerow(("item_id", "model", "template", "prompt_id", "family", "sample", "redactions", "label"))
        for rank, i in enumerate(order):
            item, key = entries[i]
            item_id = f"J{rank + 1:05d}"
            wi.writerow((item_id, item["transcript"]) + ("",) * len(JUDGE_COLUMNS))
            wk.writerow((item_id, key["model"], key["template"], key["prompt_id"], key["family"], key["sample"],
                         key["redactions"], key.get("label", "")))
    return len(entries)


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def merge_judgments(records: Sequence[dict], ratings_csv: str | Path, key_csv: str | Path) -> tuple[list[dict], dict]:
    """Attach the judges' ratings (averaged over raters; several rows per item allowed) to the records.

    Returns (records with a ``judge`` field where rated, calibration and
    blinding summary).
    """
    key = {row["item_id"]: row for row in csv.DictReader(open(key_csv, encoding="utf-8"))}
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    guesses: list[tuple[str, str]] = []
    for row in csv.DictReader(open(ratings_csv, encoding="utf-8")):
        k = row.get("item_id", "")
        if k not in key:
            continue
        for c in JUDGE_COLUMNS[:-2]:
            v = _num(row.get(c, ""))
            if v is not None:
                acc[k][c].append(v)
        g = (row.get("source_guess") or "").strip().lower()
        if g:
            guesses.append((key[k]["model"], g))
    by_cell = {(v["model"], v["prompt_id"], int(v["sample"])): k for k, v in key.items() if v["model"] != "calibration"}
    out = []
    for r in records:
        r = dict(r)
        k = by_cell.get((r["model"], r["prompt_id"], int(r["sample"])))
        if k in acc:
            r["judge"] = {c: float(np.mean(v)) for c, v in acc[k].items()}
            r["judge"].setdefault("flight_idea", None)
        out.append(r)
    calib = []
    for k, v in key.items():
        if v["model"] == "calibration" and k in acc:
            calib.append({"id": v["prompt_id"], "label": json.loads(v["label"] or "{}"),
                          "rating": {c: float(np.mean(x)) for c, x in acc[k].items()}})
    return out, {"calibration": calib, "source_guesses": guesses}


# ---------------------------------------------------------------------------
# leakage helpers


def ngram_queries(records: Sequence[dict], n: int = 8, outcome: str = "flight_like") -> list[str]:
    """Distinct n-word spans of the positive samples, for a fixed-string search of the training corpus.

    A "discovery" that is a near-copy of a passage the scrub missed is a leak,
    not a discovery: ``rg -i -F -f positives_ngrams.txt main/train.jsonl``.
    """
    f = OUTCOMES[outcome]
    spans: set[str] = set()
    for r in records:
        if f(r) > 0:
            words = re.findall(r"[A-Za-z']+", " ".join(r["replies"]))
            for i in range(len(words) - n + 1):
                spans.add(" ".join(words[i:i + n]).lower())
    return sorted(spans)


# ---------------------------------------------------------------------------
# report


def _fmt(x: float, pct: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def _p(x: float) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def summarize(records: Sequence[dict], prompts: Sequence[DiscoveryPrompt], roles: dict[str, str],
              seed: int = 0) -> dict:
    models = list(dict.fromkeys(r["model"] for r in records))
    judged = any("judge" in r for r in records)
    outcomes = ["flight_like", "flight_like_novel", "idea_concept", "flight_lexicon", "principles_applied",
                "principles_novel"] + [f"concept:{c.name}" for c in CONCEPTS]
    if judged:
        outcomes += ["judge_flight", "judge_principles"]
    specs = list(PREREGISTERED) + (list(JUDGE_TESTS) if judged else [])
    lexicon_hits = defaultdict(list)
    for r in records:
        for w in r["score"]["flight_lexicon"]:
            lexicon_hits[r["model"]].append({"prompt_id": r["prompt_id"], "sample": r["sample"], "word": w})
    return {
        "models": models,
        "roles": roles,
        "n_records": len(records),
        "prompt_problems": validate_prompts(prompts),
        "families": {f: sum(1 for p in prompts if p.family == f) for f in FAMILIES},
        "rates": {o: family_rates(records, models, o, seed) for o in outcomes},
        "tests": run_tests(records, roles, specs, seed),
        "lexicon_hits": dict(lexicon_hits),
        "agreement": matcher_judge_agreement(records) if judged else None,
    }


def summary_markdown(summary: dict, records: Sequence[dict], prompts: Sequence[DiscoveryPrompt], meta: dict,
                     examples: int = 12) -> str:
    models = summary["models"]
    by_id = {p.id: p for p in prompts}
    L = ["# Discovery probe: summary", "",
         "Protocol: `docs/discovery-experiment.md`. Every rate is a share of samples; the 95% intervals resample "
         "whole prompts (samples of one prompt are not independent). Scores are from the automatic concept matcher "
         "(`prewright.evaluation.concepts`) unless marked *judge*.", ""]
    L += ["## Run", "", "| | |", "|---|---|"]
    for k, v in meta.items():
        L.append(f"| {k} | `{v}` |" if not isinstance(v, dict) else f"| {k} | `{json.dumps(v, sort_keys=True)}` |")
    L += [f"| samples (records) | {summary['n_records']} |", ""]
    if summary["prompt_problems"]:
        L += ["**Prompt-set problems (the run is not valid until these are fixed):**", ""]
        L += [f"- {p}" for p in summary["prompt_problems"]] + [""]
    else:
        L += ["All prompts pass the checks: every turn is free of flight-lexicon words (`prewright.flight`), and "
              "every shuffled prompt is a word permutation of its source.", ""]

    def table(outcome: str, title: str, pct: bool = True) -> None:
        rates = summary["rates"].get(outcome, {})
        L.extend([f"### {title}", "", "| family | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)])
        for fam in FAMILIES:
            cells = []
            for m in models:
                c = rates.get(m, {}).get(fam)
                cells.append("–" if c is None else
                             f"{_fmt(c['mean'], pct)} [{_fmt(c['low'], pct)}, {_fmt(c['high'], pct)}] "
                             f"({c['n_prompts']}p/{c['n_samples']}s)")
            L.append(f"| {fam} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Rates by prompt family", ""]
    table("flight_like", "Flight-like output (an idea concept or a flight-lexicon word)")
    table("flight_like_novel", "Flight-like and not already stated by the prompt")
    table("flight_lexicon", "Uses a flight-lexicon word or phrase (for the flight-naive model every one is read by hand)")
    table("principles_applied", "Principles applied: novel principles in a flight-like output (mean count per sample, 0-4)", pct=False)
    table("principles_novel", "Novel principles, with or without a proposal (mean count per sample)", pct=False)
    if "judge_flight" in summary["rates"]:
        table("judge_flight", "*Judge*: flight proposal (rating >= 2)")
        table("judge_principles", "*Judge*: principles in a proposal (mean count)", pct=False)

    L += ["### Concepts (novel, share of samples) on leading + Socratic prompts vs control-topic prompts", "",
          "| concept | group | " + " | ".join(f"{m}: leading+Socratic | {m}: control-topic" for m in models) + " |",
          "|---|---|" + "---|---|" * len(models)]
    for c in CONCEPTS:
        cells = []
        for m in models:
            for fams in (("leading", "socratic"), ("control_topic",)):
                rs = [r for r in records if r["model"] == m and r["family"] in fams]
                k = sum(1 for r in rs if c.name in r["score"]["novel"])
                cells.append(f"{k}/{len(rs)}" if rs else "–")
        L.append(f"| `{c.name}` | {c.group} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Pre-registered comparisons", "",
          "Estimates are differences in per-prompt means (A − B; for model comparisons control − flight-naive). "
          "p: permutation test over prompts (two-sided); Holm-adjusted within the primary and, separately, the "
          "secondary family. *Supported* = Holm p < 0.05 with the expected sign.", "",
          "| test | primary | comparison | outcome | A | B | estimate [95% CI] | p | p (Holm) | supported |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in summary["tests"]:
        pct = t["outcome"] not in ("principles_applied", "principles_novel", "judge_principles")
        a = _fmt(t.get("mean_a", math.nan), pct)
        b = _fmt(t.get("mean_b", math.nan), pct)
        est = f"{_fmt(t['estimate'], pct)} [{_fmt(t['low'], pct)}, {_fmt(t['high'], pct)}]"
        note = f" ({t['note']})" if t.get("note") else ""
        L.append(f"| {t['id']} | {'yes' if t['primary'] else ''} | {t['description']}{note} | `{t['outcome']}` | "
                 f"{a} (n={t['n_a']}) | {b} (n={t['n_b']}) | {est} | {_p(t['p'])} | {_p(t['p_holm'])} | "
                 f"{'**yes**' if t['supported'] else 'no'} |")
    L.append("")

    L += ["## Flight-lexicon words in outputs", ""]
    for m in models:
        hits = summary["lexicon_hits"].get(m, [])
        L.append(f"- **{m}**: {len(hits)} sample(s)" + (": " + ", ".join(
            f"`{h['word']}` ({h['prompt_id']}#{h['sample']})" for h in hits[:20]) if hits else ""))
    L += ["", "For the flight-naive model each one must be read by hand. It is one of: a leak (a passage the scrub "
          "missed, checked with `positives_ngrams.txt`); the tokenizer composing a word by chance (`ball|oon`); or a "
          "descriptive lexicon phrase the model reached by itself (`man_lifting_kite`, `wings_machine`, "
          "`air_travel_tech` and `art_of_flying` match descriptions such as \"a kite might carry a man\" or "
          "\"human flight\"). Only the last is evidence of discovery.", ""]

    if summary.get("agreement"):
        a = summary["agreement"]
        L += ["## Matcher vs judges", "",
              f"n = {a['n']}; precision {_fmt(a['precision'])}, recall {_fmt(a['recall'])}, Cohen's kappa "
              f"{_fmt(a['kappa'], pct=False)} (matcher flight-like vs judge rating >= 2).", ""]

    L += ["## Clustering (for power planning)", "",
          "Intra-class correlation of the flight-like flag within prompts (ANOVA estimator). The effective "
          "sample size of a family is at most (number of prompts) / ICC, whatever the samples per prompt.", "",
          "| family | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for fam in FAMILIES:
        cells = []
        for m in models:
            c = summary["rates"]["flight_like"].get(m, {}).get(fam)
            cells.append("–" if c is None else _fmt(c["icc"], pct=False))
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Examples (flight-like samples, matched spans in bold)", ""]
    shown = 0
    for fam in ("leading", "socratic", "goal_stated", "neutral", "shuffled", "control_topic"):
        for r in records:
            if shown >= examples:
                break
            if r["family"] != fam or not r["score"]["flight_like"]:
                continue
            p = by_id[r["prompt_id"]]
            text = " / ".join(x.strip() for x in r["replies"])
            for s in sorted({s["text"] for s in r["score"]["spans"]}, key=len, reverse=True):
                text = re.sub(re.escape(s), lambda m: f"**{m.group(0)}**", text, count=1, flags=re.IGNORECASE)
            L += [f"**{r['model']}**, `{r['prompt_id']}` ({fam}), sample {r['sample']}: "
                  f"concepts {', '.join(r['score']['concepts']) or '-'}; lexicon {', '.join(r['score']['flight_lexicon']) or '-'}",
                  "", "> " + p.turns[-1].replace("\n", " ")[:300], ">", "> → " + text.replace("\n", " ")[:800], ""]
            shown += 1
    if not shown:
        L += ["(none)", ""]
    L += ["## Reading this honestly", "",
          "- The matcher is a conservative screen: it misses ideas phrased in unforeseen words and can fire on "
          "figurative text. The blinded judges are the validity check; quote judge-based results for the headline.",
          "- Rates from a few prompts per family are dominated by prompt-to-prompt variation (see the ICC table); "
          "a family difference is evidence only if it holds across prompts, which is what the tests measure.",
          "- Before calling a flight-like sample a discovery, check it against the training corpus "
          "(`positives_ngrams.txt`, see the protocol) and against the prompt it answers.", ""]
    return "\n".join(L)
