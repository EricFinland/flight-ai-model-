"""Paired surprisal probes and cloze knowledge probes.

Paired probes (``evals/paired_probes.jsonl``)
    One shared ``context`` and two continuations in the same syntactic slot.
    ``kind: "flight"`` items pit a flight wording (``flight``) against an
    everyday 1900 wording (``neutral``). ``kind: "control"`` items pit two
    non-flight wordings (``target`` against ``neutral``); the two models should
    agree on these. For each model, the *gap* of an item is
    ``log p(flight|context) - log p(neutral|context)`` (natural log). The test
    statistic is the paired difference ``d = gap_main - gap_control``: it
    cancels everything the two models share (tokenizer, word frequency, length
    of the options, grammar) and leaves what the flight scrub changed.

Knowledge probes (``evals/knowledge_probes.jsonl``)
    A ``prompt``, a correct ``answer`` and one or more ``distractors``. The
    model "knows" the item if the answer gets the highest summed log-prob.
    ``kind: "general"`` items are things both models should know;
    ``kind: "flight"`` items are flight facts only the control should know.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from prewright.evaluation import stats
from prewright.evaluation.interface import Codec, LanguageModel
from prewright.evaluation.scoring import score_continuations

# ---------------------------------------------------------------------------
# Loading


@dataclass(frozen=True)
class PairedProbe:
    id: str
    kind: str  # "flight" | "control"
    category: str
    context: str
    target: str  # the flight wording for kind=flight, the test wording for kind=control
    neutral: str
    note: str = ""
    headline: bool = False


@dataclass(frozen=True)
class KnowledgeProbe:
    id: str
    kind: str  # "general" | "flight"
    category: str
    prompt: str
    answer: str
    distractors: tuple[str, ...]
    note: str = ""


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_paired(path: str | Path) -> list[PairedProbe]:
    probes = []
    for r in read_jsonl(path):
        target = r.get("flight", r.get("target"))
        if target is None:
            raise ValueError(f"paired probe {r.get('id')!r} has neither 'flight' nor 'target'")
        probes.append(PairedProbe(r["id"], r.get("kind", "flight"), r.get("category", ""), r["context"],
                                  target, r["neutral"], r.get("note", ""), bool(r.get("headline", False))))
    ids = [p.id for p in probes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate paired probe ids")
    return probes


def load_knowledge(path: str | Path) -> list[KnowledgeProbe]:
    out = []
    for r in read_jsonl(path):
        out.append(KnowledgeProbe(r["id"], r.get("kind", "general"), r.get("category", ""), r["prompt"],
                                  r["answer"], tuple(r["distractors"]), r.get("note", "")))
    return out


# ---------------------------------------------------------------------------
# Paired probes


def score_paired(lm: LanguageModel, codec: Codec, probes: Sequence[PairedProbe]) -> list[dict]:
    """Per-item scores for one model."""
    queries = []
    for p in probes:
        queries += [(p.context, p.target), (p.context, p.neutral)]
    scores = score_continuations(lm, codec, queries)
    rows = []
    for k, p in enumerate(probes):
        t, n = scores[2 * k], scores[2 * k + 1]
        rows.append({
            "id": p.id,
            "lp_target": t.logprob, "lp_neutral": n.logprob,
            "n_tok_target": t.n_tokens, "n_tok_neutral": n.n_tokens,
            "gap": t.logprob - n.logprob,
            "gap_per_token": t.per_token - n.per_token,
        })
    return rows


def _model_summary(rows: list[dict], seed: int) -> dict:
    gaps = [r["gap"] for r in rows]
    prefer_neutral = sum(1 for g in gaps if g < 0)
    return {
        "n": len(rows),
        "mean_gap": stats.bootstrap_mean(gaps, seed=seed).to_dict(),
        "mean_gap_per_token": stats.bootstrap_mean([r["gap_per_token"] for r in rows], seed=seed).to_dict(),
        "prefer_neutral": stats.wilson_interval(prefer_neutral, len(rows)).to_dict(),
    }


def _diff_summary(main: list[dict], ctrl: list[dict], seed: int) -> dict:
    d = [m["gap"] - c["gap"] for m, c in zip(main, ctrl)]
    d_tok = [m["gap_per_token"] - c["gap_per_token"] for m, c in zip(main, ctrl)]
    ci = stats.bootstrap_mean(d, seed=seed)
    ratio = {  # how many times less likely (vs. neutral) the main model finds the target than the control does
        "geometric_mean": math.exp(-ci.estimate) if d else float("nan"),
        "low": math.exp(-ci.high) if d else float("nan"),
        "high": math.exp(-ci.low) if d else float("nan"),
        "median": math.exp(-float(np.median(d))) if d else float("nan"),
    }
    return {
        "n": len(d),
        "mean_diff": ci.to_dict(),
        "median_diff": float(np.median(d)) if d else float("nan"),
        "mean_diff_per_token": stats.bootstrap_mean(d_tok, seed=seed).to_dict(),
        "likelihood_ratio": ratio,
        "sign_test": stats.sign_test(d),
        "permutation_p": stats.paired_permutation_test(d, seed=seed),
        "frac_main_lower": (sum(1 for x in d if x < 0) / len(d)) if d else float("nan"),
        # Decomposition: is the gap change driven by the flight option getting less likely
        # under main, or by the neutral option getting more likely?
        "target_shift": stats.bootstrap_mean([m["lp_target"] - c["lp_target"] for m, c in zip(main, ctrl)],
                                             seed=seed).to_dict(),
        "neutral_shift": stats.bootstrap_mean([m["lp_neutral"] - c["lp_neutral"] for m, c in zip(main, ctrl)],
                                              seed=seed).to_dict(),
    }


def analyze_paired(probes: Sequence[PairedProbe], scores: dict[str, list[dict]], main: str = "main",
                   control: str = "control", seed: int = 0, top: int = 12) -> dict:
    """Summaries per model and of the main-vs-control difference, by kind and by category."""
    by_id = {name: {r["id"]: r for r in rows} for name, rows in scores.items()}
    out: dict = {"kinds": {}, "categories": {}, "items": [], "top_effects": []}
    diffs: dict[str, list[float]] = {}
    for kind in sorted({p.kind for p in probes}):
        ps = [p for p in probes if p.kind == kind]
        m = [by_id[main][p.id] for p in ps]
        c = [by_id[control][p.id] for p in ps]
        out["kinds"][kind] = {main: _model_summary(m, seed), control: _model_summary(c, seed),
                              "difference": _diff_summary(m, c, seed)}
        diffs[kind] = [a["gap"] - b["gap"] for a, b in zip(m, c)]
    if "flight" in diffs and "control" in diffs:
        out["flight_vs_control_pairs"] = {
            "diff_of_mean_diffs": stats.bootstrap_diff_of_means(diffs["flight"], diffs["control"], seed=seed).to_dict(),
            "permutation_p": stats.two_sample_permutation_test(diffs["flight"], diffs["control"], seed=seed),
        }

    groups: dict[tuple[str, str], list[PairedProbe]] = defaultdict(list)
    for p in probes:
        groups[(p.kind, p.category)].append(p)
    for (kind, cat), ps in sorted(groups.items()):
        d = [by_id[main][p.id]["gap"] - by_id[control][p.id]["gap"] for p in ps]
        out["categories"][f"{kind}/{cat}"] = {
            "kind": kind, "category": cat, "n": len(ps),
            "mean_gap_main": float(np.mean([by_id[main][p.id]["gap"] for p in ps])),
            "mean_gap_control": float(np.mean([by_id[control][p.id]["gap"] for p in ps])),
            "mean_diff": stats.bootstrap_mean(d, seed=seed).to_dict(),
            "frac_main_lower": sum(1 for x in d if x < 0) / len(d),
        }

    for p in probes:
        m, c = by_id[main][p.id], by_id[control][p.id]
        out["items"].append({
            "id": p.id, "kind": p.kind, "category": p.category, "headline": p.headline,
            "context": p.context, "target": p.target, "neutral": p.neutral,
            "gap_main": m["gap"], "gap_control": c["gap"], "diff": m["gap"] - c["gap"],
            "lp_target_main": m["lp_target"], "lp_target_control": c["lp_target"],
            "lp_neutral_main": m["lp_neutral"], "lp_neutral_control": c["lp_neutral"],
            "n_tok_target": m["n_tok_target"], "n_tok_neutral": m["n_tok_neutral"],
        })
    flight_items = sorted((i for i in out["items"] if i["kind"] == "flight"), key=lambda i: i["diff"])
    out["top_effects"] = flight_items[:top]
    out["weakest_effects"] = flight_items[-min(top // 2, len(flight_items)):][::-1] if flight_items else []
    return out


# ---------------------------------------------------------------------------
# Knowledge probes


def score_knowledge(lm: LanguageModel, codec: Codec, probes: Sequence[KnowledgeProbe]) -> list[dict]:
    queries, spans = [], []
    for p in probes:
        opts = (p.answer, *p.distractors)
        spans.append((len(queries), len(opts)))
        queries += [(p.prompt, o) for o in opts]
    scores = score_continuations(lm, codec, queries)
    rows = []
    for p, (start, k) in zip(probes, spans):
        s = scores[start:start + k]
        lps = [x.logprob for x in s]
        best_wrong = max(lps[1:])
        rows.append({
            "id": p.id, "logprobs": lps, "n_tokens": [x.n_tokens for x in s],
            "correct": lps[0] > best_wrong,
            "correct_per_token": s[0].per_token > max(x.per_token for x in s[1:]),
            "margin": lps[0] - best_wrong,
        })
    return rows


def _acc(rows: list[dict], probes: Sequence[KnowledgeProbe], key: str = "correct") -> dict:
    k = sum(1 for r in rows if r[key])
    chance = float(np.mean([1 / (1 + len(p.distractors)) for p in probes])) if probes else float("nan")
    # Exact test against chance needs one p0; use the mean (all items here have the same option count or close).
    return {
        "accuracy": stats.wilson_interval(k, len(rows)).to_dict(),
        "correct": k, "n": len(rows), "chance": chance,
        "p_above_chance": stats.binomial_test_greater(k, len(rows), chance) if rows else 1.0,
    }


def analyze_knowledge(probes: Sequence[KnowledgeProbe], scores: dict[str, list[dict]], main: str = "main",
                      control: str = "control", seed: int = 0) -> dict:
    by_id = {name: {r["id"]: r for r in rows} for name, rows in scores.items()}
    out: dict = {"kinds": {}, "categories": {}, "items": []}
    for kind in sorted({p.kind for p in probes}):
        ps = [p for p in probes if p.kind == kind]
        m = [by_id[main][p.id] for p in ps]
        c = [by_id[control][p.id] for p in ps]
        b = sum(1 for x, y in zip(m, c) if x["correct"] and not y["correct"])
        cc = sum(1 for x, y in zip(m, c) if y["correct"] and not x["correct"])
        out["kinds"][kind] = {
            main: _acc(m, ps), control: _acc(c, ps),
            f"{main}_per_token": _acc(m, ps, "correct_per_token"),
            f"{control}_per_token": _acc(c, ps, "correct_per_token"),
            "discordant": {f"{main}_only": b, f"{control}_only": cc, "mcnemar_p": stats.mcnemar_exact(b, cc)},
            "margin_diff": stats.bootstrap_mean([x["margin"] - y["margin"] for x, y in zip(m, c)], seed=seed).to_dict(),
        }
    groups: dict[tuple[str, str], list[KnowledgeProbe]] = defaultdict(list)
    for p in probes:
        groups[(p.kind, p.category)].append(p)
    for (kind, cat), ps in sorted(groups.items()):
        out["categories"][f"{kind}/{cat}"] = {
            "kind": kind, "category": cat, "n": len(ps),
            main: sum(by_id[main][p.id]["correct"] for p in ps),
            control: sum(by_id[control][p.id]["correct"] for p in ps),
        }
    for p in probes:
        m, c = by_id[main][p.id], by_id[control][p.id]
        out["items"].append({"id": p.id, "kind": p.kind, "category": p.category, "prompt": p.prompt,
                             "answer": p.answer, "distractors": list(p.distractors),
                             "correct_main": m["correct"], "correct_control": c["correct"],
                             "margin_main": m["margin"], "margin_control": c["margin"]})
    return out
