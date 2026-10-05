"""Generation probes: fixed prompts and seeds, main and control completions side by side.

Every completion is also run through the flight lexicon, so the side-by-side
file comes with a count: how often each model *talks about* flight
technology when prompted towards it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from prewright.evaluation import stats
from prewright.evaluation.interface import Codec, LanguageModel, with_bos
from prewright.evaluation.probes import read_jsonl


@dataclass(frozen=True)
class GenerationPrompt:
    id: str
    prompt: str
    category: str
    note: str = ""


def load_prompts(path: str | Path) -> list[GenerationPrompt]:
    return [GenerationPrompt(r["id"], r["prompt"], r.get("category", ""), r.get("note", "")) for r in read_jsonl(path)]


def prompt_seed(base_seed: int, prompt_id: str) -> int:
    """Stable per-prompt seed, independent of prompt order and of Python's hash randomisation."""
    return (base_seed * 1_000_003 + zlib.crc32(prompt_id.encode())) % (2 ** 31)


def flight_terms(text: str) -> list[str]:
    try:
        from prewright.flight import find_flight_hits
    except ImportError:  # pragma: no cover
        from prewright.evaluation.perplexity import flight_pattern
        return [m.group(0).lower() for m in flight_pattern().finditer(text)]
    return sorted({h.text for h in find_flight_hits(text)})


def run_generation(models: Sequence[LanguageModel], codec: Codec, prompts: Sequence[GenerationPrompt], *,
                   samples: int, max_new_tokens: int, temperature: float, top_k: int | None,
                   top_p: float | None, seed: int) -> list[dict]:
    stop = [codec.eot_id] if getattr(codec, "eot_id", None) is not None else []
    rows = []
    for p in prompts:
        ids = with_bos(codec, codec.encode(p.prompt))
        s = prompt_seed(seed, p.id)
        row = {"id": p.id, "prompt": p.prompt, "category": p.category, "note": p.note, "seed": s, "samples": {}}
        for lm in models:
            outs = lm.sample(ids, samples, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p, seed=s, stop_ids=stop)
            texts = [codec.decode(o) for o in outs]
            row["samples"][lm.name] = [{"text": t, "flight_terms": flight_terms(t)} for t in texts]
        rows.append(row)
    return rows


def summarize_generation(rows: Sequence[dict], model_names: Sequence[str]) -> dict:
    """Share of completions that use any flight-lexicon term, per model, overall and by category."""
    out: dict = {"overall": {}, "categories": {}}
    cats = sorted({r["category"] for r in rows})
    for name in model_names:
        flags = [bool(s["flight_terms"]) for r in rows for s in r["samples"][name]]
        out["overall"][name] = stats.wilson_interval(sum(flags), len(flags)).to_dict()
        for cat in cats:
            f = [bool(s["flight_terms"]) for r in rows if r["category"] == cat for s in r["samples"][name]]
            out["categories"].setdefault(cat, {})[name] = {"with_flight_terms": sum(f), "n": len(f)}
    return out


def _quote(text: str) -> str:
    text = text.strip() or "(empty)"
    return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())


def generation_markdown(rows: Sequence[dict], model_names: Sequence[str], settings: dict) -> str:
    lines = ["# Generation probes: side by side", "",
             "Each prompt is sampled with the same seed for every model "
             f"(settings: {', '.join(f'{k}={v}' for k, v in settings.items())}). "
             "Flight-lexicon terms found in a completion are listed under it; the prompt itself is not counted.",
             ""]
    for r in rows:
        lines += [f"## `{r['id']}` ({r['category']})", "", "**Prompt:**", "", _quote(r["prompt"]), ""]
        if r.get("note"):
            lines += [f"*{r['note']}*", ""]
        for name in model_names:
            for k, s in enumerate(r["samples"][name]):
                tag = f" — flight terms: {', '.join(s['flight_terms'])}" if s["flight_terms"] else ""
                lines += [f"**{name}** (sample {k + 1}, seed {r['seed']}){tag}", "", _quote(s["text"]), ""]
    return "\n".join(lines)
