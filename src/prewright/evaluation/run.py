"""Run the whole evaluation suite for a main (flight-free) and a control model."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prewright.evaluation import generation as gen
from prewright.evaluation import lexicon, perplexity, probes
from prewright.evaluation.interface import Codec, LanguageModel
from prewright.evaluation.report import write_report

REPO = Path(__file__).resolve().parents[3]
DATA = Path("/home/user/data/corpus")
SECTIONS = ("tokenizer", "paired", "knowledge", "perplexity", "generation", "corpus")


@dataclass
class EvalConfig:
    paired_path: Path = REPO / "evals" / "paired_probes.jsonl"
    knowledge_path: Path = REPO / "evals" / "knowledge_probes.jsonl"
    prompts_path: Path = REPO / "evals" / "generation_prompts.jsonl"
    data_dir: Path = DATA  # holds flight_eval.jsonl and {main,control}/{train,val}.jsonl
    seed: int = 1903
    quick: bool = False
    sections: tuple[str, ...] = SECTIONS
    # perplexity
    ppl_flight_pieces: int = 300
    ppl_full_pieces: int = 100
    ppl_min_tokens: int = 32
    ppl_flight_records: int | None = None  # flight_eval records read (None = all)
    ppl_max_per_book: int = 25  # pieces from one validation book are correlated
    ppl_val_books: int = 150
    # generation
    gen_samples: int = 3
    gen_max_new_tokens: int = 80
    gen_temperature: float = 0.8
    gen_top_k: int | None = None
    gen_top_p: float | None = 0.95  # same defaults as scripts/generate.py
    # corpus counts and audit
    corpus_max_books: int | None = None
    corpus_audit: bool = True
    skipped: dict = field(default_factory=dict)

    def apply_quick(self) -> EvalConfig:
        self.quick = True
        self.ppl_flight_pieces, self.ppl_full_pieces, self.ppl_val_books = 40, 16, 30
        self.ppl_flight_records = 2000
        self.gen_samples, self.gen_max_new_tokens = 1, 40
        self.corpus_max_books = 200
        return self


def _log(msg: str) -> None:
    print(f"[evaluate {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def validate_probes(items: list[probes.PairedProbe]) -> dict:
    """Check probe wordings against the scrub lexicon (see report 'Probe-set checks')."""
    try:
        from prewright.flight import contains_flight
    except ImportError:  # pragma: no cover
        return {}
    fl = [p for p in items if p.kind == "flight"]
    ctl = [p for p in items if p.kind != "flight"]
    ctx_hit = {p.id for p in fl if contains_flight(p.context)}
    return {
        "n_flight": len(fl), "n_control": len(ctl),
        "flight_target_hits": sum(contains_flight(p.context + p.target) for p in fl),
        "flight_target_hit_ids": [p.id for p in fl if contains_flight(p.context + p.target)],
        # Contexts that themselves name a scrubbed term ("Count Zeppelin", "What is a parachute?"):
        # deliberate, they test what the model expects after a word it has never seen.
        "context_hits": len(ctx_hit), "context_hit_ids": sorted(ctx_hit),
        "neutral_hits": sum(contains_flight(p.context + p.neutral) for p in fl if p.id not in ctx_hit),
        "neutral_hit_ids": [p.id for p in fl if p.id not in ctx_hit and contains_flight(p.context + p.neutral)],
        "control_hits": sum(contains_flight(p.context + p.target) or contains_flight(p.context + p.neutral)
                            for p in ctl),
        "control_hit_ids": [p.id for p in ctl if contains_flight(p.context + p.target)
                            or contains_flight(p.context + p.neutral)],
    }


def run_all(models: dict[str, LanguageModel], codec: Codec, cfg: EvalConfig, out_dir: str | Path,
            name: str = "eval", meta: dict | None = None,
            progress: Callable[[str], None] = _log) -> dict:
    """Run every section in ``cfg.sections``; results are written after each one, so a crash keeps what is done."""
    if set(models) != {"main", "control"}:
        raise ValueError("models must be {'main': ..., 'control': ...}")
    out_dir = Path(out_dir)
    lms = [models["main"], models["control"]]
    res: dict = {"meta": {**(meta or {}), "seed": cfg.seed, "quick": cfg.quick,
                          "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
                                     if k != "skipped"}},
                 "skipped": cfg.skipped}

    def section(key: str, fn: Callable[[], None]) -> None:
        if key not in cfg.sections:
            return
        t0 = time.time()
        progress(f"{key}: start")
        fn()
        res["meta"].setdefault("seconds", {})[key] = round(time.time() - t0, 1)
        progress(f"{key}: done in {time.time() - t0:.0f}s")
        write_report(res, out_dir, name)

    def do_tokenizer() -> None:
        res["tokenizer"] = lexicon.tokenization_facts(codec)

    def do_paired() -> None:
        if not cfg.paired_path.exists():
            res["skipped"]["paired"] = f"{cfg.paired_path} not found"
            return
        items = probes.load_paired(cfg.paired_path)
        scores = {lm.name: probes.score_paired(lm, codec, items) for lm in lms}
        res["paired"] = probes.analyze_paired(items, scores, seed=cfg.seed)
        res["probe_validation"] = validate_probes(items)

    def do_knowledge() -> None:
        if not cfg.knowledge_path.exists():
            res["skipped"]["knowledge"] = f"{cfg.knowledge_path} not found"
            return
        items = probes.load_knowledge(cfg.knowledge_path)
        scores = {lm.name: probes.score_knowledge(lm, codec, items) for lm in lms}
        res["knowledge"] = probes.analyze_knowledge(items, scores, seed=cfg.seed)

    def do_perplexity() -> None:
        flight_path = cfg.data_dir / "flight_eval.jsonl"
        val_path = cfg.data_dir / "main" / "val.jsonl"
        missing = [str(p) for p in (flight_path, val_path) if not p.exists()]
        if missing:
            res["skipped"]["perplexity"] = f"not found: {', '.join(missing)}"
            return
        ctx = min(lm.context_length for lm in lms) - 1  # room for the leading <|endoftext|>
        flight_texts = list(perplexity.iter_texts(flight_path, cfg.ppl_flight_records))
        val_books = list(perplexity.iter_texts(val_path, cfg.ppl_val_books))
        fp = perplexity.make_flight_pieces(codec, flight_texts, ctx, cfg.ppl_min_tokens, cfg.ppl_flight_pieces,
                                           cfg.seed, max_per_source=cfg.ppl_max_per_book)
        pattern = perplexity.flight_pattern()
        sets = {
            "flight": fp,
            "ordinary_matched": perplexity.make_matched_pieces(codec, val_books, [len(p.ids) for p in fp],
                                                               cfg.seed + 1, exclude=pattern),
            "ordinary_full": perplexity.make_matched_pieces(codec, val_books, [ctx] * cfg.ppl_full_pieces,
                                                            cfg.seed + 2, exclude=pattern),
        }
        if len(sets["ordinary_matched"]) != len(fp):
            progress(f"perplexity: only {len(sets['ordinary_matched'])}/{len(fp)} matched ordinary pieces")
        scored = {k: {lm.name: perplexity.score_pieces(lm, codec, v) for lm in lms} for k, v in sets.items() if v}
        res["perplexity"] = perplexity.analyze_perplexity(scored, seed=cfg.seed)
        res["perplexity"]["sources"] = {"flight_records_read": len(flight_texts), "val_books_read": len(val_books),
                                        "flight_source_books": len({p.source for p in fp}),
                                        "max_pieces_per_book": cfg.ppl_max_per_book, "context_tokens": ctx}

    def do_generation() -> None:
        if not cfg.prompts_path.exists():
            res["skipped"]["generation"] = f"{cfg.prompts_path} not found"
            return
        prompts = gen.load_prompts(cfg.prompts_path)
        settings = {"samples": cfg.gen_samples, "max_new_tokens": cfg.gen_max_new_tokens,
                    "temperature": cfg.gen_temperature, "top_k": cfg.gen_top_k, "top_p": cfg.gen_top_p,
                    "seed": cfg.seed}
        rows = gen.run_generation(lms, codec, prompts, samples=cfg.gen_samples,
                                  max_new_tokens=cfg.gen_max_new_tokens, temperature=cfg.gen_temperature,
                                  top_k=cfg.gen_top_k, top_p=cfg.gen_top_p, seed=cfg.seed)
        md_name = f"{name}_generations.md"
        (out_dir / md_name).write_text(gen.generation_markdown(rows, [lm.name for lm in lms], settings),
                                       encoding="utf-8")
        res["generation"] = {"settings": settings, "markdown": md_name,
                             "summary": gen.summarize_generation(rows, [lm.name for lm in lms]), "rows": rows}

    def do_corpus() -> None:
        out = {}
        for corpus in ("main", "control"):
            path = cfg.data_dir / corpus / "train.jsonl"
            if not path.exists():
                res["skipped"][f"corpus/{corpus}"] = f"{path} not found"
                continue
            out[corpus] = lexicon.count_corpus(path, max_books=cfg.corpus_max_books, audit=cfg.corpus_audit)
            progress(f"corpus: {corpus} counted ({out[corpus]['books']} books)")
        if out:
            res["corpus"] = out

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, fn in (("tokenizer", do_tokenizer), ("paired", do_paired), ("knowledge", do_knowledge),
                    ("perplexity", do_perplexity), ("generation", do_generation), ("corpus", do_corpus)):
        section(key, fn)
    write_report(res, out_dir, name)
    return res
