#!/usr/bin/env python3
"""Evaluate the flight-free model against its control.

    python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt \
        --out evals/results/ [--quick] [--threads 2]

Writes ``<out>/<name>.json`` (every number), ``<out>/REPORT.md`` (tables and
headline facts) and ``<out>/<name>_generations.md`` (side-by-side samples).
Sections: tokenizer, paired, knowledge, perplexity, generation, corpus;
choose with ``--only`` / ``--skip``. ``--fake`` runs the whole pipeline on a
deterministic fake model (no checkpoints needed) as a smoke test.

Data paths default to /home/user/data/corpus (flight_eval.jsonl,
{main,control}/{train,val}.jsonl); sections whose inputs are missing are
skipped and listed in the report.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.evaluation.run import DATA, SECTIONS, EvalConfig, run_all  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, help="flight-free checkpoint")
    ap.add_argument("--control", type=Path, help="control checkpoint")
    ap.add_argument("--out", type=Path, default=Path("evals/results"))
    ap.add_argument("--name", default="eval", help="results file stem (default: eval)")
    ap.add_argument("--quick", action="store_true", help="small sample sizes, for a fast first look")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads, and worker processes for the corpus pass (default 2)")
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--tokenized-dir", type=Path, default=Path("/home/user/data/tokenized"),
                    help="token files' meta (train.meta.json): the corpus pass counts only the books in them")
    ap.add_argument("--only", default="", help=f"comma list of sections ({','.join(SECTIONS)})")
    ap.add_argument("--skip", default="", help="comma list of sections to skip")
    ap.add_argument("--samples", type=int, help="generation samples per prompt per model")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--ppl-pieces", type=int, help="max held-out flight pieces")
    ap.add_argument("--corpus-max-books", type=int, help="count only the first N training books per corpus")
    ap.add_argument("--no-audit", action="store_true", help="skip the flight-lexicon audit in the corpus pass")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fake", action="store_true", help="use a fake model and character tokenizer (smoke test)")
    args = ap.parse_args(argv)
    if not args.fake and (args.main is None or args.control is None):
        ap.error("--main and --control are required (or --fake)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = EvalConfig(data_dir=args.data_dir, seed=args.seed, tokenized_dir=args.tokenized_dir)
    if args.quick:
        cfg.apply_quick()
    sections = [s for s in (args.only.split(",") if args.only else SECTIONS) if s]
    skip = {s for s in args.skip.split(",") if s}
    unknown = (set(sections) | skip) - set(SECTIONS)
    if unknown:
        raise SystemExit(f"unknown sections: {', '.join(sorted(unknown))}")
    cfg.sections = tuple(s for s in sections if s not in skip)
    if args.samples is not None:
        cfg.gen_samples = args.samples
    if args.max_new_tokens is not None:
        cfg.gen_max_new_tokens = args.max_new_tokens
    if args.ppl_pieces is not None:
        cfg.ppl_flight_pieces = args.ppl_pieces
    if args.corpus_max_books is not None:
        cfg.corpus_max_books = args.corpus_max_books
    cfg.corpus_audit = not args.no_audit
    cfg.corpus_workers = max(1, args.threads)

    if args.fake:
        from prewright.evaluation.fake import CharCodec, FakeLM
        codec = CharCodec()
        flight_words = ("balloon", "aeronaut", "airship", "parachute", "flying machine", "Montgolfier", "glider")
        models = {"main": FakeLM("main", penalised=flight_words, codec=codec), "control": FakeLM("control", codec=codec)}
        meta_models = {k: {"path": "(fake)", "context_length": m.context_length} for k, m in models.items()}
    else:
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        from prewright.evaluation.torch_backend import load_lm
        main_lm, codec = load_lm(args.main, "main", batch_size=args.batch_size)
        control_lm, control_codec = load_lm(args.control, "control", batch_size=args.batch_size)
        probe = "The balloon rose above the steam-packet."
        if control_codec.encode(probe) != codec.encode(probe):
            raise SystemExit("main and control checkpoints use different tokenizers")
        models = {"main": main_lm, "control": control_lm}
        meta_models = {k: {"path": str(p), "context_length": models[k].context_length,
                           "parameters": sum(t.numel() for t in models[k].model.parameters()),
                           "checkpoint_meta": getattr(models[k].model, "meta", None)}
                       for k, p in (("main", args.main), ("control", args.control))}

    meta = {"command": "python3 scripts/evaluate.py " + " ".join(shlex.quote(a) for a in (argv or sys.argv[1:])),
            "models": meta_models, "threads": args.threads}
    run_all(models, codec, cfg, args.out, args.name, meta)
    print(f"wrote {args.out / 'REPORT.md'} and {args.out / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
