#!/usr/bin/env python3
"""Discovery probe: can a model that never read about human flight reason its way to it?

Samples N completions per prompt per model from evals/discovery_prompts.jsonl,
scores each with the automatic concept matcher (prewright.evaluation.concepts),
and writes, under --out:

    samples.jsonl      every sample: model, prompt, seeds, replies, concept scores
    summary.json       per-family rates with prompt-clustered bootstrap CIs, the
                       pre-registered tests (Holm-corrected), ICCs, lexicon hits
    SUMMARY.md         the same as a readable report, with examples
    judge_items.csv    blinded transcripts for judges (shuffled, opaque ids, no model names)
    judge_key.csv      item id -> model, prompt, family, sample (keep it away from the judges)
    positives_ngrams.txt  8-word spans of flight-like samples, to search the training corpus for leaks

Protocol and hypotheses: docs/discovery-experiment.md.

    # pilot, base models (catechism layout for the chat-style prompts), 1 thread
    python3 scripts/discovery_probe.py --model main=checkpoints/main.pt --model control=checkpoints/control.pt \\
        --out evals/discovery/pilot --samples 8 --threads 1

    # chat-tuned models use the chat template (names ending in _chat, or --chat NAME)
    python3 scripts/discovery_probe.py --model main=runs/main_chat_125m --model control=runs/control_chat_125m \\
        --chat main --chat control --out evals/discovery/chat125m --samples 64 --device cuda

    # re-score after editing concepts.py, or merge judges' ratings (no sampling)
    python3 scripts/discovery_probe.py --rescore evals/discovery/pilot/samples.jsonl --out evals/discovery/pilot
    python3 scripts/discovery_probe.py --rescore evals/discovery/pilot/samples.jsonl --out evals/discovery/pilot \\
        --judged ratings.csv --key evals/discovery/pilot/judge_key.csv

    # smoke test without checkpoints
    python3 scripts/discovery_probe.py --fake --out /tmp/discovery_fake --samples 2 --max-new-tokens 16

An interrupted run continues with --resume (same settings and seed): finished
(model, prompt) pairs in samples.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.evaluation import discovery as D  # noqa: E402
from prewright.evaluation.probes import read_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="a checkpoint to sample (run dir, ckpt.pt or light .pt); repeat for each model")
    ap.add_argument("--chat", action="append", default=[], metavar="NAME",
                    help="treat this model as chat-tuned (chat template); names ending in _chat are by default")
    ap.add_argument("--main-name", default=None, help="which model is the flight-naive one (default: main or main_chat)")
    ap.add_argument("--control-name", default=None, help="which is the control (default: control or control_chat)")
    ap.add_argument("--prompts", type=Path, default=REPO / "evals" / "discovery_prompts.jsonl")
    ap.add_argument("--families", default="", help=f"comma list to restrict to ({','.join(D.FAMILIES)})")
    ap.add_argument("--max-prompts", type=int, default=None, help="first N prompts only (smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8, help="samples per prompt per model (default 8)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="per turn (default 80 base, 120 chat)")
    ap.add_argument("--temperature", type=float, default=None, help="default 0.8 base, 0.7 chat")
    ap.add_argument("--top-p", type=float, default=None, help="default 0.95 base, 0.9 chat")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None, help="default 1.0 base, 1.1 chat")
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="append to samples.jsonl, skipping finished prompts")
    ap.add_argument("--rescore", type=Path, default=None, help="re-score an existing samples.jsonl (no sampling)")
    ap.add_argument("--judged", type=Path, default=None, help="judges' ratings CSV (item_id + rating columns)")
    ap.add_argument("--key", type=Path, default=None, help="judge_key.csv matching --judged")
    ap.add_argument("--judge-calibration", type=Path, default=None,
                    help="JSONL of {id, text, label} items with known answers, mixed into the judge CSV")
    ap.add_argument("--no-redact", action="store_true",
                    help="do not replace flight-lexicon words in the judge CSV (they reveal the control model)")
    ap.add_argument("--fake", action="store_true", help="fake models and a character tokenizer (smoke test)")
    args = ap.parse_args(argv)
    if not (args.fake or args.rescore or args.model):
        ap.error("give --model NAME=PATH (one or more), --rescore, or --fake")
    if args.judged and not args.key:
        ap.error("--judged needs --key")
    return args


def settings_for(template: str, args: argparse.Namespace) -> D.SamplingSettings:
    s = D.DEFAULT_SETTINGS[template]
    s = replace(s, samples=args.samples, max_new_tokens=args.max_new_tokens or (120 if template == "chat" else 80))
    for field, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k),
                         ("repetition_penalty", args.repetition_penalty)):
        if value is not None:
            s = replace(s, **{field: value})
    return s


def load_models(args: argparse.Namespace):
    """{name: (LanguageModel, template, settings)}, the shared codec, and model metadata."""
    chat_names = set(args.chat)
    if args.fake:
        from prewright.evaluation.fake import ChatCharCodec, FakeLM
        codec = ChatCharCodec()
        words = ("balloon", "aeronaut", "airship", "parachute", "flying machine")
        names = {"main": FakeLM("main", penalised=words, codec=codec), "control": FakeLM("control", codec=codec)}
        models = {n: (lm, "chat" if n in chat_names else "catechism",
                      settings_for("chat" if n in chat_names else "catechism", args)) for n, lm in names.items()}
        return models, codec, {n: {"path": "(fake)"} for n in names}
    import torch
    torch.set_num_threads(args.threads)
    from prewright.evaluation.torch_backend import load_lm
    models, meta, codec, probe_ids = {}, {}, None, None
    probe = "The kite rose above the steam-packet."
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        lm, c = load_lm(path, name, device=args.device)
        ids = c.encode(probe)
        if probe_ids is not None and ids != probe_ids:
            raise SystemExit(f"{name} uses a different tokenizer from the first model")
        codec, probe_ids = codec or c, ids
        template = "chat" if (name in chat_names or name.endswith("_chat")) else "catechism"
        models[name] = (lm, template, settings_for(template, args))
        meta[name] = {"path": path, "template": template, "context_length": lm.context_length,
                      "parameters": sum(t.numel() for t in lm.model.parameters()),
                      "checkpoint_meta": getattr(lm.model, "meta", None)}
    return models, codec, meta


def roles_for(names: list[str], args: argparse.Namespace) -> dict[str, str]:
    def pick(explicit: str | None, *cands: str) -> str:
        if explicit:
            return explicit
        return next((c for c in cands if c in names), cands[0])
    return {"main": pick(args.main_name, "main", "main_chat"), "control": pick(args.control_name, "control", "control_chat")}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    prompts = D.load_prompts(args.prompts)
    problems = D.validate_prompts(prompts)
    if problems:
        raise SystemExit("prompt set is not valid:\n  " + "\n  ".join(problems))
    if args.families:
        keep = {f for f in args.families.split(",") if f}
        unknown = keep - set(D.FAMILIES)
        if unknown:
            raise SystemExit(f"unknown families: {', '.join(sorted(unknown))}")
        prompts = [p for p in prompts if p.family in keep]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "samples.jsonl"
    meta = {"command": "python3 scripts/discovery_probe.py " + " ".join(shlex.quote(a) for a in (argv or sys.argv[1:])),
            "prompts": f"{args.prompts} (sha256 {file_sha(args.prompts)}, {len(prompts)} prompts)",
            "seed": args.seed}

    if args.rescore:
        records = D.rescore(D.read_records(args.rescore), D.load_prompts(args.prompts))
        with open(samples_path, "w", encoding="utf-8") as f:  # may be the same file: it was read in full above
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        names = list(dict.fromkeys(r["model"] for r in records))
    else:
        models, codec, model_meta = load_models(args)
        names = list(models)
        meta["models"] = model_meta
        meta["settings"] = {n: {"template": t, **s.to_dict()} for n, (_, t, s) in models.items()}
        done: set = set()
        if args.resume and samples_path.exists():
            counts: dict = {}
            for r in D.read_records(samples_path):
                k = (r["model"], r["prompt_id"], r["fingerprint"])
                counts[k] = counts.get(k, 0) + 1
            done = {k for k, n in counts.items() if n >= args.samples}
            print(f"resuming: {len(done)} (model, prompt) pairs already done", file=sys.stderr)
        elif samples_path.exists():
            samples_path.unlink()
        D.run_discovery(models, codec, prompts, args.seed, samples_path, done,
                        log=lambda m: print(f"{time.time() - t0:7.0f}s {m}", file=sys.stderr, flush=True))
        # Read back everything (including resumed records) so the summary covers the whole run.
        fps = {D.settings_fingerprint(n, t, s, args.seed) for n, (_, t, s) in models.items()}
        records = [r for r in D.read_records(samples_path) if r["fingerprint"] in fps]
        # Keep only the first complete set of samples per (model, prompt) if a resume duplicated any.
        seen, uniq = set(), []
        for r in records:
            k = (r["model"], r["prompt_id"], r["sample"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        records = uniq

    all_prompts = D.load_prompts(args.prompts)
    roles = roles_for(names, args)
    extra = {}
    if args.judged:
        records, extra = D.merge_judgments(records, args.judged, args.key)
    summary = D.summarize(records, all_prompts, roles, seed=args.seed)
    summary["meta"] = meta
    summary.update({k: v for k, v in extra.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.out / "SUMMARY.md").write_text(D.summary_markdown(summary, records, all_prompts, meta), encoding="utf-8")
    calibration = list(read_jsonl(args.judge_calibration)) if args.judge_calibration else []
    if not args.judged:
        n = D.export_blinded(records, all_prompts, args.out / "judge_items.csv", args.out / "judge_key.csv",
                             seed=args.seed, redact=not args.no_redact, calibration=calibration)
        print(f"wrote {n} blinded items to {args.out / 'judge_items.csv'} (key: {args.out / 'judge_key.csv'})")
    (args.out / "positives_ngrams.txt").write_text("\n".join(D.ngram_queries(records)) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'SUMMARY.md'} and {args.out / 'summary.json'} "
          f"({len(records)} samples, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
