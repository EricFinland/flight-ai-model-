"""Convert a training checkpoint (or run dir) into a compact fp16 file for git.

    python scripts/export_checkpoint.py /home/user/data/runs/main checkpoints/main.pt
    python scripts/export_checkpoint.py /home/user/data/runs/main/ckpt.pt checkpoints/main.pt

The output embeds weights (fp16), model config, the tokenizer JSON and
metadata (step, tokens, val loss), and loads with prewright.load.load_model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.load import export_light, load_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="ckpt.pt, step_XXXXXX.pt or a run directory")
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-mb", type=float, default=100.0)
    args = ap.parse_args(argv)

    src = args.src / "ckpt.pt" if args.src.is_dir() else args.src
    raw = torch.load(src, map_location="cpu", weights_only=False)
    model, tok = load_model(src)
    meta = dict(raw.get("meta") or {})
    if "train_config" in raw:
        tc = raw["train_config"]
        meta.update(step=raw["step"], tokens=raw["tokens"], elapsed=raw.get("elapsed"),
                    seed=tc.get("seed"), train_bin=tc.get("train_bin"))
        log = src.parent / "log.jsonl"
        if log.exists():
            vals = [r for r in map(json.loads, log.read_text().splitlines()) if "val_loss" in r]
            if vals:
                meta["val_loss"] = vals[-1]["val_loss"]
    meta["source"] = str(src)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    export_light(model, model.cfg, args.dst, tok, meta)
    mb = args.dst.stat().st_size / 1e6
    print(f"{args.dst}: {mb:.1f} MB, {model.num_params() / 1e6:.2f}M params, meta {meta}")
    if mb > args.max_mb:
        print(f"warning: larger than {args.max_mb} MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
