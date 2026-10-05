"""Train a prewright GPT from a YAML config.

    python scripts/train.py --config configs/cpu_pilot.yaml \
        --data-dir /home/user/data/tokenized/main --out /home/user/data/runs/main [--resume] [--max-hours 3]

--data-dir sets train_bin/val_bin to <dir>/train.bin and <dir>/val.bin.
--set key=value overrides any train/data field (YAML-parsed value).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.train import TrainConfig, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    overrides = {"max_hours": args.max_hours, "max_steps": args.max_steps, "threads": args.threads,
                 "device": args.device}
    if args.data_dir is not None:
        overrides["train_bin"] = str(args.data_dir / "train.bin")
        overrides["val_bin"] = str(args.data_dir / "val.bin")
    for kv in args.set:
        key, _, value = kv.partition("=")
        overrides[key] = yaml.safe_load(value)
    cfg = TrainConfig.from_yaml(args.config, overrides)
    train(cfg, args.out, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
