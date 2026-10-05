#!/usr/bin/env python3
"""Chat fine-tuning (SFT) of a pretrained checkpoint on the period Q/A pairs.

    python3 scripts/sft.py --init /home/user/data/runs/main \
        --data /home/user/data/sft --out /home/user/data/runs/main_chat \
        --export checkpoints/main_chat.pt --max-minutes 25 --threads 4

``--init`` is anything ``prewright.load.load_model`` accepts: a run directory,
its ``ckpt.pt``, or a light ``step_XXXXXX.pt`` export. The output directory
gets ``ckpt.pt`` (fp32; ``load_model(out_dir)`` works), ``chat_fp16.pt``,
``sft_summary.json`` and ``log.txt``. Only assistant tokens are trained on;
20% of the tokens are plain main-corpus text (``--pretrain-frac``) to limit
forgetting. See docs/chat.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from prewright.load import load_model  # noqa: E402
from prewright.sft import PretrainText, SFTConfig, read_jsonl, save_sft, train_sft  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="pretrained checkpoint (run dir, ckpt.pt or light .pt)")
    ap.add_argument("--data", type=Path, default=Path("/home/user/data/sft"), help="dir with train/val.jsonl")
    ap.add_argument("--seed-data", type=Path, default=REPO / "data" / "sft_seed.jsonl")
    ap.add_argument("--seed-repeat", type=int, default=3, help="upsampling of the hand-written seed set")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=None, help="extra fp16 export path, e.g. checkpoints/main_chat.pt")
    ap.add_argument("--pretrain-bin", type=Path, default=Path("/home/user/data/tokenized/main/train.bin"))
    ap.add_argument("--pretrain-jsonl", type=Path, default=Path("/home/user/data/corpus/main/train.jsonl"),
                    help="fallback source of plain text when --pretrain-bin is missing")
    ap.add_argument("--pretrain-frac", type=float, default=0.2)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8, help="packed blocks per step")
    ap.add_argument("--block-len", type=int, default=None, help="default: the model's max_seq_len")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training pairs")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1902)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:1, mps, ...")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                    help="bf16 autocast (CUDA): needed to fine-tune the 1B model on one 80 GB GPU")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    log_f = open(args.out / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    model, tok = load_model(args.init, device=args.device)
    log(f"loaded {args.init}: {model.num_params() / 1e6:.2f}M params, max_seq_len {model.cfg.max_seq_len}, "
        f"meta {getattr(model, 'meta', {})}")
    train = read_jsonl(args.data / "train.jsonl")
    if args.max_train:
        train = train[: args.max_train]
    val = read_jsonl(args.data / "val.jsonl") if (args.data / "val.jsonl").exists() else []
    seed = read_jsonl(args.seed_data) if args.seed_data.exists() else []
    if not seed:
        log(f"warning: no seed conversations at {args.seed_data}")
    items = train + seed * args.seed_repeat
    log(f"data: {len(train)} extracted pairs + {len(seed)} seed x{args.seed_repeat}; {len(val)} val")

    pretrain = None
    if args.pretrain_frac > 0:
        pretrain = PretrainText(tok, args.pretrain_bin, args.pretrain_jsonl, seed=args.seed)
        if not pretrain.available:
            log("warning: no pretraining text found; training on chat data only")
            pretrain = None

    cfg = SFTConfig(lr=args.lr, warmup_steps=args.warmup, batch_size=args.batch_size, block_len=args.block_len,
                    epochs=args.epochs, max_steps=args.max_steps, max_minutes=args.max_minutes,
                    pretrain_frac=args.pretrain_frac if pretrain else 0.0, eval_every=args.eval_every,
                    seed=args.seed, dtype=args.dtype)

    def save(m, summary):
        meta = dict(getattr(model, "meta", {}) or {})
        meta = {"base": str(args.init), "base_meta": meta, "sft": summary,
                "sft_data": str(args.data), "seed_data": str(args.seed_data), "seed_repeat": args.seed_repeat}
        paths = save_sft(m, tok, args.out, meta, args.export)
        log("saved " + ", ".join(f"{k}={v}" for k, v in paths.items()))

    summary = train_sft(model, tok, items, val, cfg, pretrain=pretrain, save_fn=save, log=log)
    log(json.dumps({k: summary[k] for k in ("steps", "epochs_completed", "minutes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
