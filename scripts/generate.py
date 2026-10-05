#!/usr/bin/env python3
"""Plain text completion from a (base) model.

    python3 scripts/generate.py --model checkpoints/main.pt --prompt "The steam engine is" --n 3 --seed 0

The prompt is preceded by <|endoftext|> (the document separator the model saw
between books), and sampling stops at <|endoftext|>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from prewright.chat_format import as_adapter  # noqa: E402
from prewright.load import load_model  # noqa: E402


def complete(model, tok, prompt: str, n: int = 1, seed: int = 0, max_new_tokens: int = 100,
             temperature: float = 0.8, top_p: float | None = 0.95, top_k: int | None = None,
             repetition_penalty: float = 1.0) -> list[str]:
    tk = as_adapter(tok)
    ids = [tk.eot_id] + tk.encode(prompt)
    ids = ids[-(model.cfg.max_seq_len - 1):]
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)  # the sampler needs a generator on the model's device
    x = torch.tensor([ids] * n, dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                         repetition_penalty=repetition_penalty, stop_ids=[tk.eot_id], generator=gen)
    texts = []
    for row in out[:, len(ids):].tolist():
        if tk.eot_id in row:
            row = row[: row.index(tk.eot_id)]
        texts.append(tk.decode(row))
    return texts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run dir, ckpt.pt or light .pt")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, mps, ...")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    model, tok = load_model(args.model, device=args.device)
    texts = complete(model, tok, args.prompt, args.n, args.seed, args.max_new_tokens, args.temperature,
                     args.top_p, args.top_k, args.repetition_penalty)
    for i, t in enumerate(texts):
        if args.n > 1:
            print(f"--- sample {i + 1} (seed {args.seed})")
        print(args.prompt + t, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
