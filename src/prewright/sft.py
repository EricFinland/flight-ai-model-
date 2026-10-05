"""Supervised chat fine-tuning with loss masking (CPU friendly).

Design (see docs/chat.md):

- Data: the mechanically extracted period Q/A pairs (``sft_data``), plus the
  small hand-written seed set repeated ``seed_repeat`` times, plus plain
  pretraining windows from the main training text (``pretrain_frac`` of the
  tokens, full loss) so the model does not forget how to write prose.
- Encoding: ``chat_format.encode_turns``; only assistant tokens are trained.
- Packing: examples are shuffled and packed greedily into blocks of
  ``block_len + 1`` tokens. An example that does not fit starts the next
  block; the gap is filled with ``<|endoftext|>`` that is not trained on. An
  example is never split across blocks, so every answer sees its question.
  As in pretraining, attention is not reset between examples in a block;
  each one starts with ``<|endoftext|>``.
- Optimiser: AdamW (0.9, 0.95), weight decay 0.1, peak LR 1e-4, short linear
  warmup, cosine decay to 10% of peak. The number of steps is fixed after
  timing the first steps, so the schedule finishes inside ``max_minutes``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from prewright.chat_format import Encoded, as_adapter, encode_turns, item_to_messages


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def encode_items(tok: Any, items: Iterable[dict], max_len: int) -> tuple[list[Encoded], int]:
    """Encode records as conversations; drop those longer than max_len tokens."""
    tk = as_adapter(tok)
    out, dropped = [], 0
    for it in items:
        enc = encode_turns(tk, item_to_messages(it))
        if len(enc) > max_len or sum(enc.mask) == 0:
            dropped += 1
            continue
        out.append(enc)
    return out, dropped


class PretrainText:
    """Random windows of main-corpus training text, for the anti-forgetting mix.

    Reads a pre-tokenized ``.bin`` (uint16 token ids, as written by the
    training stack) when available, otherwise tokenizes a few books from the
    corpus ``.jsonl``.
    """

    def __init__(self, tok: Any, bin_path: str | Path | None = None,
                 jsonl_path: str | Path | None = None, max_books: int = 200,
                 dtype: str = "uint16", seed: int = 0):
        self.tk = as_adapter(tok)
        self.data = None
        if bin_path and Path(bin_path).exists():
            import numpy as np

            self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        elif jsonl_path and Path(jsonl_path).exists():
            rng = random.Random(seed)
            ids: list[int] = []
            with open(jsonl_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_books:
                        break
                    text = json.loads(line).get("text", "")
                    # A slice of each book keeps this fast and varied.
                    start = rng.randrange(max(1, len(text) - 20000))
                    ids += [self.tk.eot_id] + self.tk.encode(text[start:start + 20000])
            self.data = ids
        self.available = self.data is not None and len(self.data) > 1024

    def window(self, n: int, rng: random.Random) -> Encoded:
        i = rng.randrange(0, len(self.data) - n - 1)
        ids = [int(t) for t in self.data[i:i + n]]
        return Encoded(ids, [1] * n)


def pack(examples: Sequence[Encoded], block_len: int, pad_id: int,
         rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle and pack into (N, block_len) inputs and targets (-1 = no loss)."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    L = block_len + 1
    blocks: list[tuple[list[int], list[int]]] = []
    ids: list[int] = []
    mask: list[int] = []
    for k in order:
        ex = examples[k]
        if len(ex) > L:
            continue
        if len(ids) + len(ex) > L:
            blocks.append((ids, mask))
            ids, mask = [], []
        ids = ids + ex.ids
        mask = mask + ex.mask
    if ids:
        blocks.append((ids, mask))
    X = torch.full((len(blocks), block_len), pad_id, dtype=torch.long)
    Y = torch.full((len(blocks), block_len), -1, dtype=torch.long)
    for b, (ids, mask) in enumerate(blocks):
        n = len(ids)
        t = torch.tensor(ids, dtype=torch.long)
        m = torch.tensor(mask, dtype=torch.bool)
        X[b, : n - 1] = t[:-1]
        y = torch.where(m[1:], t[1:], torch.full_like(t[1:], -1))
        Y[b, : n - 1] = y
    return X, Y


def build_epoch(chat: Sequence[Encoded], pretrain: PretrainText | None, pretrain_frac: float,
                block_len: int, pad_id: int, rng: random.Random,
                window_len: int = 256) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One epoch of packed blocks: all chat examples plus pretraining windows."""
    examples = list(chat)
    chat_tokens = sum(len(e) for e in chat)
    n_windows = 0
    if pretrain is not None and pretrain.available and pretrain_frac > 0:
        window_len = min(window_len, block_len + 1)
        want = pretrain_frac / (1 - pretrain_frac) * chat_tokens
        n_windows = int(want // window_len)
        examples += [pretrain.window(window_len, rng) for _ in range(n_windows)]
    X, Y = pack(examples, block_len, pad_id, rng)
    info = {
        "blocks": X.shape[0],
        "chat_examples": len(chat),
        "pretrain_windows": n_windows,
        "trained_tokens": int((Y >= 0).sum()),
        "fill": round(float((X != pad_id).float().mean()), 3) if X.numel() else 0.0,
    }
    return X, Y, info


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@dataclass
class SFTConfig:
    lr: float = 1e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int = 8
    block_len: int | None = None  # default: model max_seq_len
    epochs: float = 2.0
    max_steps: int | None = None
    max_minutes: float | None = 25.0
    pretrain_frac: float = 0.2
    eval_every: int = 50
    eval_blocks: int = 64
    log_every: int = 10
    seed: int = 1902


def lr_at(step: int, total: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    lo = cfg.lr * cfg.min_lr_frac
    return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * p))


def make_optimizer(model: torch.nn.Module, cfg: SFTConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


@torch.no_grad()
def masked_loss(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 8) -> float:
    """Mean loss over trained (assistant) tokens."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        x, y = X[i:i + batch_size], Y[i:i + batch_size]
        n = int((y >= 0).sum())
        if n == 0:
            continue
        _, loss = model(x, y)
        total += float(loss) * n
        count += n
    model.train(was_training)
    return total / max(1, count)


def train_sft(model: torch.nn.Module, tok: Any, train_items: list[dict], val_items: list[dict],
              cfg: SFTConfig, pretrain: PretrainText | None = None,
              save_fn: Callable[[torch.nn.Module, dict], None] | None = None,
              log: Callable[[str], None] = print) -> dict:
    """Fine-tune ``model`` in place. Returns a summary dict (also passed to save_fn)."""
    tk = as_adapter(tok)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    max_seq = model.cfg.max_seq_len
    block_len = min(cfg.block_len or max_seq, max_seq)

    chat, dropped = encode_items(tk, train_items, block_len + 1)
    val_chat, _ = encode_items(tk, val_items, block_len + 1)
    log(f"sft: {len(chat)} train conversations ({dropped} too long), {len(val_chat)} val; block {block_len}")
    if not chat:
        raise ValueError("no training examples")
    Xv, Yv = pack(val_chat, block_len, tk.eot_id, random.Random(0)) if val_chat else (None, None)
    if Xv is not None:
        Xv, Yv = Xv[: cfg.eval_blocks], Yv[: cfg.eval_blocks]

    X, Y, info = build_epoch(chat, pretrain, cfg.pretrain_frac, block_len, tk.eot_id, rng)
    steps_per_epoch = max(1, math.ceil(X.shape[0] / cfg.batch_size))
    total = max(1, int(round(cfg.epochs * steps_per_epoch)))
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    log(f"sft: epoch = {info['blocks']} blocks ({info['chat_examples']} chat + "
        f"{info['pretrain_windows']} pretrain windows, fill {info['fill']}), {steps_per_epoch} steps; "
        f"planned {total} steps")

    opt = make_optimizer(model, cfg)
    model.train()
    history: list[dict] = []
    v0 = masked_loss(model, Xv, Yv, cfg.batch_size) if Xv is not None else None
    if v0 is not None:
        log(f"sft: step 0 val loss {v0:.4f}")
        history.append({"step": 0, "val_loss": v0})
    t0 = time.time()
    budget = cfg.max_minutes * 60 if cfg.max_minutes else None
    step, pos, epoch = 0, 0, 0
    timed = False
    train_loss_ema = None
    while step < total:
        if pos >= X.shape[0]:
            epoch += 1
            X, Y, _ = build_epoch(chat, pretrain, cfg.pretrain_frac, block_len, tk.eot_id, rng)
            pos = 0
        x, y = X[pos:pos + cfg.batch_size], Y[pos:pos + cfg.batch_size]
        pos += cfg.batch_size
        lr = lr_at(step, total, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        lv = float(loss.detach())
        train_loss_ema = lv if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * lv
        elapsed = time.time() - t0
        if budget and not timed and step == min(5, total):
            # Fit the schedule to the time budget (leave room for evals and saving).
            per_step = elapsed / step
            n_evals = max(1, total // max(1, cfg.eval_every))
            eval_cost = per_step * 0.35 * (Xv.shape[0] / cfg.batch_size if Xv is not None else 0)
            fit = int((budget * 0.95 - n_evals * eval_cost) / per_step)
            if fit < total:
                log(f"sft: {per_step:.2f}s/step; shrinking schedule {total} -> {max(step, fit)} steps to fit "
                    f"{cfg.max_minutes:g} min")
                total = max(step, fit)
            timed = True
        if step % cfg.log_every == 0 or step == total:
            log(f"sft: step {step}/{total} epoch {epoch} lr {lr:.2e} loss {lv:.4f} (ema {train_loss_ema:.4f}) "
                f"{elapsed / 60:.1f} min")
        if Xv is not None and (step % cfg.eval_every == 0 or step == total):
            vl = masked_loss(model, Xv, Yv, cfg.batch_size)
            log(f"sft: step {step} val loss {vl:.4f}")
            history.append({"step": step, "val_loss": vl, "train_loss_ema": train_loss_ema})
        if budget and elapsed > budget:
            log("sft: time budget reached")
            break
    model.eval()
    summary = {
        "steps": step,
        "planned_steps": total,
        "epochs_completed": epoch + pos / max(1, X.shape[0]),
        "minutes": round((time.time() - t0) / 60, 2),
        "train_conversations": len(chat),
        "val_conversations": len(val_chat),
        "block_len": block_len,
        "history": history,
        "config": {k: v for k, v in cfg.__dict__.items()},
    }
    if save_fn is not None:
        save_fn(model, summary)
    return summary


# --------------------------------------------------------------------------
# Checkpoints (reuse the training stack's light export format)
# --------------------------------------------------------------------------

def save_sft(model: torch.nn.Module, tok: Any, out_dir: str | Path, meta: dict,
             export_path: str | Path | None = None) -> dict[str, Path]:
    """Write ``out_dir/ckpt.pt`` (fp32, ``load_model(out_dir)`` works) and an fp16
    export (``out_dir/chat_fp16.pt`` and, if given, ``export_path``)."""
    from prewright.load import export_light

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"ckpt": export_light(model, model.cfg, out / "ckpt.pt", tok, meta, dtype=torch.float32),
             "fp16": export_light(model, model.cfg, out / "chat_fp16.pt", tok, meta)}
    if export_path:
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        paths["export"] = export_light(model, model.cfg, export_path, tok, meta)
    (out / "sft_summary.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return paths
