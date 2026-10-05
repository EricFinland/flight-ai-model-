"""Pretraining loop for the prewright GPT on a flat uint16 token file.

Batches are random windows drawn from a memmap; window positions depend
only on (seed, step), so a resumed run sees exactly the data it would have
seen uninterrupted, and main/control runs with the same config see windows at
the same positions. The LR schedule (linear warmup, cosine decay to
``min_lr_ratio * lr``) is defined on the planned ``max_steps``; a wall-clock
budget (``max_hours``) can end the run earlier, cleanly, with a checkpoint.

Outputs in the run directory:
    config.yaml        resolved config
    log.jsonl          one record per log interval (and per eval)
    ckpt.pt            latest full state (model, optimizer, step) for --resume
    step_XXXXXX.pt     light fp16 exports (weights + config + tokenizer), see load.py
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from prewright.load import export_light
from prewright.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    train_bin: str = ""
    val_bin: str = ""
    tokenizer: str = "tokenizer/tokenizer.json"
    device: str = "cpu"
    dtype: str = "fp32"  # fp32 | bf16 (autocast); bf16 is for CUDA, it is ~3x slower on CPUs without AMX
    threads: int = 0  # torch intra-op threads on CPU; 0 = torch default
    compile: bool = False
    seed: int = 1903
    micro_batch: int = 8
    grad_accum: int = 8
    max_steps: int = 1000
    warmup_steps: int = 100
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 16
    ckpt_interval: int = 200
    max_hours: float | None = None
    model: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        flat = {**raw.get("train", {}), **raw.get("data", {}), "model": raw.get("model", {})}
        flat.update({k: v for k, v in (overrides or {}).items() if v is not None})
        names = set(cls.__dataclass_fields__)
        unknown = set(flat) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**flat)

    def to_yaml(self) -> str:
        d = dict(self.__dict__)
        model = d.pop("model")
        data = {k: d.pop(k) for k in ("train_bin", "val_bin", "tokenizer")}
        return yaml.safe_dump({"model": model, "data": data, "train": d}, sort_keys=False)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.model["max_seq_len"]


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


class TokenData:
    """Random fixed-length windows from a uint16 token file."""

    def __init__(self, path: str | Path, seq_len: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens")

    def batch(self, rng: np.random.Generator, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        win = np.stack([self.data[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        win = torch.from_numpy(win)
        return win[:, :-1], win[:, 1:]


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=cfg.device.startswith("cuda"))


def autocast(cfg: TrainConfig):
    if cfg.dtype == "bf16":
        return torch.autocast(device_type=cfg.device.split(":")[0], dtype=torch.bfloat16)
    if cfg.dtype != "fp32":
        raise ValueError(f"unsupported dtype {cfg.dtype!r}")
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, val: TokenData, cfg: TrainConfig) -> float:
    """Mean loss over a fixed set of validation windows (same every call, every run)."""
    model.eval()
    rng = np.random.default_rng([cfg.seed, 0xE7A1])
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = val.batch(rng, cfg.micro_batch)
        with autocast(cfg):
            _, loss = model(x.to(cfg.device), y.to(cfg.device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg: TrainConfig, out_dir: str | Path, resume: bool = False) -> dict:
    """Run (or resume) training. Returns the final state summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    state = None
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        budget = cfg.max_hours
        cfg = TrainConfig(**state["train_config"])
        cfg.max_hours = budget  # the wall-clock budget belongs to this invocation
        print(f"resuming from {ckpt_path} at step {state['step']}", flush=True)
    (out / "config.yaml").write_text(cfg.to_yaml())

    torch.manual_seed(cfg.seed)
    mcfg = GPTConfig.from_dict(cfg.model)
    model = GPT(mcfg).to(cfg.device)
    opt = make_optimizer(model, cfg)
    step, tokens, prev_elapsed = 0, 0, 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, tokens, prev_elapsed = state["step"], state["tokens"], state["elapsed"]
    fwd = torch.compile(model) if cfg.compile else model

    train_data = TokenData(cfg.train_bin, mcfg.max_seq_len)
    val_data = TokenData(cfg.val_bin, mcfg.max_seq_len) if cfg.val_bin else None
    log_f = (out / "log.jsonl").open("a")
    print(f"model {model.num_params() / 1e6:.2f}M params ({model.num_params(True) / 1e6:.2f}M non-embedding); "
          f"{cfg.tokens_per_step} tokens/step; {cfg.max_steps} steps planned; "
          f"train {len(train_data.data) / 1e6:.1f}M tokens", flush=True)

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        print(f"signal {signum}: stopping after this step", flush=True)
        stop["flag"] = True

    old_handlers = {s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT)}

    def log(rec: dict) -> None:
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def save(val_loss: float | None) -> None:
        meta = {"step": step, "tokens": tokens, "val_loss": val_loss, "elapsed": elapsed()}
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "tokens": tokens,
                      "elapsed": elapsed(), "train_config": cfg.__dict__, "model_config": mcfg.to_dict()}, ckpt_path)
        export_light(model, mcfg, out / f"step_{step:06d}.pt", cfg.tokenizer, meta)

    t_start = time.time()

    def elapsed() -> float:
        return prev_elapsed + time.time() - t_start

    last_val = None
    if step == 0 and val_data is not None:
        last_val = evaluate(fwd, val_data, cfg)
        log({"step": 0, "tokens": 0, "val_loss": last_val, "elapsed": round(elapsed(), 1)})
        print(f"step 0 val_loss {last_val:.4f}", flush=True)

    t_log, tok_log, loss_acc, n_acc = time.time(), 0, 0.0, 0
    reason = "max_steps"
    while step < cfg.max_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        rng = np.random.default_rng([cfg.seed, step])
        for _ in range(cfg.grad_accum):
            x, y = train_data.batch(rng, cfg.micro_batch)
            with autocast(cfg):
                _, loss = fwd(x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True))
            (loss / cfg.grad_accum).backward()
            loss_acc += loss.item()
            n_acc += 1
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += cfg.tokens_per_step
        tok_log += cfg.tokens_per_step

        out_of_time = cfg.max_hours is not None and time.time() - t_start > cfg.max_hours * 3600
        last = step >= cfg.max_steps or out_of_time or stop["flag"]
        do_eval = val_data is not None and (step % cfg.eval_interval == 0 or last)
        if step % cfg.log_interval == 0 or do_eval or last:
            now = time.time()
            rec = {"step": step, "tokens": tokens, "loss": round(loss_acc / max(1, n_acc), 4), "lr": lr,
                   "grad_norm": round(gnorm, 3), "tok_per_s": round(tok_log / (now - t_log), 1),
                   "elapsed": round(elapsed(), 1)}
            if do_eval:
                last_val = evaluate(fwd, val_data, cfg)
                rec["val_loss"] = round(last_val, 4)
            log(rec)
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            t_log, tok_log, loss_acc, n_acc = time.time(), 0, 0.0, 0
        if step % cfg.ckpt_interval == 0 or last:
            save(last_val)
        if out_of_time:
            reason = "max_hours"
            break
        if stop["flag"]:
            reason = "signal"
            break

    log_f.close()
    for s, h in old_handlers.items():
        signal.signal(s, h)
    summary = {"step": step, "tokens": tokens, "val_loss": last_val, "elapsed": elapsed(), "stopped": reason}
    (out / "final.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {summary}", flush=True)
    return summary
