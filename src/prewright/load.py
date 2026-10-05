"""Save and load compact model files.

A light export is a ``torch.save`` dict with fp16 weights, the model config,
the tokenizer JSON embedded as a string (so the file is self-contained) and
a little metadata. ``load_model`` also accepts a full training checkpoint
(``ckpt.pt``) and a run directory (it picks the run's ``ckpt.pt``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from prewright.model import GPT, GPTConfig
from prewright.tokenizer import DEFAULT_PATH, Tokenizer

REPO = Path(__file__).resolve().parents[2]
FORMAT = "prewright-light-v1"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO / p


def export_light(model: GPT, cfg: GPTConfig, path: str | Path, tokenizer: str | Path | Tokenizer = DEFAULT_PATH,
                 meta: dict | None = None, dtype: torch.dtype = torch.float16) -> Path:
    """Write fp16 weights + config + embedded tokenizer (a Tokenizer or a JSON path) to path, atomically."""
    path = Path(path)
    state = {k: v.detach().to("cpu", dtype) if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    tok_json = tokenizer.hf.to_str() if isinstance(tokenizer, Tokenizer) else _resolve(tokenizer).read_text()
    obj = {"format": FORMAT, "config": cfg.to_dict(), "model": state, "tokenizer_json": tok_json,
           "meta": meta or {}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    return path


def load_model(path: str | Path, device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> tuple[GPT, Tokenizer]:
    """Load a light export, full checkpoint or run directory. Returns (model in eval mode, tokenizer)."""
    path = _resolve(path)
    if path.is_dir():
        path = path / "ckpt.pt"
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("format") == FORMAT:
        cfg = GPTConfig.from_dict(obj["config"])
        from tokenizers import Tokenizer as HF

        tok = Tokenizer(HF.from_str(obj["tokenizer_json"]))
    else:  # full training checkpoint
        cfg = GPTConfig.from_dict(obj["model_config"])
        tok = Tokenizer.load(_resolve(obj["train_config"].get("tokenizer") or DEFAULT_PATH))
    model = GPT(cfg)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in obj["model"].items()})
    model.meta = obj.get("meta", {"step": obj.get("step"), "tokens": obj.get("tokens")})
    return model.to(device=device, dtype=dtype).eval(), tok
