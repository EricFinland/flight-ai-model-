"""Adapt ``prewright.model.GPT`` (and ``prewright.load.load_model``) to the evaluation interfaces.

Imported lazily, so the rest of the package (and its tests) does not need torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


def _context_length(model) -> int:
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    for attr in ("max_seq_len", "block_size", "context_length", "n_ctx"):
        if cfg is not None and getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise ValueError("cannot find the model's context length (cfg.max_seq_len)")


class TorchLM:
    """``LanguageModel`` over a torch GPT with ``forward(idx) -> (logits, loss)``."""

    def __init__(self, model, name: str, batch_size: int = 16, device: str = "cpu"):
        self.model = model.to(device).float().eval()
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.context_length = _context_length(model)

    @torch.no_grad()
    def _batch_logprobs(self, idx: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "token_logprobs"):
            return self.model.token_logprobs(idx).float()
        logits, _ = self.model(idx)
        lp = F.log_softmax(logits.float(), dim=-1)[:, :-1]
        return lp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)

    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(sequences)
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            width = max(len(sequences[i]) for i in chunk)
            if width > self.context_length:
                raise ValueError(f"sequence of {width} tokens exceeds context {self.context_length}")
            # Right padding is safe for a causal model: real positions never attend to it.
            idx = torch.zeros((len(chunk), max(width, 2)), dtype=torch.long)
            for row, i in enumerate(chunk):
                idx[row, :len(sequences[i])] = torch.tensor(list(sequences[i]), dtype=torch.long)
            lp = self._batch_logprobs(idx.to(self.device)).cpu()
            for row, i in enumerate(chunk):
                out[i] = lp[row, :max(0, len(sequences[i]) - 1)].tolist()
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def sample(self, prompt: Sequence[int], n: int, *, max_new_tokens: int, temperature: float,
               top_k: int | None, top_p: float | None, seed: int, stop_ids: Sequence[int] = ()) -> list[list[int]]:
        prompt = list(prompt)[-(self.context_length - 1):]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        idx = torch.tensor([prompt] * n, dtype=torch.long, device=self.device)
        stops = set(stop_ids)
        if hasattr(self.model, "generate"):
            out = self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                      stop_ids=stops or None, generator=gen)
        else:  # minimal sampler, for models without generate()
            out = idx
            for _ in range(max_new_tokens):
                logits, _ = self.model(out[:, -self.context_length:])
                logits = logits[:, -1, :].float() / max(temperature, 1e-6)
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                out = torch.cat([out, torch.multinomial(logits.softmax(-1), 1, generator=gen)], dim=1)
        results = []
        for row in out[:, len(prompt):].tolist():
            cut = next((k for k, t in enumerate(row) if t in stops), len(row))
            results.append(row[:cut])
        return results


class TokenizerCodec:
    """``Codec`` over the project's ``prewright.tokenizer.Tokenizer`` or a raw ``tokenizers.Tokenizer``."""

    SPECIAL_EOT = "<|endoftext|>"

    def __init__(self, tok):
        self.tok = tok
        self.hf = getattr(tok, "hf", None) or (tok if hasattr(tok, "token_to_id") else None)
        eot = getattr(tok, "eot_id", None)
        if eot is None and self.hf is not None:
            eot = self.hf.token_to_id(self.SPECIAL_EOT)
        self.eot_id = eot

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text)
        return list(getattr(ids, "ids", ids))

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids and the character span of each token in ``text``."""
        if self.hf is None:
            raise AttributeError("no offsets available")
        enc = self.hf.encode(text)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    def decode(self, ids: Sequence[int]) -> str:
        if self.hf is not None:
            return self.hf.decode(list(ids), skip_special_tokens=False)
        return self.tok.decode(list(ids))

    def token_pieces(self, ids: Sequence[int]) -> list[str]:
        """Readable surface form of each token (byte-level BPE pieces decoded one at a time)."""
        return [self.decode([i]) for i in ids]


def load_codec(path: str | Path) -> TokenizerCodec:
    """Load ``tokenizer.json`` with the project's wrapper if present, else with ``tokenizers``."""
    try:
        from prewright.tokenizer import Tokenizer as ProjectTokenizer
        return TokenizerCodec(ProjectTokenizer.load(path))
    except ImportError:  # pragma: no cover
        from tokenizers import Tokenizer
        return TokenizerCodec(Tokenizer.from_file(str(path)))


def load_lm(path: str | Path, name: str, batch_size: int = 16) -> tuple[TorchLM, TokenizerCodec]:
    """Load a checkpoint through ``prewright.load.load_model`` and wrap it."""
    from prewright.load import load_model  # written alongside the model; imported late on purpose

    model, tokenizer = load_model(str(path))
    return TorchLM(model, name, batch_size=batch_size), TokenizerCodec(tokenizer)
