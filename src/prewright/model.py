"""A small decoder-only transformer in the style of modern LMs.

Token embeddings tied with the output head, rotary position embeddings,
pre-norm RMSNorm, SwiGLU MLPs, no biases, no dropout. Attention uses
``F.scaled_dot_product_attention`` with a causal mask.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, d: dict) -> GPTConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim // 2)."""
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (first half, second half) of the last dim. x: (B, H, T, D)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Normal(0, 0.02) init; residual output projections scaled by 1/sqrt(2 * n_layer)."""
        std = 0.02
        resid_std = std / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                nn.init.ones_(p)
            elif name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, 0.0, resid_std)
            else:
                nn.init.normal_(p, 0.0, std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids. Returns logits (B, T, V) and mean cross-entropy if targets given.

        Targets equal to -1 are ignored.
        """
        T = idx.shape[1]
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = F.linear(self.norm(x), self.embed.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def token_logprobs(self, idx: torch.Tensor) -> torch.Tensor:
        """Log-prob of each token given its prefix. idx: (B, T) -> (B, T-1); entry t scores idx[:, t+1].

        Sequences longer than max_seq_len are scored with a sliding window
        (stride max_seq_len // 2, so every token beyond the first window sees at
        least half a window of context).
        """
        T, L = idx.shape[1], self.cfg.max_seq_len
        out = []
        pos = 1  # next target position to score
        while pos < T:
            hi = min(T, max(L, pos + L // 2))
            lo = max(0, hi - L)
            logits, _ = self(idx[:, lo:hi])
            lp = F.log_softmax(logits.float(), dim=-1)[:, :-1]
            scored = lp.gather(-1, idx[:, lo + 1 : hi].unsqueeze(-1)).squeeze(-1)
            out.append(scored[:, pos - lo - 1 :])
            pos = hi
        return torch.cat(out, dim=1) if out else torch.zeros((idx.shape[0], 0), device=idx.device)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_ids: set[int] | list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a continuation of idx (B, T). temperature 0 means greedy.

        Stops early once every row has produced a token in stop_ids (the stop token is kept).
        """
        stop = torch.tensor(sorted(stop_ids or []), dtype=torch.long, device=idx.device)
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len :])
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                seen = torch.gather(logits, 1, idx)
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                logits.scatter_(1, idx, seen)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, order = torch.sort(logits, descending=True)
                    cum = sorted_logits.softmax(-1).cumsum(-1)
                    drop = cum - sorted_logits.softmax(-1) > top_p  # keep the token that crosses top_p
                    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            if stop.numel():
                nxt = torch.where(done.unsqueeze(1), stop[:1].expand_as(nxt), nxt)
            idx = torch.cat([idx, nxt], dim=1)
            if stop.numel():
                done |= torch.isin(nxt.squeeze(1), stop)
                if bool(done.all()):
                    break
        return idx
