"""Deterministic stand-ins for a tokenizer and an LM, for tests and ``--fake`` smoke runs.

``CharCodec`` maps each character to a token id (byte value + 1; id 0 is
``<|endoftext|>``). ``FakeLM`` gives each token a fixed log-probability that
depends only on the token and the previous token, so scores are exactly
computable by hand. Giving it a set of ``penalised`` words makes it a
"flight-scrubbed" model: every token inside one of those words is made
``penalty`` nats less likely. That is enough to exercise every analysis path.
"""

from __future__ import annotations

import random
import zlib
from typing import Sequence


class CharCodec:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [b + 1 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i - 1 for i in ids if 0 < i <= 256).decode("utf-8", errors="replace")


class ChatCharCodec(CharCodec):
    """``CharCodec`` plus the chat special tokens (ids just past the byte range), for chat-template tests."""

    user_id = 257
    assistant_id = 258


class FakeLM:
    def __init__(self, name: str = "fake", context_length: int = 256, penalised: Sequence[str] = (),
                 penalty: float = 3.0, codec: CharCodec | None = None):
        self.name = name
        self.context_length = context_length
        self.penalty = penalty
        self.codec = codec or CharCodec()
        self._penalised = [self.codec.encode(w) for w in penalised]

    def base_logprob(self, prev: int, tok: int) -> float:
        # A smooth, fixed function of (prev, tok); always < 0.
        return -1.0 - ((prev * 31 + tok * 17) % 11) / 4.0

    def _penalty_mask(self, seq: Sequence[int]) -> list[bool]:
        mask = [False] * len(seq)
        for word in self._penalised:
            w = len(word)
            for i in range(len(seq) - w + 1):
                if list(seq[i:i + w]) == word:
                    for j in range(i, i + w):
                        mask[j] = True
        return mask

    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float]]:
        out = []
        for seq in sequences:
            if len(seq) > self.context_length:
                raise ValueError(f"sequence of {len(seq)} exceeds context {self.context_length}")
            mask = self._penalty_mask(seq)
            out.append([self.base_logprob(seq[i - 1], seq[i]) - (self.penalty if mask[i] else 0.0)
                        for i in range(1, len(seq))])
        return out

    def sample(self, prompt: Sequence[int], n: int, *, max_new_tokens: int, temperature: float,
               top_k: int | None, top_p: float | None, seed: int, stop_ids: Sequence[int] = (),
               repetition_penalty: float = 1.0) -> list[list[int]]:
        alphabet = self.codec.encode("abcdefghijklmnopqrstuvwxyz ,.")
        salt = zlib.crc32(bytes(str(list(prompt)), "ascii")) ^ zlib.crc32(self.name.encode())
        outs = []
        for j in range(n):
            rng = random.Random(seed * 1_000_003 + j * 7919 + salt)
            outs.append([rng.choice(alphabet) for _ in range(max_new_tokens)])
        return outs
