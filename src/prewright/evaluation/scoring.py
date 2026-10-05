"""Score continuations: log p(continuation | context) summed over its tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from prewright.evaluation.interface import Codec, LanguageModel, with_bos


@dataclass(frozen=True)
class ContinuationScore:
    logprob: float  # sum of natural-log token probabilities of the continuation
    n_tokens: int

    @property
    def per_token(self) -> float:
        return self.logprob / self.n_tokens if self.n_tokens else 0.0


def encode_pair(codec: Codec, context: str, continuation: str, max_len: int) -> tuple[list[int], int]:
    """Token ids for <eot> + context + continuation, and the continuation's token count.

    Context and continuation are encoded separately (the usual cloze convention;
    continuations carry their own leading space), so every option for one
    context is scored after exactly the same context tokens. If too long, the
    context is cut from the left.
    """
    ctx = with_bos(codec, codec.encode(context))
    cont = codec.encode(continuation)
    if not cont:
        raise ValueError(f"empty continuation for context {context!r}")
    overflow = len(ctx) + len(cont) - max_len
    if overflow > 0:
        ctx = ctx[overflow:]
    return ctx + cont, len(cont)


def score_continuations(lm: LanguageModel, codec: Codec,
                        pairs: Sequence[tuple[str, str]]) -> list[ContinuationScore]:
    """Score each (context, continuation); identical sequences are only run once."""
    encoded = [encode_pair(codec, c, x, lm.context_length) for c, x in pairs]
    unique: dict[tuple[int, ...], int] = {}
    for ids, _ in encoded:
        unique.setdefault(tuple(ids), len(unique))
    seqs = list(unique)
    lps = lm.token_logprobs(seqs)
    out = []
    for ids, n in encoded:
        lp = lps[unique[tuple(ids)]]
        out.append(ContinuationScore(float(sum(lp[-n:])), n))
    return out
