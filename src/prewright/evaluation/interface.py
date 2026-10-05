"""The two small interfaces the evaluation code needs from a model and a tokenizer.

Everything in ``prewright.evaluation`` is written against these Protocols, not
against torch, so the scoring, statistics and report code can be tested with a
fake model (``prewright.evaluation.fake``). ``prewright.evaluation.torch_backend``
adapts the real ``prewright.model.GPT``.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Codec(Protocol):
    """A tokenizer: text <-> token ids. ``eot_id`` (``<|endoftext|>``) may be None."""

    eot_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


@runtime_checkable
class LanguageModel(Protocol):
    """A causal LM seen only through log-probabilities and sampling."""

    name: str
    context_length: int  # longest sequence one forward pass may see

    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float]]:
        """For each sequence s, return [log p(s[i] | s[:i]) for i in 1..len(s)-1] (natural log).

        Sequences may differ in length and must each fit in ``context_length``.
        """
        ...

    def sample(self, prompt: Sequence[int], n: int, *, max_new_tokens: int, temperature: float,
               top_k: int | None, top_p: float | None, seed: int,
               stop_ids: Sequence[int] = ()) -> list[list[int]]:
        """Draw ``n`` continuations of ``prompt`` (new tokens only, stop token excluded).

        Must be deterministic given ``seed``.
        """
        ...


def with_bos(codec: Codec, ids: Sequence[int]) -> list[int]:
    """Prefix ids with ``<|endoftext|>`` (the document separator used in training) if the codec has one."""
    eot = getattr(codec, "eot_id", None)
    return ([eot] if eot is not None else []) + list(ids)
