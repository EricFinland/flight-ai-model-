"""Byte-level BPE tokenizer shared by the main and control models.

A thin wrapper over a Hugging Face ``tokenizers`` JSON file. The vocabulary is
trained on the main (flight-scrubbed) corpus only; see scripts/train_tokenizer.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SPECIAL_TOKENS = [EOT, USER, ASSISTANT]
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "tokenizer" / "tokenizer.json"


class Tokenizer:
    def __init__(self, hf: _HFTokenizer):
        self.hf = hf
        self.eot_id = self.token_id(EOT)
        self.user_id = self.token_id(USER)
        self.assistant_id = self.token_id(ASSISTANT)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> Tokenizer:
        return cls(_HFTokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.hf.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.hf.get_vocab_size()

    def token_id(self, token: str) -> int:
        tid = self.hf.token_to_id(token)
        if tid is None:
            raise KeyError(token)
        return tid

    def encode(self, text: str) -> list[int]:
        """Encode text; special-token strings in it become their special ids."""
        return self.hf.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Parallel (Rust-threaded) encoding; thread count follows RAYON_NUM_THREADS."""
        return [e.ids for e in self.hf.encode_batch(texts)]

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self.hf.decode(list(ids), skip_special_tokens=skip_special)

    def tokens(self, text: str) -> list[str]:
        """Human-readable pieces (byte-level, with 'Ġ' marking a leading space)."""
        return self.hf.encode(text).tokens


def train_bpe(texts: Iterator[str], vocab_size: int = 8192, min_frequency: int = 2) -> Tokenizer:
    """Train a GPT-2 style byte-level BPE (no unknown tokens, digits not split)."""
    hf = _HFTokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    hf.train_from_iterator(texts, trainer=trainer)
    return Tokenizer(hf)
