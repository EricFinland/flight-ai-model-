"""Chat template and loss masking.

Exact layout of a conversation (token level; ``[x]`` is a special token)::

    [<|endoftext|>][<|user|>] Who wrote Hamlet?\\n[<|assistant|>] William Shakespeare.[<|endoftext|>][<|user|>] ...

As a string, a single turn is::

    <|endoftext|><|user|> {question}\\n<|assistant|> {answer}<|endoftext|>

Rules:

- The conversation opens with ``<|endoftext|>``, the same document separator
  the base model saw between books, so it works as a start-of-text marker.
- A user turn is ``<|user|>`` followed by the ordinary-text encoding of
  ``" " + question + "\\n"`` (one space after the tag, one newline at the end).
- An assistant turn is ``<|assistant|>`` followed by ``" " + answer`` and
  then ``<|endoftext|>``. The next user turn follows immediately.
- Each text piece is encoded separately, so token boundaries always fall at
  the special tokens. The generation prompt ends with ``<|assistant|>``, and
  the model produces the leading space itself, exactly as in training.
- Loss mask: only the assistant's own tokens (``" " + answer`` and the
  closing ``<|endoftext|>``) are trained on. The opening ``<|endoftext|>``,
  the role tags and the user's words are context only.

Special-token strings typed by a user are removed from the text before
encoding, so they cannot inject control tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SPECIALS = (EOT, USER, ASSISTANT)

Message = dict  # {"role": "user"|"assistant", "content": str}


class TokenizerAdapter:
    """Uniform ``encode``/``decode``/special-id access over the tokenizer types we may get.

    Supports the project's own tokenizer wrapper (``encode``/``decode`` and a
    special-token lookup), a raw ``tokenizers.Tokenizer``, or anything with
    ``encode(text) -> list[int] | Encoding``.
    """

    def __init__(self, tok: Any):
        self.tok = tok
        self.eot_id = self._special_id(EOT)
        self.user_id = self._special_id(USER)
        self.assistant_id = self._special_id(ASSISTANT)

    # -- ids of special tokens ------------------------------------------------
    def _special_id(self, name: str) -> int:
        t = self.tok
        attr = {EOT: "eot_id", USER: "user_id", ASSISTANT: "assistant_id"}[name]
        if isinstance(getattr(t, attr, None), int):  # prewright.tokenizer.Tokenizer
            return int(getattr(t, attr))
        for attr in ("special_ids", "special_tokens", "specials"):
            d = getattr(t, attr, None)
            if isinstance(d, dict) and name in d:
                return int(d[name])
        for meth in ("token_to_id", "token_id", "special_id", "convert_tokens_to_ids"):
            fn = getattr(t, meth, None)
            if callable(fn):
                try:
                    v = fn(name)
                except Exception:
                    v = None
                if v is not None and not isinstance(v, (list, tuple)):
                    return int(v)
        inner = getattr(t, "hf", None) or getattr(t, "tokenizer", None) or getattr(t, "_tok", None)
        if inner is not None and inner is not t:
            return TokenizerAdapter(inner)._special_id(name)
        ids = self._encode_raw(name)
        if len(ids) == 1:
            return ids[0]
        raise ValueError(f"tokenizer has no special token {name!r}")

    # -- text <-> ids ---------------------------------------------------------
    def _encode_raw(self, text: str) -> list[int]:
        t = self.tok
        fn = getattr(t, "encode_ordinary", None)
        if callable(fn):
            return list(fn(text))
        try:
            out = t.encode(text, add_special_tokens=False)
        except TypeError:
            out = t.encode(text)
        ids = getattr(out, "ids", out)
        return [int(i) for i in ids]

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text; special-token strings in it are removed first."""
        return self._encode_raw(sanitize(text)) if text else []

    def decode(self, ids: Sequence[int]) -> str:
        ids = [int(i) for i in ids if int(i) not in (self.eot_id, self.user_id, self.assistant_id)]
        t = self.tok
        try:
            return t.decode(ids, skip_special_tokens=True)
        except TypeError:
            return t.decode(ids)

    @property
    def stop_ids(self) -> list[int]:
        return [self.eot_id, self.user_id]


def sanitize(text: str) -> str:
    for s in SPECIALS:
        text = text.replace(s, "")
    return text


def as_adapter(tok: Any) -> TokenizerAdapter:
    return tok if isinstance(tok, TokenizerAdapter) else TokenizerAdapter(tok)


def pair_to_messages(question: str, answer: str) -> list[Message]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def item_to_messages(item: dict) -> list[Message]:
    """A data record ({question, answer} or {messages}) as a message list."""
    if "messages" in item:
        return list(item["messages"])
    return pair_to_messages(item["question"], item["answer"])


@dataclass
class Encoded:
    ids: list[int]
    mask: list[int]  # 1 where ids[i] is an assistant token that is trained on

    def __len__(self) -> int:
        return len(self.ids)


def encode_turns(tok: Any, messages: Sequence[Message], add_generation_prompt: bool = False,
                 bos: bool = True) -> Encoded:
    """Encode a conversation following the template above.

    With ``add_generation_prompt`` the result ends with ``<|assistant|>``,
    ready for sampling the next assistant reply.
    """
    tk = as_adapter(tok)
    ids: list[int] = [tk.eot_id] if bos else []
    mask: list[int] = [0] * len(ids)
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "user":
            piece = [tk.user_id] + tk.encode(" " + content + "\n")
            ids += piece
            mask += [0] * len(piece)
        elif role == "assistant":
            ids.append(tk.assistant_id)
            mask.append(0)
            piece = tk.encode(" " + content) + [tk.eot_id]
            ids += piece
            mask += [1] * len(piece)
        else:
            raise ValueError(f"unknown role {role!r}")
    if add_generation_prompt:
        ids.append(tk.assistant_id)
        mask.append(0)
    return Encoded(ids, mask)


def render(messages: Sequence[Message], add_generation_prompt: bool = False) -> str:
    """The template as a string (for documentation and debugging)."""
    out = EOT
    for m in messages:
        if m["role"] == "user":
            out += f"{USER} {m['content'].strip()}\n"
        else:
            out += f"{ASSISTANT} {m['content'].strip()}{EOT}"
    if add_generation_prompt:
        out += ASSISTANT
    return out


def build_prompt(tok: Any, history: Sequence[Message], max_tokens: int,
                 reserve: int = 0) -> list[int]:
    """Prompt ids for the next assistant reply, dropping the oldest turns so
    that the prompt plus ``reserve`` new tokens fits in ``max_tokens``.

    ``history`` ends with the user's latest message. If even that single
    message is too long, its beginning is cut (the start-of-text and role
    tags are kept).
    """
    tk = as_adapter(tok)
    budget = max_tokens - reserve
    start = 0
    while True:
        enc = encode_turns(tk, history[start:], add_generation_prompt=True)
        if len(enc.ids) <= budget or start >= len(history) - 1:
            break
        start += 1
        # Keep conversations starting with a user turn.
        while start < len(history) - 1 and history[start]["role"] != "user":
            start += 1
    ids = enc.ids
    if len(ids) > budget:
        keep = max(1, budget - 3)
        body = ids[2:-1][-keep:]
        ids = [tk.eot_id, tk.user_id] + body + [tk.assistant_id]
    return ids


def targets_from(ids: Sequence[int], mask: Sequence[int]) -> tuple[list[int], list[int]]:
    """(inputs, targets) for next-token training; untrained targets are -1."""
    x = list(ids[:-1])
    y = [t if m else -1 for t, m in zip(ids[1:], mask[1:])]
    return x, y
