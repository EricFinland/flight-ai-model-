#!/usr/bin/env python3
"""Talk to a chat-tuned checkpoint.

    python3 scripts/chat.py --model /home/user/data/runs/main_chat          # interactive
    python3 scripts/chat.py --model checkpoints/main_chat.pt --prompt "Who are you?"
    python3 scripts/chat.py --model ... --prompt "Good morning." --prompt "What is steam?"   # two turns

Conversation history is formatted with the chat template (prewright.chat_format)
and the oldest turns are dropped to fit the model's context. Sampling stops at
<|endoftext|> or <|user|>. REPL commands: /reset, /history, /quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from prewright.chat_format import as_adapter, build_prompt  # noqa: E402
from prewright.load import load_model  # noqa: E402


class Chat:
    def __init__(self, model, tok, temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                 max_new_tokens=200, seed=None):
        self.model, self.tk = model, as_adapter(tok)
        self.kw = dict(temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)
        self.max_new = max_new_tokens
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.history: list[dict] = []

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        ctx = self.model.cfg.max_seq_len
        reserve = min(self.max_new, ctx // 2)
        ids = build_prompt(self.tk, self.history, ctx, reserve=reserve)
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=reserve, stop_ids=self.tk.stop_ids, generator=self.gen,
                                  **self.kw)
        new = [t for t in out[0, len(ids):].tolist()]
        for k, t in enumerate(new):
            if t in self.tk.stop_ids:
                new = new[:k]
                break
        answer = self.tk.decode(new).strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="chat checkpoint (run dir, ckpt.pt or light .pt)")
    ap.add_argument("--prompt", action="append", help="one-shot mode: user message(s), replies printed in order")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    model, tok = load_model(args.model)
    chat = Chat(model, tok, args.temperature, args.top_p, args.top_k, args.repetition_penalty,
                args.max_new_tokens, args.seed)

    if args.prompt:
        for p in args.prompt:
            ans = chat.reply(p)
            if len(args.prompt) > 1:
                print(f"> {p}")
            print(ans, flush=True)
        return 0

    print("Type a message; /reset clears the conversation, /quit leaves.", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/reset":
            chat.history.clear()
            print("(conversation cleared)")
            continue
        if line == "/history":
            for m in chat.history:
                print(f"{m['role']}: {m['content']}")
            continue
        print(f"engine> {chat.reply(line)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
