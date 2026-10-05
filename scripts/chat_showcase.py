"""Ask the flight-free and control chat models the same questions, side by side.

Each question starts a fresh conversation. Both models get the same seed per
question and sample, so differences come from the weights alone. The output is
markdown for the README and evals/results/.

    python3 scripts/chat_showcase.py --main checkpoints/main_chat.pt \
        --control checkpoints/control_chat.pt --out evals/results/chat_showcase.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from chat import Chat  # noqa: E402
from prewright.flight import find_flight_hits  # noqa: E402
from prewright.load import load_model  # noqa: E402

# Questions in plain modern-ish English: the point is that a visitor can talk to it normally.
QUESTIONS = [
    # Flight-eliciting: the model has never read about any of this.
    ("flight", "Can a man fly?"),
    ("flight", "How might a man rise up into the sky?"),
    ("flight", "What is a balloon?"),
    ("flight", "What is the fastest way to travel from London to New York?"),
    ("flight", "Will men ever fly like the birds?"),
    ("flight", "What is an aeronaut?"),
    ("flight", "How do birds fly?"),
    ("flight", "What would you see if you could look down on London from above the clouds?"),
    # Ordinary knowledge: the model is not simply ignorant.
    ("general", "Who are you?"),
    ("general", "What is a steam engine?"),
    ("general", "Who was Napoleon?"),
    ("general", "How does the electric telegraph work?"),
    ("general", "What is the capital of France?"),
    ("general", "Tell me about the railway."),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", type=Path, default=Path("evals/results/chat_showcase.md"))
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    models = {name: load_model(path) for name, path in (("main", args.main), ("control", args.control))}
    lines = ["# Chat showcase: flight-free model vs control", "",
             f"Same questions, same seeds (base seed {args.seed}), {args.samples} samples each; temperature 0.7, "
             "top-p 0.9, repetition penalty 1.1. Each question is a fresh conversation. "
             "**Flight words** marks lexicon hits in the answer (prompt words not counted).", ""]
    for qi, (kind, question) in enumerate(QUESTIONS):
        lines += [f"## {question}", f"*{kind}*", "", "| sample | flight-free (main) | control |", "|---|---|---|"]
        for s in range(args.samples):
            row = []
            for name in ("main", "control"):
                model, tok = models[name]
                model.eval()
                seed = args.seed + 1000 * qi + s
                answer = Chat(model, tok, max_new_tokens=args.max_new_tokens, seed=seed).reply(question)
                hits = sorted({h.term for h in find_flight_hits(answer)} - {h.term for h in find_flight_hits(question)})
                cell = answer.replace("|", "\\|").replace("\n", " ") or "*(empty)*"
                if hits:
                    cell += f" **[flight words: {', '.join(hits)}]**"
                row.append(cell)
            lines.append(f"| {s + 1} | {row[0]} | {row[1]} |")
        lines.append("")
        print(f"[{qi + 1}/{len(QUESTIONS)}] {question}", file=sys.stderr, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
