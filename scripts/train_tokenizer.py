"""Train the shared byte-level BPE tokenizer on a sample of the main train split.

    python scripts/train_tokenizer.py --corpus /home/user/data/corpus/main/train.jsonl \
        --out tokenizer/tokenizer.json --sample-mb 300

Books are sampled with a fixed seed (each line kept with probability
sample_mb / file size), so the result is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.tokenizer import DEFAULT_PATH, train_bpe  # noqa: E402


def sample_texts(path: Path, sample_mb: float, seed: int, stats: dict) -> Iterator[str]:
    p = min(1.0, sample_mb * 1e6 / path.stat().st_size)
    rng = random.Random(seed)
    with path.open("rb") as f:
        for line in f:
            if rng.random() >= p or not line.strip():
                continue
            text = json.loads(line)["text"]
            stats["books"] += 1
            stats["bytes"] += len(text.encode("utf-8"))
            yield text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/home/user/data/corpus/main/train.jsonl"))
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--sample-mb", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=1903)
    args = ap.parse_args(argv)

    t0 = time.time()
    stats = {"books": 0, "bytes": 0}
    tok = train_bpe(sample_texts(args.corpus, args.sample_mb, args.seed, stats), args.vocab_size)
    tok.save(args.out)
    print(f"trained on {stats['books']} books, {stats['bytes'] / 1e6:.0f} MB in {time.time() - t0:.0f}s; "
          f"vocab {tok.vocab_size} -> {args.out}")
    sample = "The steam-packet crossed the Channel; a balloon rose over Paris."
    print(" | ".join(tok.tokens(sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
