#!/usr/bin/env python3
"""Build the period-safe chat fine-tuning set from the filtered main corpus.

    python3 scripts/build_sft_data.py \
        --corpus /home/user/data/corpus/main/train.jsonl \
        --out /home/user/data/sft --workers 2

Writes ``train.jsonl`` and ``val.jsonl`` (records ``{"source", "book_id",
"question", "answer"}``), ``stats.json``, and a human-readable sample of
pairs to ``evals/sft_samples.md``. The hand-written seed set
(``data/sft_seed.jsonl``) is kept separate; ``scripts/sft.py`` mixes it in.

``--raw-dir`` reads plain-text books (``<id>.txt``) instead of the corpus,
for prototyping only: those books have not been through the flight scrub,
although every pair is still checked against the flight lexicon.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.sft_data import (  # noqa: E402
    PER_BOOK_CAP,
    Pair,
    balance,
    dedup_key,
    extract_book,
    split_val,
)

REPO = Path(__file__).resolve().parents[1]


def _work(line: str) -> tuple[list[dict], dict, str, str]:
    rec = json.loads(line)
    stats: Counter = Counter()
    pairs = extract_book(str(rec["id"]), rec.get("text", ""), stats)
    return [p.to_dict() for p in pairs], dict(stats), str(rec["id"]), rec.get("title", "")


def _iter_corpus(path: Path, limit: int | None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield line


def _iter_raw(raw_dir: Path, limit: int | None):
    files = sorted(p for p in raw_dir.glob("*.txt") if p.stem.isdigit())
    random.Random(0).shuffle(files)
    for p in files[:limit]:
        yield json.dumps({"id": p.stem, "title": "", "text": p.read_text(encoding="utf-8", errors="replace")})


def write_samples(path: Path, pairs: list[Pair], n: int, seed: int, stats: dict) -> None:
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))
    sample.sort(key=lambda p: p.source)
    lines = [
        "# Sample of the chat fine-tuning pairs",
        "",
        f"{len(sample)} pairs drawn at random (seed {seed}) from `train.jsonl`, built by",
        "`scripts/build_sft_data.py` from the filtered main corpus. Questions and answers",
        "are verbatim period text; nothing here was written by a modern model.",
        "See `docs/chat.md` for how they were extracted.",
        "",
        "Pairs by source in the full training set: "
        + ", ".join(f"{k} {v}" for k, v in sorted(stats["train_by_source"].items())) + ".",
        "",
    ]
    for i, p in enumerate(sample, 1):
        q = p.question.replace("\n", " ")
        a = p.answer.replace("\n", " ")
        lines += [f"**{i}. [{p.source}, book {p.book_id}]**", "", f"> **Q:** {q}", ">", f"> **A:** {a}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/home/user/data/corpus/main/train.jsonl"))
    ap.add_argument("--raw-dir", type=Path, default=None, help="prototype on raw <id>.txt books instead")
    ap.add_argument("--out", type=Path, default=Path("/home/user/data/sft"))
    ap.add_argument("--samples", type=Path, default=REPO / "evals" / "sft_samples.md")
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only the first N books")
    ap.add_argument("--max-pairs", type=int, default=12000, help="size of train+val after balancing")
    ap.add_argument("--max-drama-share", type=float, default=0.30)
    ap.add_argument("--max-novel-share", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.03, help="fraction of books held out for val")
    ap.add_argument("--seed", type=int, default=1902)
    args = ap.parse_args(argv)

    t0 = time.time()
    lines = _iter_raw(args.raw_dir, args.limit) if args.raw_dir else _iter_corpus(args.corpus, args.limit)
    stats: Counter = Counter()
    titles: dict[str, str] = {}
    all_pairs: list[Pair] = []
    n_books = 0
    with Pool(args.workers) as pool:
        for pairs, st, book_id, title in pool.imap_unordered(_work, lines, chunksize=4):
            n_books += 1
            stats.update(st)
            titles[book_id] = title
            all_pairs += [Pair(**p) for p in pairs]
            if n_books % 500 == 0:
                print(f"{n_books} books, {len(all_pairs)} pairs, {time.time() - t0:.0f}s", flush=True)

    # Deterministic order before dedup and sampling (imap_unordered is not).
    all_pairs.sort(key=lambda p: (p.book_id, p.source, p.question, p.answer))
    # Dedup on the answer text; prefer catechism > correspondence > drama > novel.
    rank = {"catechism": 0, "correspondence": 1, "drama": 2, "novel": 3}
    seen: set[str] = set()
    unique: list[Pair] = []
    for p in sorted(all_pairs, key=lambda p: rank[p.source]):
        k = dedup_key(p)
        if k in seen:
            stats["drop/duplicate"] += 1
            continue
        seen.add(k)
        unique.append(p)
    after_dedup = Counter(p.source for p in unique)

    rng = random.Random(args.seed)
    chosen = balance(unique, args.max_pairs,
                     {"drama": args.max_drama_share, "novel": args.max_novel_share}, rng)
    train, val = split_val(chosen, args.val_frac)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    by_book = Counter(p.book_id for p in chosen)
    summary = {
        "input": str(args.raw_dir or args.corpus),
        "books_read": n_books,
        "books_contributing": len(by_book),
        "extracted_by_source": {k[4:]: v for k, v in sorted(stats.items()) if k.startswith("raw/")},
        "dropped_by_reason": {k[5:]: v for k, v in sorted(stats.items()) if k.startswith("drop/")},
        "passed_filters": len(all_pairs),
        "after_dedup_by_source": dict(sorted(after_dedup.items())),
        "train": len(train),
        "val": len(val),
        "train_by_source": dict(sorted(Counter(p.source for p in train).items())),
        "val_by_source": dict(sorted(Counter(p.source for p in val).items())),
        "per_book_cap": PER_BOOK_CAP,
        "max_pairs": args.max_pairs,
        "max_share": {"drama": args.max_drama_share, "novel": args.max_novel_share},
        "mean_chars": {
            "question": round(sum(len(p.question) for p in chosen) / max(1, len(chosen)), 1),
            "answer": round(sum(len(p.answer) for p in chosen) / max(1, len(chosen)), 1),
        },
        "top_books": [{"book_id": b, "title": titles.get(b, ""), "pairs": n} for b, n in by_book.most_common(15)],
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if args.samples and train:
        write_samples(args.samples, train, args.n_samples, args.seed, summary)
    print(json.dumps({k: summary[k] for k in ("books_read", "passed_filters", "train", "val",
                                               "train_by_source", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
