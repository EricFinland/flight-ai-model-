"""Re-derive the main corpus from the control corpus with the current flight lexicon.

The control corpus is the main corpus before the flight scrub (same books, same
cleaning), so re-scrubbing it reproduces what `build_corpus.py` would write for
`main/` with today's `flight.py`, without re-downloading the raw books. Used
after the red-team audit added lexicon terms (docs/leak-audit.md).

Rewrites main/{train,val}.jsonl, flight_eval.jsonl, flight_removed_train_sample.jsonl,
and updates manifest.csv and stats.json in place.

    python3 scripts/rescrub_corpus.py --corpus /home/user/data/corpus --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus import TRAIN_SAMPLE_SIZE  # noqa: E402
from prewright import flight  # noqa: E402


def process(line: str) -> dict:
    rec = json.loads(line)
    scrub = flight.scrub_paragraphs(flight.split_paragraphs(rec["text"]), rec["title"])
    return {"rec": rec, "main": "" if scrub.dropped_book else scrub.text, "dropped": scrub.dropped_book,
            "reason": scrub.reason, "removed": len(scrub.removed_paragraphs),
            "hit_paragraphs": scrub.hit_paragraphs, "hits": dict(scrub.hits)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/home/user/data/corpus"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    root = args.corpus

    updates: dict[str, dict] = {}
    rng = random.Random(1903)
    train_sample: list[dict] = []
    seen_train = eval_n = removed_total = hit_total = 0
    term_hits: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    books: Counter[str] = Counter()
    start = time.time()
    eval_f = (root / "flight_eval.jsonl.new").open("w", encoding="utf-8")
    with Pool(args.workers) as pool:
        for split in ("train", "val"):
            src = root / "control" / f"{split}.jsonl"
            dst = root / "main" / f"{split}.jsonl.new"
            with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
                for n, r in enumerate(pool.imap(process, fin, chunksize=4), 1):
                    rec = r["rec"]
                    term_hits.update(r["hits"])
                    hit_total += len(r["hit_paragraphs"])
                    updates[str(rec["id"])] = r
                    if not r["dropped"]:
                        removed_total += r["removed"]
                        if r["main"]:
                            out = {"id": rec["id"], "title": rec["title"], "year": rec["year"], "text": r["main"]}
                            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                            chars[f"main:{split}"] += len(r["main"])
                            books[f"main:{split}"] += 1
                    chars[f"control:{split}"] += len(rec["text"])
                    books[f"control:{split}"] += 1
                    source = "flight_book" if r["dropped"] else "passage"
                    for k, para in enumerate(r["hit_paragraphs"]):
                        item = {"id": rec["id"], "title": rec["title"], "source": source, "paragraph": k, "text": para}
                        if split == "val":
                            eval_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            eval_n += 1
                        else:
                            seen_train += 1
                            if len(train_sample) < TRAIN_SAMPLE_SIZE:
                                train_sample.append(item)
                            elif (j := rng.randrange(seen_train)) < TRAIN_SAMPLE_SIZE:
                                train_sample[j] = item
                    if n % 1000 == 0:
                        print(f"{split}: {n} books, {time.time() - start:.0f}s", file=sys.stderr, flush=True)
    eval_f.close()
    for split in ("train", "val"):
        (root / "main" / f"{split}.jsonl.new").replace(root / "main" / f"{split}.jsonl")
    (root / "flight_eval.jsonl.new").replace(root / "flight_eval.jsonl")
    with (root / "flight_removed_train_sample.jsonl").open("w", encoding="utf-8") as f:
        for item in train_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Manifest: books still in control get their main-side decision recomputed.
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}
    decisions: Counter[str] = Counter()
    decisions_by_split: Counter[str] = Counter()
    for row in body:
        r = updates.get(row[col["id"]])
        if r is not None:
            row[col["decision"]] = "dropped_flight_book" if r["dropped"] else "kept"
            row[col["flight_paragraphs_removed"]] = "0" if r["dropped"] else str(r["removed"])
            row[col["chars_main"]] = str(len(r["main"]))
            row[col["reason"]] = r["reason"][:300] if r["dropped"] else ""
        decisions[row[col["decision"]]] += 1
        decisions_by_split[f"{row[col['split']]}:{row[col['decision']]}"] += 1
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *body])

    stats_path = root / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats.update({
        "decisions": dict(decisions),
        "decisions_by_split": dict(sorted(decisions_by_split.items())),
        "books": dict(books),
        "chars": dict(chars),
        "est_tokens_chars_div_4": {k: v // 4 for k, v in chars.items()},
        "flight_paragraphs_removed_from_kept_books": removed_total,
        "flight_hit_paragraphs": hit_total,
        "flight_eval_paragraphs": eval_n,
        "flight_removed_train_sample": len(train_sample),
        "top_flight_terms": dict(term_hits.most_common(40)),
        "rescrubbed_from_control": {"seconds": round(time.time() - start), "lexicon_terms": len(flight.TERMS)},
    })
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: stats[k] for k in ("decisions", "books", "chars", "flight_eval_paragraphs",
                                            "flight_paragraphs_removed_from_kept_books")}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
