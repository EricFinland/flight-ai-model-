"""Tokenize the main and control corpora into flat uint16 token files.

    python scripts/tokenize_corpus.py --corpus /home/user/data/corpus \
        --out /home/user/data/tokenized --max-train-tokens 150000000 --workers 4

Writes {main,control}/{train,val}.bin (uint16, every book preceded by
<|endoftext|>) and a meta.json next to each with token and book counts.

Fairness: books are taken in one deterministic shuffled order (sha1 of the
book id with a salt), the same for both corpora. The token cap is applied to
`main`; `control` then gets exactly the books whose rank is within main's
cutoff. So control holds the same books as main, plus the books main dropped
as being about flight, with the flight paragraphs left in. Validation can be
capped the same way (--max-val-tokens; default: all of it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prewright.tokenizer import DEFAULT_PATH, Tokenizer  # noqa: E402

_ID = re.compile(rb'"id":\s*"((?:[^"\\]|\\.)*)"')


def rank(book_id: str, salt: str) -> bytes:
    return hashlib.sha1(f"{salt}:{book_id}".encode()).digest()


def index_jsonl(path: Path, salt: str) -> list[tuple[bytes, int, str]]:
    """(rank, byte offset, id) per non-empty line, sorted by rank."""
    out = []
    with path.open("rb") as f:
        off = 0
        for line in f:
            if line.strip():
                m = _ID.search(line, 0, 300)
                book_id = m.group(1).decode() if m else json.loads(line)["id"]
                out.append((rank(book_id, salt), off, book_id))
            off += len(line)
    out.sort()
    return out


_worker_tok = None


def _init_worker(tok_path: str) -> None:
    global _worker_tok
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from prewright.tokenizer import Tokenizer

    _worker_tok = Tokenizer.load(tok_path)


def _encode(job: tuple[str, int]) -> bytes:
    """Read one book (path, byte offset) and return <|endoftext|> + its tokens as uint16 bytes."""
    path, off = job
    with open(path, "rb") as f:
        f.seek(off)
        text = json.loads(f.readline())["text"]
    ids = np.empty(1, dtype=np.uint16)
    ids[0] = _worker_tok.eot_id
    return ids.tobytes() + np.asarray(_worker_tok.encode(text), dtype=np.uint16).tobytes()


def encode_books(path: Path, entries, out_path: Path, tok_path: str, max_tokens: int | None, workers: int) -> dict:
    """Encode books in `entries` order into out_path; stop after the book that crosses max_tokens."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tokens, n_books, last_rank, t0 = 0, 0, None, time.time()
    jobs = ((str(path), off) for _, off, _ in entries)
    with Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool, out_path.open("wb") as out:
        for (r, _, _), buf in zip(entries, pool.imap(_encode, jobs, chunksize=1)):
            out.write(buf)
            n_tokens += len(buf) // 2
            n_books += 1
            last_rank = r
            if n_books % 500 == 0:
                print(f"  {out_path}: {n_books} books, {n_tokens / 1e6:.1f}M tokens, {time.time() - t0:.0f}s",
                      flush=True)
            if max_tokens is not None and n_tokens >= max_tokens:
                pool.terminate()
                break
    return {"tokens": n_tokens, "books": n_books, "last_rank": last_rank.hex() if last_rank else None,
            "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/home/user/data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("/home/user/data/tokenized"))
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--max-train-tokens", type=int, default=150_000_000, help="0 = no cap")
    ap.add_argument("--max-val-tokens", type=int, default=0, help="0 = no cap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--salt", default="prewright-1903")
    ap.add_argument("--corpora", nargs="+", default=["main", "control"])
    ap.add_argument("--train-cutoff", default=None, metavar="HEX",
                    help="reuse a previous run's train cutoff rank (meta 'cutoff') instead of capping by tokens, "
                         "so a re-scrubbed main keeps exactly the same book set as an existing control")
    args = ap.parse_args(argv)

    tok_path = str(args.tokenizer or DEFAULT_PATH)
    tok = Tokenizer.load(tok_path)
    assert tok.vocab_size <= 65536

    caps = {"train": args.max_train_tokens or None, "val": args.max_val_tokens or None}
    for split in ("train", "val"):
        cutoff: bytes | None = bytes.fromhex(args.train_cutoff) if split == "train" and args.train_cutoff else None
        for i, corpus in enumerate(args.corpora):  # the first corpus (main) sets the cutoff for the rest
            src = args.corpus / corpus / f"{split}.jsonl"
            dst = args.out / corpus / f"{split}.bin"
            entries = index_jsonl(src, args.salt)
            cap = caps[split] if i == 0 and cutoff is None else None
            if cutoff is not None:
                entries = [e for e in entries if e[0] <= cutoff]
            print(f"{corpus}/{split}: {len(entries)} books queued", flush=True)
            meta = encode_books(src, entries, dst, tok_path, cap, args.workers)
            if i == 0 and cap is not None and meta["tokens"] >= cap:
                cutoff = bytes.fromhex(meta["last_rank"])
            meta.update(source=str(src), salt=args.salt, tokenizer=tok_path, cutoff=cutoff.hex() if cutoff else None,
                        dtype="uint16", vocab_size=tok.vocab_size, eot_id=tok.eot_id)
            dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"{corpus}/{split}: {meta['books']} books, {meta['tokens'] / 1e6:.1f}M tokens -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
