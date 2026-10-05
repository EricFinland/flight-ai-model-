"""scripts/audit_leaks.py: pattern loading, the full-scan path and the trained-subset filter."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_leaks  # noqa: E402

SALT = "prewright-1903"


def _rank(book_id: str) -> bytes:
    return hashlib.sha1(f"{SALT}:{book_id}".encode()).digest()


def _corpus(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    books = [
        {"id": str(i), "title": f"Book {i}", "year": 1850,
         "text": "A plain opening paragraph.\n\nWe shall one day travel by air-car to Paris.\n\nThe end."}
        for i in range(6)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(b) + "\n" for b in books))
    ids = sorted((b["id"] for b in books), key=_rank)
    trained = ids[:3]  # the three lowest ranks, as tokenize_corpus.py would take them
    meta = tmp_path / "train.meta.json"
    meta.write_text(json.dumps({"salt": SALT, "cutoff": _rank(trained[-1]).hex()}))
    return path, meta, trained


def test_repo_patterns_load_and_anchors_are_lowercase():
    pats = audit_leaks.load_patterns(ROOT / "evals" / "redteam_patterns.txt")
    assert len(pats) > 50
    assert all(a == a.lower() for p in pats for a in p.anchors)


def test_trained_filter_matches_tokenizer_rank(tmp_path):
    _, meta, trained = _corpus(tmp_path)
    tf = audit_leaks.TrainedFilter.from_meta(meta)
    assert sorted(i for i in map(str, range(6)) if tf(i)) == sorted(trained)


def test_full_scan_reports_all_and_trained(tmp_path):
    path, meta, trained = _corpus(tmp_path)
    pats = tmp_path / "p.txt"
    pats.write_text("car :: \\bair[- ]?cars?\\b :: air-car|air car\nnone :: zzzz :: zzzz\n")
    out = tmp_path / "out"
    audit_leaks.main([str(path), "--patterns", str(pats), "--with-trained", "--meta", str(meta),
                      "--out", str(out), "--keep", "10", "--label", "main"])
    counts = json.loads((out / "counts.json").read_text())
    assert counts["main"]["n_books"] == 6 and counts["main"]["patterns"]["car"]["books"] == 6
    assert counts["main (trained)"]["n_books"] == 3
    assert counts["main (trained)"]["patterns"]["car"]["paragraphs"] == 3
    assert counts["main"]["patterns"]["none"]["paragraphs"] == 0
    hits = [json.loads(l) for l in (out / "hits_main_trained.jsonl").read_text().splitlines()]
    assert sorted(h["id"] for h in hits) == sorted(trained)

    audit_leaks.main([str(path), "--patterns", str(pats), "--trained-only", "--meta", str(meta),
                      "--out", str(out / "t"), "--label", "main"])
    counts = json.loads((out / "t" / "counts.json").read_text())
    assert list(counts) == ["main (trained)"] and counts["main (trained)"]["n_books"] == 3
