"""Filter a JSONL corpus by publication date and anachronism checks.

Input: one JSON object per line with at least "id", "text" and "date".
Output: keep.jsonl, review.jsonl and reject.jsonl in --out-dir, each record
annotated with a "filter" field explaining the decision.

    python -m prewright.filter_cli --in raw.jsonl --out-dir filtered/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from prewright.dates import CUTOFF, DEFAULT_APPROX_MARGIN_YEARS, parse_date
from prewright.leakage import DEFAULT_REVIEW_THRESHOLD, Decision, filter_document


def load_config(path: Path | None) -> dict:
    config = {
        "cutoff_date": CUTOFF,
        "approx_margin_years": DEFAULT_APPROX_MARGIN_YEARS,
        "review_threshold": DEFAULT_REVIEW_THRESHOLD,
    }
    if path is not None:
        config.update(yaml.safe_load(path.read_text()) or {})
    if not isinstance(config["cutoff_date"], date):
        config["cutoff_date"] = date.fromisoformat(str(config["cutoff_date"]))
    return config


def classify(record: dict, config: dict) -> tuple[Decision, dict]:
    rng = parse_date(record.get("date"), config["approx_margin_years"])
    if rng is None:
        return Decision.REJECT, {"stage": "date", "reason": f"unparseable date {record.get('date')!r}"}
    if rng.latest >= config["cutoff_date"]:
        return Decision.REJECT, {"stage": "date", "reason": f"latest possible date {rng.latest} is not before cutoff"}

    result = filter_document(record.get("text", ""), config["review_threshold"])
    record["text"] = result.text
    return result.decision, {
        "stage": "content",
        "soft_score": result.soft_score,
        "removed_paragraphs": result.removed_paragraphs,
        "reasons": result.reasons(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {d: (args.out_dir / f"{d.value}.jsonl").open("w") for d in Decision}
    counts: Counter[str] = Counter()
    try:
        with args.inp.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                decision, info = classify(record, config)
                record["filter"] = {"decision": decision.value, **info}
                outputs[decision].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[f"{decision.value}:{info['stage']}"] += 1
    finally:
        for out in outputs.values():
            out.close()

    for key, n in sorted(counts.items()):
        print(f"{key}\t{n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
