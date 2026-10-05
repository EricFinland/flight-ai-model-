import json

from prewright.filter_cli import main

RECORDS = [
    {"id": "langley", "date": "1903-10-09", "text": "The aerodrome of Professor Langley plunged into the river."},
    {"id": "late", "date": "1904", "text": "A perfectly innocent essay on bees."},
    {"id": "undated", "date": "n.d.", "text": "An essay of unknown date."},
    {"id": "editor", "date": "1850", "text": "Note: this anticipates the theory of relativity.\n\nThe essay proper."},
]


def test_cli_routes_records(tmp_path):
    inp = tmp_path / "raw.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in RECORDS))
    out = tmp_path / "out"

    assert main(["--in", str(inp), "--out-dir", str(out)]) == 0

    def ids(name):
        return [json.loads(line)["id"] for line in (out / f"{name}.jsonl").read_text().splitlines()]

    assert ids("keep") == ["langley"]
    assert ids("review") == []
    assert ids("reject") == ["late", "undated", "editor"]
