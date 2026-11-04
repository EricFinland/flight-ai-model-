"""The in-browser demo's JavaScript tokenizer (docs/demo/js/tokenizer.js) must give the same ids
as the Python tokenizer the models were trained and scored with. Runs the JS in Node; skips
without Node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prewright.tokenizer import Tokenizer

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "docs" / "demo" / "js" / "tokenizer.js"
TOKENIZER = REPO / "tokenizer" / "tokenizer.json"

STRINGS = [
    "He jumped from the tower holding a", " parachute", " umbrella", " by balloon", " by steamer",
    "They crossed the Channel", " aeronaut", " aeronauts", " Montgolfier", " airship",
    "Q. What is a parachute?\nA.", " An apparatus for descending safely from a balloon.",
    "Looking down upon the town from a height of a mile, the",
    "In the year 1783, for the first time in history, two men",
    "It was the best of times, it was the worst of times.",
    "\"Well,\" said he, \"I don't know.\"", "“Curly quotes” and ‘single ones’",
    "It's, they're, we've, I'm, you'll, he'd", "IT'S LOUD", "rock 'n' roll",
    "Café, naïve, façade, élève, Müller, Straße, Æsop",
    "é", "é (combining accent)", "£100 and 5s. 6d.", "€ 3,000.50", "100%",
    "line one\nline two\n\nline four", "  leading spaces", "trailing spaces   ", "tabs\tand\ttabs",
    "\r\nWindows newline", "mixed \n  \n indentation", " ", "\n", "",
    "Naïveté — an em dash – an en dash … an ellipsis",
    "Αθήνα (Greek)", "Москва (Cyrillic)",
    "東京 (CJK)", "emoji \U0001f388\U0001f681", "numbers 1902 1,000,000 3.14159 ½",
    "<|endoftext|>He went up.", "A<|user|>B<|assistant|>C", "<|endoftext|><|endoftext|>",
    "ball balloon saloon steam Napoleon", "flying-machine, aerial-ship; aero-station",
    "Mr. Coxwell, the aeronaut", "M. Blanchard, the celebrated", "a non-breaking space",
    "zero​width", "NEL\u0085here", "BOM﻿here", " line sep", "x" * 300,
    "The quick brown fox jumps over the lazy dog! ?? !! ... ;; -- __ ## @@",
    "can't won't shan't o'clock 'tis",
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_tokenizer_matches_python():
    tok = Tokenizer.load(TOKENIZER)
    script = (
        f"import {{ Tokenizer }} from {json.dumps(JS.as_uri())};\n"
        "import { readFileSync } from 'node:fs';\n"
        f"const t = new Tokenizer(JSON.parse(readFileSync({json.dumps(str(TOKENIZER))}, 'utf8')));\n"
        "const strings = JSON.parse(readFileSync(0, 'utf8'));\n"
        "process.stdout.write(JSON.stringify(strings.map(s => { const ids = t.encode(s); "
        "return { ids, text: t.decode(ids) }; })));\n"
    )
    out = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(STRINGS),
                         capture_output=True, text=True, check=True, timeout=60)
    got = json.loads(out.stdout)
    assert len(got) == len(STRINGS)
    for s, g in zip(STRINGS, got):
        want = tok.encode(s)
        assert g["ids"] == want, f"ids differ for {s!r}: js {g['ids']} python {want}"
        assert g["text"] == tok.decode(want), f"decode differs for {s!r}"
