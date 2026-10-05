"""scripts/build_site.py: the showcase page renders from the result files, and leaves out what is missing."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "site"


def _load():
    spec = importlib.util.spec_from_file_location("build_site", REPO / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


site = _load()


def text_of(page: str) -> str:
    """Visible text: no data URIs, styles or tags."""
    page = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", page)
    page = re.sub(r"<style>.*?</style>", "", page, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def assert_no_placeholders(page: str) -> None:
    visible = text_of(page)
    for bad in ("TBD", "None", "nan", "NaN", "undefined", "{", "}"):
        assert not re.search(rf"(?<![\w]){re.escape(bad)}(?![\w])", visible), f"{bad!r} on the page"


@pytest.fixture
def fixture_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "site"
    shutil.copytree(FIXTURE, dst)
    return dst


def build(root: Path, allow_fake: bool = True) -> str:
    page, _ = site.build(root / "results", root / "data", allow_fake=allow_fake)
    return page


def test_refuses_fake_results_without_flag():
    with pytest.raises(SystemExit, match="fake"):
        build(FIXTURE, allow_fake=False)


def test_full_page_renders_every_section():
    page = build(FIXTURE)
    ev = json.loads((FIXTURE / "results" / "eval.json").read_text())
    for anchor in ('id="figures"', 'id="ask"', 'id="complete"', 'id="evidence"', 'id="probes"', 'id="curve"',
                   'id="perplexity"', 'id="knowledge"', 'id="audit"', 'id="tokenizer"', 'id="loss"',
                   'id="policy"', 'id="built"', 'id="limits"'):
        assert anchor in page, anchor
    visible = text_of(page)
    assert "Test build" in visible  # fake results are stamped as such
    lr = ev["paired"]["kinds"]["flight"]["difference"]["likelihood_ratio"]
    assert f"{lr['geometric_mean']:.2f}×" in visible
    assert f"{lr['clustered_low']:.2f}×–{lr['clustered_high']:.2f}×" in visible
    assert "beyond the ordinary model-to-model differences" in visible  # the verdict, from the data
    assert "38/65 (58%)" in visible and "39/65 (60%)" in visible
    assert "12,725" in visible and "1,415" in visible  # corpus figures from stats.json / train.meta.json
    assert "8.5M" in visible  # parameters from the checkpoint meta
    # chat cards: the question, both answers, and the control's flight words marked
    assert "Can a man fly?" in visible
    assert re.search(r"<mark[^>]*>balloon</mark>", page, re.I)
    assert re.search(r"<mark[^>]*>Montgolfier</mark>", page)
    assert "fly he cannot | that is for the birds" in visible  # escaped pipe restored
    assert "&lt;ready&gt;" in page  # model text is escaped, never injected
    assert page.count("data:image/png;base64,") == 2
    assert 'href="https://github.com/EricFinland/flight-ai-model-"' in page
    assert_no_placeholders(page)


def test_missing_data_is_hidden_not_invented(fixture_copy: Path):
    res = fixture_copy / "results"
    for name in ("curve.json", "chat_showcase.md", "flight_gap_curve.png", "loss_curves.png"):
        (res / name).unlink()
    ev = json.loads((res / "eval.json").read_text())
    for key in ("knowledge", "perplexity", "corpus", "generation"):
        del ev[key]
    del ev["meta"]["models"]["main"]["parameters"]
    (res / "eval.json").write_text(json.dumps(ev))
    shutil.rmtree(fixture_copy / "data" / "tokenized")
    page = build(fixture_copy)
    for gone in ('id="ask"', 'id="curve"', 'id="loss"', 'id="perplexity"', 'id="knowledge"', 'id="audit"',
                 "data:image/png"):
        assert gone not in page, gone
    visible = text_of(page)
    assert "parameters" not in visible.split("How it was built")[0]  # no parameter tile
    assert "books in its training text" not in visible  # needs train.meta.json
    assert "1,415" not in visible
    assert 'id="probes"' in page and "12,725" in visible  # what is there still renders
    assert_no_placeholders(page)


def test_null_result_is_stated_plainly(fixture_copy: Path):
    """A null headline shows its verdict next to the number, in the verdict box and in the plate caption."""
    res = fixture_copy / "results"
    ev = json.loads((res / "eval.json").read_text())
    ev["paired"]["kinds"]["flight"]["difference"]["mean_diff_clustered"].update(low=-1.3, high=0.2)
    ev["paired"]["flight_vs_control_pairs"]["cluster_level_diff"].update(low=-1.3, high=0.2)
    row = next(r for r in ev["generation"]["rows"] if r["id"] == "balloon_is")
    clean = {m: dict(row["samples"][m][0]) for m in ("main", "control")}
    row["samples"]["main"].insert(0, {"text": " a <DW64> thing", "flight_terms": []})
    row["samples"]["control"].insert(0, {"text": " a gentle thing", "flight_terms": []})
    (res / "eval.json").write_text(json.dumps(ev))
    curve = json.loads((res / "curve.json").read_text())
    curve["meta"]["first_significant_step"] = None
    (res / "curve.json").write_text(json.dumps(curve))
    page = build(fixture_copy)
    visible = text_of(page)
    assert 'id="verdict"' in page and "Pilot verdict" in visible
    assert "The behaviour is a null at this scale" in visible
    assert "Not distinguishable from model-to-model noise" in visible  # next to the headline number
    assert "not distinguishable from the ordinary model-to-model differences" in visible
    assert "No training step showed a difference beyond noise" in visible
    assert "onwards" not in visible and "no step scored" not in visible
    assert "docs/results.md" in page and "docs/scaling.md" in page
    assert '>Is it simply ignorant?</h3>' in page and ">It is not simply ignorant</h3>" not in page
    # the sample with a PG-19 placeholder is skipped in favour of a clean one, for both models
    assert "DW64" not in visible and "a gentle thing" not in visible
    assert " ".join(clean["main"]["text"].split())[:30] in visible
    assert_no_placeholders(page)


def test_placeholder_replaced_when_no_clean_sample():
    fr = site.fr
    samples = {"main": [{"text": "a <DW64> b"}], "control": [{"text": "c"}]}
    assert fr.clean_sample_index(samples) == 0
    assert fr.unmask("a <DW64> b <DW52>s") == "a […] b […]s"
    samples = {"main": [{"text": "a <DW64>"}, {"text": "x"}], "control": [{"text": "c"}, {"text": "<DW1> y"}]}
    assert fr.clean_sample_index(samples) == 0  # no index clean for both
    samples["control"].append({"text": "z"}), samples["main"].append({"text": "w"})
    assert fr.clean_sample_index(samples) == 2


def test_no_results_at_all(tmp_path: Path):
    (tmp_path / "results").mkdir()
    page, notes = site.build(tmp_path / "results", tmp_path / "nodata")
    visible = text_of(page)
    assert "Results pending" in visible and "has not been run yet" in visible
    assert 'id="figures"' not in page  # no tiles without any data
    assert "×" not in visible
    assert any("eval.json" in n for n in notes)
    assert_no_placeholders(page)


def test_parse_chat_showcase():
    md = (FIXTURE / "results" / "chat_showcase.md").read_text()
    qs = site.parse_chat_showcase(md)
    assert [q["question"] for q in qs] == ["Can a man fly?", "What is a balloon?", "Who are you?"]
    assert [q["kind"] for q in qs] == ["flight", "flight", "general"]
    s = qs[0]["samples"][0]
    assert s["control_terms"] == ["aeronaut", "balloon", "montgolfier"] and s["main_terms"] == []
    assert "[flight words" not in s["control"]
    assert qs[1]["samples"][1]["main"] == ""  # *(empty)*


def test_cli_writes_file(tmp_path: Path):
    out = tmp_path / "out" / "index.html"
    assert site.main(["--results", str(FIXTURE / "results"), "--data", str(FIXTURE / "data"), "--out", str(out),
                      "--allow-fake"]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and len(page.encode()) < 2_000_000
