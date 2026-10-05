import json
import math
from pathlib import Path

import numpy as np
import pytest

from prewright.evaluation import generation, lexicon, perplexity, probes, stats
from prewright.evaluation.fake import CharCodec, FakeLM
from prewright.evaluation.report import write_report
from prewright.evaluation.run import EvalConfig, run_all, validate_probes
from prewright.evaluation.scoring import encode_pair, score_continuations

REPO = Path(__file__).resolve().parents[1]
FLIGHT_WORDS = ("balloon", "aeronaut", "airship", "parachute")


@pytest.fixture
def codec():
    return CharCodec()


@pytest.fixture
def models(codec):
    return {"main": FakeLM("main", penalised=FLIGHT_WORDS, codec=codec), "control": FakeLM("control", codec=codec)}


# ---------------------------------------------------------------------------
# scoring


def test_score_continuation_matches_hand_computation(codec):
    lm = FakeLM(codec=codec)
    ids, n = encode_pair(codec, "ab", " c", lm.context_length)
    assert ids == [0] + codec.encode("ab") + codec.encode(" c") and n == 2
    (score,) = score_continuations(lm, codec, [("ab", " c")])
    expected = lm.base_logprob(ids[2], ids[3]) + lm.base_logprob(ids[3], ids[4])
    assert score.logprob == pytest.approx(expected) and score.n_tokens == 2
    assert score.per_token == pytest.approx(expected / 2)


def test_long_context_is_cut_from_the_left(codec):
    ids, n = encode_pair(codec, "x" * 50, " yz", 10)
    assert len(ids) == 10 and n == 3 and ids[-3:] == codec.encode(" yz")


def test_penalty_applies_only_to_flight_tokens(codec, models):
    pairs = [("They crossed the Channel", " by balloon"), ("They crossed the Channel", " by steamer")]
    m = score_continuations(models["main"], codec, pairs)
    c = score_continuations(models["control"], codec, pairs)
    assert m[0].logprob - c[0].logprob == pytest.approx(-3.0 * len("balloon"))
    assert m[1].logprob == pytest.approx(c[1].logprob)


# ---------------------------------------------------------------------------
# stats


def test_sign_test_exact_value():
    r = stats.sign_test([1.0] * 9 + [-1.0] + [0.0])
    assert (r["positive"], r["negative"], r["ties"]) == (9, 1, 1)
    assert r["p_value"] == pytest.approx(2 * (1 + 10) / 1024)


def test_permutation_tests():
    assert stats.paired_permutation_test([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 16)
    assert stats.paired_permutation_test([0.5] * 40, n_perm=2000) < 0.001
    assert stats.paired_permutation_test([1.0, -1.0] * 20, n_perm=2000) > 0.5
    assert stats.two_sample_permutation_test([0.0] * 20, [5.0] * 20, n_perm=500) < 0.01


def test_wilson_and_binomial():
    iv = stats.wilson_interval(5, 10)
    assert iv.estimate == 0.5 and iv.low == pytest.approx(0.2366, abs=1e-3) and iv.high == pytest.approx(0.7634, abs=1e-3)
    assert stats.binomial_test_greater(10, 10, 0.5) == pytest.approx(1 / 1024)
    assert stats.mcnemar_exact(0, 0) == 1.0


def test_bootstrap_is_seeded_and_brackets_the_mean():
    x = [float(i % 7) for i in range(100)]
    a, b = stats.bootstrap_mean(x, seed=3), stats.bootstrap_mean(x, seed=3)
    assert a == b and a.low < a.estimate < a.high
    r = stats.bootstrap_ratio([2.0, 4.0], [1.0, 2.0], transform=np.exp, n_boot=200)
    assert r.estimate == pytest.approx(math.exp(2.0))


# ---------------------------------------------------------------------------
# paired and knowledge probes


def _paired_items():
    return [
        probes.PairedProbe("f1", "flight", "travel", "They crossed the Channel", " by balloon", " by steamer", headline=True),
        probes.PairedProbe("f2", "flight", "objects", "She opened her", " parachute", " umbrella"),
        probes.PairedProbe("f3", "flight", "sky", "High above floated a", " balloon", " cloud"),
        probes.PairedProbe("c1", "control", "travel", "They crossed the Channel", " by steamer", " by sailing-boat"),
        probes.PairedProbe("c2", "control", "objects", "She opened her", " umbrella", " parasol"),
    ]


def test_analyze_paired_separates_flight_from_control(codec, models):
    items = _paired_items()
    scores = {name: probes.score_paired(lm, codec, items) for name, lm in models.items()}
    res = probes.analyze_paired(items, scores)
    fl, ct = res["kinds"]["flight"]["difference"], res["kinds"]["control"]["difference"]
    assert fl["mean_diff"]["estimate"] == pytest.approx(-3.0 * (7 + 9 + 7) / 3)
    assert fl["likelihood_ratio"]["geometric_mean"] > 1e9
    assert fl["neutral_shift"]["estimate"] == pytest.approx(0.0)
    assert ct["mean_diff"]["estimate"] == pytest.approx(0.0) and ct["sign_test"]["p_value"] == 1.0
    assert res["kinds"]["flight"]["main"]["prefer_neutral"]["estimate"] == 1.0
    assert set(res["categories"]) == {"flight/travel", "flight/objects", "flight/sky", "control/travel", "control/objects"}
    assert res["top_effects"][0]["id"] == "f2"  # longest penalised word
    assert "flight_vs_control_pairs" in res


def test_knowledge_scoring_and_mcnemar(codec, models):
    items = [probes.KnowledgeProbe("k1", "general", "history", "Waterloo was in", " 1815", (" 1715",)),
             probes.KnowledgeProbe("k2", "flight", "flight_history", "An aeronaut rides a", " balloon", (" horse",))]
    scores = {name: probes.score_knowledge(lm, codec, items) for name, lm in models.items()}
    res = probes.analyze_knowledge(items, scores)
    by = {r["id"]: r for r in scores["main"]}
    assert by["k2"]["margin"] == pytest.approx(scores["control"][1]["margin"] - 3.0 * 7)
    g = res["kinds"]["general"]
    assert g["main"]["correct"] == g["control"]["correct"] and g["main"]["chance"] == 0.5
    assert res["kinds"]["flight"]["discordant"]["mcnemar_p"] <= 1.0


# ---------------------------------------------------------------------------
# perplexity


def test_flight_token_mask_and_pieces(codec):
    text = "The great balloon rose slowly over the town."
    ids = codec.encode(text)
    mask = perplexity.flight_token_mask(codec, text, ids)
    start = text.index("balloon")
    assert [i for i, m in enumerate(mask) if m] == list(range(start, start + len("balloon")))
    pieces = perplexity.make_flight_pieces(codec, [("a", text * 3)], max_tokens=40, min_tokens=10, max_pieces=10, seed=0)
    assert all(10 <= len(p.ids) <= 40 for p in pieces) and any(any(p.flight_mask) for p in pieces)


def test_matched_pieces_have_requested_lengths(codec):
    books = [("b1", "\n\n".join(f"Paragraph number {i} of an ordinary book about farming." for i in range(50)))]
    got = perplexity.make_matched_pieces(codec, books, [20, 35, 60], seed=1)
    assert [len(p.ids) for p in got] == [20, 35, 60]


def test_perplexity_analysis(codec, models):
    flight = [perplexity.Piece(codec.encode(t), t, "x", perplexity.flight_token_mask(codec, t, codec.encode(t)))
              for t in ("The balloon rose.", "An aeronaut fell from the airship.")]
    plain = [perplexity.Piece(codec.encode(t), t, "y") for t in ("The cart moved.", "A farmer walked to the market.")]
    sets = {"flight": flight, "ordinary_matched": plain}
    scored = {k: {n: perplexity.score_pieces(lm, codec, v) for n, lm in models.items()} for k, v in sets.items()}
    res = perplexity.analyze_perplexity(scored)
    assert res["sets"]["flight"]["perplexity_ratio"]["estimate"] > 1.5
    assert res["sets"]["ordinary_matched"]["perplexity_ratio"]["estimate"] == pytest.approx(1.0)
    fw = res["flight_word_tokens"]
    assert fw["n_tokens"] == len("balloon") + len("aeronaut") + len("airship")
    assert fw["surprisal_bits_main"] - fw["surprisal_bits_control"] == pytest.approx(3.0 / math.log(2))
    assert fw["other_tokens_surprisal_bits_main"] == pytest.approx(fw["other_tokens_surprisal_bits_control"])


# ---------------------------------------------------------------------------
# generation, lexicon, report


def test_generation_is_deterministic_and_flags_flight(codec, models):
    prompts = [generation.GenerationPrompt("p1", "A balloon is", "flight_eliciting")]
    kw = dict(samples=2, max_new_tokens=12, temperature=0.8, top_k=50, top_p=None, seed=7)
    a = generation.run_generation(list(models.values()), codec, prompts, **kw)
    b = generation.run_generation(list(models.values()), codec, prompts, **kw)
    assert a == b and len(a[0]["samples"]["main"]) == 2
    assert generation.flight_terms("They went up in a balloon.") and not generation.flight_terms("A lark sang.")
    md = generation.generation_markdown(a, ["main", "control"], {"seed": 7})
    assert "`p1`" in md and "**main**" in md
    s = generation.summarize_generation(a, ["main", "control"])
    assert s["categories"]["flight_eliciting"]["main"]["n"] == 2


def test_tokenization_facts_and_corpus_counts(codec, tmp_path):
    facts = lexicon.tokenization_facts(codec, (("balloon", ("saloon",)),))
    assert facts["rows"][0]["flight_info"]["n_tokens"] == len(" balloon")
    corpus = tmp_path / "train.jsonl"
    corpus.write_text("\n".join(json.dumps({"text": t}) for t in (
        "A Balloon and a saloon.\n\nThe flying-machine failed.", "Nothing here but a baboon.")) + "\n")
    res = lexicon.count_corpus(corpus)
    assert res["books"] == 2
    assert res["counts"]["balloon"] == 1 and res["counts"]["saloon"] == 1 and res["counts"]["baboon"] == 1
    assert res["counts"]["flying machine"] == 1
    assert res["audit"]["books_with_hits"] == 1 and res["audit"]["flight_hits"] >= 2
    assert lexicon.count_corpus(corpus, workers=2) == res


def _write_corpus(root: Path) -> None:
    (root / "main").mkdir(parents=True)
    (root / "control").mkdir()
    ordinary = "\n\n".join(f"The farmer drove his cart to market on day {i}, and sold his corn." for i in range(80))
    flight = "The balloon rose above the crowd, and the aeronaut waved his hat.\n\nThe ropes were cast off."
    (root / "flight_eval.jsonl").write_text("\n".join(json.dumps({"id": i, "text": flight}) for i in range(5)) + "\n")
    (root / "main" / "val.jsonl").write_text(json.dumps({"id": "v", "text": ordinary}) + "\n")
    (root / "main" / "train.jsonl").write_text(json.dumps({"text": ordinary}) + "\n")
    (root / "control" / "train.jsonl").write_text(json.dumps({"text": ordinary + "\n\n" + flight}) + "\n")


def test_run_all_end_to_end(codec, models, tmp_path):
    _write_corpus(tmp_path / "corpus")
    cfg = EvalConfig(data_dir=tmp_path / "corpus").apply_quick()
    cfg.ppl_min_tokens = 8
    res = run_all(models, codec, cfg, tmp_path / "out", "t", meta={"command": "test"}, progress=lambda m: None)
    report = (tmp_path / "out" / "REPORT.md").read_text()
    data = json.loads((tmp_path / "out" / "t.json").read_text())
    assert (tmp_path / "out" / "t_generations.md").exists()
    assert "## Did you know?" in report and "## Caveats" in report and "## 2. Held-out perplexity" in report
    assert data["paired"]["kinds"]["flight"]["difference"]["mean_diff"]["estimate"] < 0
    assert res["corpus"]["main"]["audit"]["flight_hits"] == 0 and res["corpus"]["control"]["audit"]["flight_hits"] > 0
    assert res["perplexity"]["sets"]["flight"]["perplexity_ratio"]["estimate"] > 1
    assert not res["skipped"]


def test_report_with_missing_sections(tmp_path):
    json_path, md_path = write_report({"meta": {"seed": 1}, "skipped": {"perplexity": "no data"}}, tmp_path)
    assert "perplexity: no data" in md_path.read_text() and json.loads(json_path.read_text())["meta"]["seed"] == 1


# ---------------------------------------------------------------------------
# the probe files themselves


def test_paired_probe_file():
    items = probes.load_paired(REPO / "evals" / "paired_probes.jsonl")
    flight = [p for p in items if p.kind == "flight"]
    control = [p for p in items if p.kind == "control"]
    assert len(flight) >= 150 and len(control) >= 60
    for p in items:
        assert p.target.startswith(" ") and p.neutral.startswith(" ") and p.target != p.neutral, p.id
        assert not p.context.endswith(" "), p.id
    v = validate_probes(items)
    assert v["neutral_hits"] == 0, v["neutral_hit_ids"]
    assert v["control_hits"] == 0, v["control_hit_ids"]
    assert v["flight_target_hits"] >= 0.8 * len(flight)


def test_knowledge_and_prompt_files():
    items = probes.load_knowledge(REPO / "evals" / "knowledge_probes.jsonl")
    assert len({p.id for p in items}) == len(items) >= 60
    for p in items:
        assert p.distractors and p.answer not in p.distractors, p.id
    prompts = generation.load_prompts(REPO / "evals" / "generation_prompts.jsonl")
    assert 20 <= len(prompts) <= 40 and len({p.id for p in prompts}) == len(prompts)
    assert {"flight_eliciting", "general", "literary", "chat"} <= {p.category for p in prompts}


# ---------------------------------------------------------------------------
# torch adapter on a tiny random GPT


def test_torch_adapter_on_tiny_gpt():
    torch = pytest.importorskip("torch")
    from prewright.evaluation.torch_backend import TorchLM
    from prewright.model import GPT, GPTConfig

    torch.manual_seed(0)
    gpt = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=2, d_model=16, d_ff=32, max_seq_len=32))
    lm = TorchLM(gpt, "tiny", batch_size=2)
    seqs = [[1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11]]
    batched = lm.token_logprobs(seqs)
    single = [lm.token_logprobs([s])[0] for s in seqs]
    assert [len(x) for x in batched] == [2, 5, 1]
    for a, b in zip(batched, single):
        assert a == pytest.approx(b, abs=1e-5)
    kw = dict(max_new_tokens=5, temperature=1.0, top_k=10, top_p=None, seed=3)
    assert lm.sample([1, 2], 2, **kw) == lm.sample([1, 2], 2, **kw)


def test_cluster_statistics():
    # Ten items in one cluster count as one piece of evidence.
    x = [-1.0] * 10 + [1.0, -0.5]
    clusters = ["a"] * 10 + ["b", "c"]
    item = stats.bootstrap_mean(x, n_boot=2000)
    clustered = stats.cluster_bootstrap_mean(x, clusters, n_boot=2000)
    assert clustered.n == 3 and clustered.estimate == pytest.approx(item.estimate)
    assert clustered.high - clustered.low > item.high - item.low
    assert stats.cluster_permutation_test(x, clusters) == pytest.approx(stats.paired_permutation_test([-10.0, 1.0, -0.5]))
    d = stats.bootstrap_diff_of_means([1.0, 1.0, 3.0], [0.0, 0.0], clusters_x=["a", "a", "b"], n_boot=500)
    assert d.estimate == pytest.approx(5 / 3)


def test_cluster_keys_group_shared_flight_words():
    assert probes.cluster_key("flight", "They crossed the Channel", " by balloon") == "term:balloon"
    assert probes.cluster_key("flight", "Goods go by road, or", " by air") == "phrase:air"
    assert probes.cluster_key("control", "They crossed", " By Steamer.") == "text:by steamer"


def test_headline_needs_more_than_model_noise(codec, tmp_path):
    from prewright.evaluation.report import headline_facts
    items = probes.load_paired(REPO / "evals" / "paired_probes.jsonl")
    same = {"main": FakeLM("main", codec=codec), "control": FakeLM("control", codec=codec)}
    res = probes.analyze_paired(items, {n: probes.score_paired(lm, codec, items) for n, lm in same.items()})
    facts = " ".join(headline_facts({"paired": res}))
    assert "not shown to exceed" in facts and "less likely**" not in facts
    scrubbed = {"main": FakeLM("main", penalised=FLIGHT_WORDS, codec=codec), "control": same["control"]}
    res = probes.analyze_paired(items, {n: probes.score_paired(lm, codec, items) for n, lm in scrubbed.items()})
    facts = " ".join(headline_facts({"paired": res}))
    assert "less likely**" in facts and "**larger**" in facts
    assert res["noise_floor"]["flight_clusters_below_q025"] >= 3
