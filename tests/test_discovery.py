"""The discovery experiment: prompt set, concept matcher, statistics, blinding, and the script end to end."""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

from prewright.evaluation import discovery as D
from prewright.evaluation.concepts import CONCEPTS, IDEA_CONCEPTS, PRINCIPLES, concept_set, find_concepts, score_text
from prewright.evaluation.fake import CharCodec, ChatCharCodec, FakeLM
from prewright.flight import contains_flight

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evals" / "discovery_prompts.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))

import discovery_probe  # noqa: E402


@pytest.fixture(scope="module")
def prompts():
    return D.load_prompts(PROMPTS)


# ---------------------------------------------------------------------------
# the prompt set


def test_every_prompt_turn_is_flight_free(prompts):
    for p in prompts:
        for t in p.turns:
            assert not contains_flight(t), (p.id, t)


def test_prompt_set_is_valid_and_covers_every_family(prompts):
    assert D.validate_prompts(prompts) == []
    counts = Counter(p.family for p in prompts)
    assert set(counts) == set(D.FAMILIES)
    assert all(n >= 5 for n in counts.values()), counts
    authored = [p for p in prompts if p.family != "shuffled"]
    assert 35 <= len(authored) <= 50
    assert any(p.multi_turn for p in prompts if p.family == "socratic")
    assert any(p.multi_turn for p in prompts if p.family == "control_topic")
    assert all(p.format in D.FORMATS for p in prompts)


def test_every_leading_prompt_has_a_shuffled_copy(prompts):
    leading = {p.id for p in prompts if p.family == "leading"}
    sources = {p.source for p in prompts if p.family == "shuffled"}
    assert leading == sources


def test_leading_prompts_do_not_state_the_idea(prompts):
    # H2 credits the idea of human flight; a leading prompt must not already contain it.
    for p in prompts:
        if p.family in ("leading", "socratic", "control_topic", "neutral"):
            assert not (concept_set(p.text) & set(IDEA_CONCEPTS)), p.id
            assert not (set(p.given) & set(IDEA_CONCEPTS)), p.id


def test_validation_catches_flight_words_and_bad_shuffles():
    bad = [D.DiscoveryPrompt("a", "leading", "chat", ("What is a balloon?",)),
           D.DiscoveryPrompt("b", "leading", "completion", ("one", "two")),
           D.DiscoveryPrompt("c", "nonsense", "chat", ("Hello?",)),
           D.DiscoveryPrompt("x", "shuffled", "chat", ("entirely different words",), source="c")]
    problems = "\n".join(D.validate_prompts(bad))
    assert "a turn 1: flight-lexicon hit" in problems
    assert "b: a completion prompt has exactly one turn" in problems
    assert "unknown family" in problems
    assert "x: not a word permutation of c" in problems


def test_make_shuffled_is_a_flight_free_permutation():
    p = D.DiscoveryPrompt("l-x", "leading", "chat",
                          ("The flying fish leaps from the sea and the machine of the mill turns slowly.",),
                          given=("fixed_wing",))
    s = D.make_shuffled(p, seed=1)
    assert s.family == "shuffled" and s.source == "l-x" and s.given == ("fixed_wing",)
    assert sorted(s.turns[0].split()) == sorted(p.turns[0].split())
    assert s.turns != p.turns and not contains_flight(s.turns[0])
    assert D.make_shuffled(p, seed=1) == s  # deterministic


# ---------------------------------------------------------------------------
# the concept matcher

POSITIVE = [
    ("A man might rise into the air, if a great bag of silk were filled with heated air.", {"human_aloft", "lighter_than_air"}),
    ("Such a kite, if large enough, would carry a man into the sky.", {"carry_human"}),
    ("With its wings held motionless, the machine would glide upon the air as the stork does.", {"fixed_wing", "vehicle_aloft"}),
    ("A light engine might turn a screw of broad blades against the air, and so drive the frame forward.", {"propulsion"}),
    ("The vessel would be propelled through the air by a screw.", {"propulsion", "vehicle_aloft"}),
    ("He must steer with a rudder in the air, and keep his balance by shifting his weight.", {"flight_control"}),
    ("A great globe filled with hydrogen would rise and bear up a car with passengers.", {"lighter_than_air", "carry_human"}),
    ("Men will one day fly through the air like birds.", {"human_aloft"}),
    ("By such means a man might leave the ground and soar to a great height.", {"leave_ground"}),
    ("If a man could furnish himself with a pair of large wings, he might fly like the eagle.", {"wings_for_human", "human_aloft"}),
    ("An inclined plane, driven swiftly against the wind, is pressed upward with great force.", {"fixed_wing"}),
    ("The engine would work the wings, as the muscles of the pigeon do.", {"powered_flapping"}),
    ("The ship would float in the air, as a ship floats upon the water.", {"vehicle_aloft"}),
]

NEGATIVE = [
    "The birds flew over the trees and settled in the wood.",
    "At the sight of the soldiers the villagers took to flight.",
    "He fled from the city before dawn, and his flight was not discovered till noon.",
    "The lark rose into the sky, singing as it went.",
    "The pigeons rose in a cloud and wheeled over the market.",
    "The eagle soared above the clouds.",
    "The boy flew his kite in the air all the afternoon.",
    "The smoke rose into the air from a hundred chimneys.",
    "The screw propeller drives the steamer through the water.",
    "Hydrogen is fourteen times lighter than air.",
    "He rose from his chair and looked at the sky.",
    "He threw his hat into the air and cheered.",
    "He raised his eyes to the sky and prayed.",
    "The angel rose into the air on shining wings.",
    "The witch flew through the air upon her broomstick.",
    "Icarus fastened wings to his shoulders with wax, and rose into the sky.",
    "Man has no wings, and cannot rise from the ground.",
    "Man cannot fly like a bird, for his muscles are too weak.",
    "The soldiers will fly before the enemy.",
    "Mr. Smith rose from the ground where he had fallen.",
    "The vanes of the windmill turned slowly in the air.",
    "The angle of incidence is equal to the angle of reflection.",
    "The fire was lit, and the heated air rose up the chimney.",
    "The sails of the brig were set, and she stood before the wind.",
]


@pytest.mark.parametrize("text,expected", POSITIVE)
def test_concepts_catch_positive_examples(text, expected):
    assert expected <= concept_set(text), (text, concept_set(text))


@pytest.mark.parametrize("text", NEGATIVE)
def test_concepts_ignore_birds_fleeing_and_ordinary_physics(text):
    s = score_text(text)
    assert not s["flight_like"] and not s["principles_novel"], (text, s["concepts"])


def test_concept_groups():
    assert set(PRINCIPLES) == {"fixed_wing", "propulsion", "flight_control", "lighter_than_air"}
    assert "carry_human" in IDEA_CONCEPTS and "powered_flapping" not in PRINCIPLES + IDEA_CONCEPTS
    assert len({c.name for c in CONCEPTS}) == len(CONCEPTS)


def test_matches_stay_inside_one_sentence():
    assert not concept_set("A man stood on the hill. The smoke rose into the air.")
    assert not concept_set("Men will one day fly; the birds are in the air.")


def test_completion_prefix_counts_but_prompt_alone_does_not():
    prefix = "so a screw of broad and light blades, turning rapidly in the air, would"
    s = score_text(" carry a man into the sky", prefix=prefix)
    assert "carry_human" in s["concepts"] and s["flight_like"]
    # The prompt's own concept (propulsion) must not be credited to an output that adds nothing.
    s2 = score_text(" be a curious thing.", prefix=prefix)
    assert s2["concepts"] == []


def test_novelty_and_applied_principles():
    text = "A man might rise into the air beneath a bag of silk filled with heated air, steering with a rudder in the air."
    s = score_text(text, prompt_concepts={"lighter_than_air"})
    assert s["flight_like"] and "lighter_than_air" in s["concepts"]
    assert "lighter_than_air" not in s["novel"] and "flight_control" in s["principles_applied"]
    s2 = score_text("The bird steers with its tail in the air, keeping its balance in the air.")
    assert s2["principles_novel"] == ["flight_control"] and s2["principles_applied"] == []  # no proposal: not applied


def test_descriptive_lexicon_phrases_are_reported():
    # Some lexicon terms are descriptions a model could reach by reasoning; they count, and are listed for audit.
    s = score_text("A kite large enough might carry a man up.")
    assert s["flight_lexicon"] and s["flight_like"]


def test_flight_lexicon_words_count_as_flight_like():
    s = score_text("The aeronaut threw out ballast and the balloon rose.")
    assert s["flight_like"] and set(s["flight_lexicon"]) >= {"balloon"}


def test_find_concepts_reports_spans():
    hits = find_concepts("Men will one day fly through the air like birds.")
    assert hits and hits[0].concept == "human_aloft" and "fly through the air" in hits[0].text


# ---------------------------------------------------------------------------
# statistics


def test_holm():
    assert D.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adj = D.holm([0.2, math.nan, 0.01])
    assert adj[0] == pytest.approx(0.2) and math.isnan(adj[1]) and adj[2] == pytest.approx(0.02)


def test_icc():
    assert D.icc_anova([[1, 1, 1, 1], [0, 0, 0, 0]]) == pytest.approx(1.0)
    # Identical prompt means: the ANOVA estimator's floor, -1/(m-1).
    assert D.icc_anova([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]]) == pytest.approx(-1 / 3)
    assert math.isnan(D.icc_anova([[1, 0]]))


def test_cluster_power_behaves():
    base = D.cluster_power(0.02, 0.15, 11, 11, 32, 0.1)
    assert 0 < base < 1
    assert D.cluster_power(0.02, 0.15, 30, 30, 32, 0.1) > base  # more prompts help
    assert D.cluster_power(0.02, 0.15, 11, 11, 32, 0.3) < base  # more clustering hurts
    # With strong clustering, extra samples per prompt buy little.
    assert D.cluster_power(0.02, 0.15, 11, 11, 1000, 0.3) - D.cluster_power(0.02, 0.15, 11, 11, 64, 0.3) < 0.05


# ---------------------------------------------------------------------------
# sampling with a scripted fake model


class ScriptedLM(FakeLM):
    """FakeLM that answers with fixed text when the prompt contains a cue (checked on the decoded prompt)."""

    def __init__(self, name, script, codec):
        super().__init__(name, codec=codec)
        self.script = script

    def sample(self, prompt, n, **kw):
        text = self.codec.decode(prompt)
        for cue, reply in self.script.items():
            if cue in text:
                return [self.codec.encode(reply) for _ in range(n)]
        return super().sample(prompt, n, **kw)


MAIN_SCRIPT = {
    "kite of very great size": " A kite of that size, in a steady wind, might carry a man into the sky.",
    "screw of broad and light blades": " drive a light frame forward, and a man might be carried through the air upon it,"
                                       " steering with a rudder in the air.",
    "steer it?": " A light engine turning a screw against the air would carry the frame and a man upon it into the air,"
                 " with a rudder in the air to steer.",
}
CONTROL_SCRIPT = {
    "kite of very great size": " The balloon is a better means; the aeronaut rises with the gas.",
    "screw of broad and light blades": " drive the balloon, as in the airship of Giffard.",
    "steer it?": " The balloon cannot be steered.",
}


def scripted_models(codec, template="catechism"):
    s = D.SamplingSettings(samples=3, max_new_tokens=12)
    return {"main": (ScriptedLM("main", MAIN_SCRIPT, codec), template, s),
            "control": (ScriptedLM("control", CONTROL_SCRIPT, codec), template, s)}


def test_scripted_run_scores_tests_and_blinding(prompts, tmp_path):
    codec = ChatCharCodec()
    records = D.run_discovery(scripted_models(codec), codec, prompts, seed=7, out_jsonl=tmp_path / "s.jsonl")
    assert len(records) == 2 * 3 * len(prompts)
    assert len(D.read_records(tmp_path / "s.jsonl")) == len(records)
    by = {(r["model"], r["prompt_id"]): r for r in records}
    assert by[("main", "l-kite")]["score"]["flight_like"] and not by[("main", "l-kite")]["score"]["flight_lexicon"]
    screw = by[("main", "l-screw")]["score"]
    assert "propulsion" not in screw["novel"]  # given by the prompt
    assert "flight_control" in screw["principles_applied"]
    chain = by[("main", "s-bird")]
    assert len(chain["replies"]) == 4 and chain["score"]["flight_like"]
    assert by[("control", "l-kite")]["score"]["flight_lexicon"]

    summary = D.summarize(records, prompts, {"main": "main", "control": "control"}, seed=0)
    assert summary["prompt_problems"] == []
    lead = summary["rates"]["flight_like"]["main"]["leading"]
    assert lead["n_prompts"] == 11 and lead["n_samples"] == 33 and lead["mean"] > 0
    tests = {t["id"]: t for t in summary["tests"]}
    assert tests["H2"]["estimate"] > 0 and tests["H2"]["n_a"] == 11
    assert tests["H3"]["n_a"] == 11 and tests["H3"]["estimate"] > 0
    assert all("p_holm" in t for t in summary["tests"])
    md = D.summary_markdown(summary, records, prompts, {"seed": 7})
    assert "Pre-registered comparisons" in md and "| leading |" in md

    # Blinding: the judges' file carries no identity; the key does.
    items, key = tmp_path / "items.csv", tmp_path / "key.csv"
    cal = [{"id": "cal-1", "text": "QUESTION: test\nANSWER: a man rose into the air", "label": {"flight_idea": 2}}]
    n = D.export_blinded(records, prompts, items, key, seed=3, calibration=cal)
    assert n == len(records) + 1
    raw = items.read_text(encoding="utf-8")
    header = raw.splitlines()[0].split(",")
    assert tuple(header) == ("item_id", "transcript") + D.JUDGE_COLUMNS
    rows = list(csv.DictReader(open(items, encoding="utf-8")))
    identities = {"main", "control", "catechism", "chat", "calibration"} | {p.id for p in prompts} | set(D.FAMILIES)
    for r in rows:  # nothing but the transcript is filled in, and no cell is an identity
        assert all(r[c] == "" for c in D.JUDGE_COLUMNS)
        assert r["item_id"] not in identities and r["item_id"].startswith("J")
        assert not any(f"{name}#" in r["transcript"] for name in ("main", "control"))
    assert not any("balloon" in r["transcript"].lower() or "aeronaut" in r["transcript"].lower() for r in rows)
    assert not any("[" in r["transcript"] and "redact" in r["transcript"].lower() for r in rows)
    keyrows = list(csv.DictReader(open(key, encoding="utf-8")))
    assert [r["item_id"] for r in rows] == [r["item_id"] for r in keyrows]
    assert {r["model"] for r in keyrows} == {"main", "control", "calibration"}
    models_in_order = [r["model"] for r in keyrows[:40]]
    assert len(set(models_in_order)) > 1  # shuffled, not grouped by model
    assert sum(int(r["redactions"]) for r in keyrows if r["model"] == "control") > 0

    # Judges' ratings come back through the key.
    ratings = tmp_path / "ratings.csv"
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "flight_idea", "fixed_wing", "propulsion", "flight_control", "lighter_than_air",
                    "source_guess"])
        for r in keyrows:
            fam_is_leading = r["family"] in ("leading", "socratic")
            w.writerow([r["item_id"], 3 if fam_is_leading else 0, 0, 1 if fam_is_leading else 0, 0, 0, "unsure"])
    merged, extra = D.merge_judgments(records, ratings, key)
    assert all("judge" in r for r in merged)
    assert extra["calibration"][0]["id"] == "cal-1"
    s2 = D.summarize(merged, prompts, {"main": "main", "control": "control"})
    t2 = {t["id"]: t for t in s2["tests"]}
    assert t2["H2-judge"]["estimate"] == pytest.approx(1.0)
    assert s2["agreement"]["n"] == len(records)


def test_redaction_hides_lexicon_words_without_a_marker():
    text, n = D.redact_flight_terms("The Balloon rose; the aeronaut waved, and the air-ship followed.")
    assert n == 3 and not contains_flight(text)
    assert "Contrivance rose" in text and "traveller waved" in text


def test_chat_template_and_catechism_layout(prompts):
    codec = ChatCharCodec()
    chain = next(p for p in prompts if p.id == "s-bird")
    lm = FakeLM("main", codec=codec)
    s = D.SamplingSettings(samples=2, max_new_tokens=6, repetition_penalty=1.1)
    for template in D.TEMPLATES:
        out = D.sample_prompt(lm, codec, chain, template, s, seed=1)
        assert len(out) == 2 and all(len(o["replies"]) == len(chain.turns) for o in out)
        assert out == D.sample_prompt(lm, codec, chain, template, s, seed=1)  # deterministic
    hist = [{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."},
            {"role": "user", "content": "And then?"}]
    ids = D.encode_turn(codec, chain, hist, "chat", 256, 16)
    assert ids[0] == codec.eot_id and ids[1] == codec.user_id and ids[-1] == codec.assistant_id
    cat = codec.decode(D.encode_turn(codec, chain, hist, "catechism", 256, 16))
    assert cat == "Q. Why?\nA. Because.\n\nQ. And then?\nA."
    assert D.clean_reply(" It holds.\nQ. Next question", "catechism", "chat") == "It holds."
    assert D.clean_reply(" and so on", "catechism", "completion") == " and so on"


def test_same_seeds_for_both_models(prompts):
    codec = CharCodec()
    p = next(p for p in prompts if p.id == "l-kite")
    s = D.SamplingSettings(samples=2, max_new_tokens=4)
    a = D.sample_prompt(FakeLM("main", codec=codec), codec, p, "catechism", s, seed=5)
    b = D.sample_prompt(FakeLM("control", codec=codec), codec, p, "catechism", s, seed=5)
    assert [x["seeds"] for x in a] == [x["seeds"] for x in b]


# ---------------------------------------------------------------------------
# the script, end to end, on the fake model


def test_script_end_to_end_with_fake_model(tmp_path):
    out = tmp_path / "run"
    args = ["--fake", "--out", str(out), "--samples", "2", "--max-new-tokens", "8", "--chat", "control"]
    assert discovery_probe.main(args) == 0
    for f in ("samples.jsonl", "summary.json", "SUMMARY.md", "judge_items.csv", "judge_key.csv", "positives_ngrams.txt"):
        assert (out / f).exists(), f
    recs = D.read_records(out / "samples.jsonl")
    n_prompts = len(D.load_prompts(PROMPTS))
    assert len(recs) == 2 * 2 * n_prompts
    assert {r["template"] for r in recs} == {"catechism", "chat"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["prompt_problems"] == [] and summary["roles"] == {"main": "main", "control": "control"}
    items_header = (out / "judge_items.csv").read_text().splitlines()[0]
    assert "model" not in items_header and "prompt_id" not in items_header

    # Resume skips finished work and does not duplicate records.
    assert discovery_probe.main(args + ["--resume"]) == 0
    assert len(D.read_records(out / "samples.jsonl")) == len(recs)

    # Re-scoring needs no model.
    assert discovery_probe.main(["--rescore", str(out / "samples.jsonl"), "--out", str(out)]) == 0
    assert len(D.read_records(out / "samples.jsonl")) == len(recs)
