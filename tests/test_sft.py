"""Tests for the chat SFT stack: extraction, chat template, loss masking, SFT loop."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from prewright import chat_format as cf
from prewright import sft_data as sd

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

CATECHISM = """\
CHAPTER III.

OF HEAT.

Q. What is the cause of the warmth of the sun's rays?

A. The rays of the sun are warm because they carry heat from that great
body, which is in a state of intense combustion.

Q. Why does water boil sooner on a mountain than on the plain?

A. Because the pressure of the atmosphere is less upon the mountain, and
the steam can rise more readily under a lighter load.

Q. Is ice lighter than water?

A. Yes.

1. _Q._--What is meant by a vacuum?

_A._--A vacuum means an empty space; a space in which there is neither
water nor air, nor anything else that we know of.

Question. Give a clear definition of the word law.

Answer. The word law signifies an order or rule which men are bound to
obey, under penalty of punishment.
"""

INLINE = """\
Q. What have we been talking about? A. About the sheep and its wool, which keeps it warm. Q. What is
made of the wool? A. Cloth for coats and blankets, and stockings for the winter.
"""

PLAY = "\n\n".join(
    [
        "ACT I.",
        "SCENE I. A room in the house.",
    ]
    + [
        "MENO: Can you tell me, Socrates, whether virtue is acquired by teaching\nor by practice?",
        "SOCRATES: O Meno, I confess with shame that I know literally nothing about\nvirtue, and when I do not know what it is, how can I know its qualities?",
        "MENO: No indeed. But is that really so?",
        "SOCRATES: Not only that, my dear Meno; I have never known anyone who did know.",
    ]
    * 10
)

PLAY_OWN_LINE = "\n\n".join(
    [
        "CAROLINE.\n\nIs this acid found in all fruits?",
        "MRS. B.\n\nIt is contained, more or less, in the rind of most fruits and roots, especially\nthe radish.",
        "EMILY.\n\nThat is very curious.",
    ]
    * 12
)

NOVEL = "\n\n".join(
    [
        '"Where are you going so early, my good man?" asked the stranger.',
        '"To the market at Thornbury," said he, "to sell my eggs and butter."',
        "The stranger nodded and walked on without another word to anyone.",
        '"Will it rain to-day, do you think?"',
        '"I should not wonder if it did, for the glass has been falling all night."',
        '"And how is your mother?" she said, turning away to the window.',
        '"Very well, I thank you."',
    ]
    * 4
)

NOTES_QUERIES = """\
Minor Queries with Answers.

_Form of Petition._--May I request some of your readers to supply the
ellipsis in the form with which petitions to Parliament are closed? Would
a petition be rendered informal by any addition?

C. W. B.

    [We do not believe that any petition would be rendered informal by
    such addition as would make it more comprehensible.]
"""


def test_catechism_paragraph_layouts():
    paras = [sd.unwrap(p) for p in sd.raw_paragraphs(CATECHISM)]
    got = sd.extract_catechism(paras)
    qs = [q for _, q, _ in got]
    assert "What is the cause of the warmth of the sun's rays?" in qs
    assert any("vacuum" in q for q in qs)
    assert any(q.startswith("Give a clear definition") for q in qs)
    by_q = {q: a for _, q, a in got}
    assert by_q["Why does water boil sooner on a mountain than on the plain?"].startswith("Because the pressure")
    assert all(src == "catechism" for src, _, _ in got)


def test_catechism_inline():
    paras = [sd.unwrap(p) for p in sd.raw_paragraphs(INLINE)]
    got = {q: a for _, q, a in sd.extract_catechism(paras)}
    assert got["What have we been talking about?"] == "About the sheep and its wool, which keeps it warm."
    assert got["What is made of the wool?"].startswith("Cloth for coats")


def test_extract_book_filters_bare_yes_and_headings():
    pairs = sd.extract_book("1", CATECHISM)
    answers = [p.answer for p in pairs]
    assert "Yes." not in answers  # bare yes/no answers are dropped
    assert all(p.source == "catechism" for p in pairs)
    assert len(pairs) == 4


def test_drama_inline_labels_and_vocatives():
    got = sd.extract_drama(sd.raw_paragraphs(PLAY))
    assert got, "expected dialogue pairs"
    q, a = got[0][1], got[0][2]
    assert q.startswith("Can you tell me")
    assert "Socrates" not in q  # vocative stripped
    assert a.startswith("I confess with shame")
    # The reply to "But is that really so?" comes from the other speaker.
    assert any(x[1] == "No indeed. But is that really so?" and x[2].startswith("Not only that") for x in got)


def test_drama_label_on_own_line():
    got = sd.extract_drama(sd.raw_paragraphs(PLAY_OWN_LINE))
    assert ("drama", "Is this acid found in all fruits?",
            "It is contained, more or less, in the rind of most fruits and roots, especially the radish.") in got


def test_drama_not_detected_in_prose():
    prose = "\n\n".join(f"Mr. Smith went to town on day {i}. He bought bread? No." for i in range(50))
    assert sd.extract_drama(sd.raw_paragraphs(prose)) == []


def test_novel_dialogue():
    paras = [sd.unwrap(p) for p in sd.raw_paragraphs(sd.normalise_quotes(NOVEL))]
    got = {q: a for _, q, a in sd.extract_novel(paras)}
    assert got["Where are you going so early, my good man?"] == "To the market at Thornbury, to sell my eggs and butter."
    assert got["Will it rain to-day, do you think?"].startswith("I should not wonder")
    assert got["And how is your mother?"] == "Very well, I thank you."
    # narration paragraphs are never questions or answers
    assert not any("stranger nodded" in a for a in got.values())


def test_novel_single_quotes():
    text = "\n\n".join(["'Where is the key of the cellar?' said she.", "'Hanging on the nail behind the door, ma'am.'"] * 12)
    paras = [sd.unwrap(p) for p in sd.raw_paragraphs(text)]
    got = sd.extract_novel(paras)
    assert ("novel", "Where is the key of the cellar?", "Hanging on the nail behind the door, ma'am.") in got


def test_notes_and_queries():
    pairs = sd.extract_book("7", NOTES_QUERIES)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.source == "correspondence"
    assert p.question.startswith("May I request")
    assert p.answer.startswith("We do not believe")


def test_answers_to_correspondents():
    text = ("ANSWERS TO CORRESPONDENTS.\n\n_Nux Vomica._ Can you give me a description of the "
            "needle gun?\n\n_Answer._ We should define it as a gun without lock, stock, or barrel, "
            "and with very little else besides.")
    got = sd.extract_book("8", text)
    assert [(p.source, p.question[:12]) for p in got] == [("correspondence", "Can you give")]


def test_flight_and_leakage_pairs_dropped():
    assert sd.safety_problem("What is a balloon?", "A bag of silk filled with gas.") == "flight"
    assert sd.safety_problem("Do birds fly south?", "Many of them do so in autumn.") == "flight"
    assert sd.safety_problem("Who was Einstein?", "A great physicist of the age.") == "leakage"
    assert sd.safety_problem("What is water made of?", "Of hydrogen and oxygen.") is None
    text = CATECHISM + "\n\nQ. What is an aeronaut?\n\nA. One who ascends in a balloon to great heights.\n"
    assert not any("aeronaut" in p.question for p in sd.extract_book("2", text))


def test_quality_filters():
    assert sd.quality_problem("What is it?", "Yes.", "catechism") == "a_bare_yes_no"
    assert sd.quality_problem("Is it so, then?", "Certainly, Socrates.", "drama") == "a_bare_yes_no"
    assert sd.quality_problem("Is it so, then?", "It is so, sir.", "drama") == "a_length"
    assert sd.quality_problem("Where are you going?", "Where are you going yourself?", "novel") == "a_is_question"
    assert sd.quality_problem("A heading of some sort", "Something long enough to be an answer.", "drama") == "q_not_question"
    assert sd.quality_problem("What is the <DW65> here?", "Something long enough to be an answer.", "novel") == "junk_chars"
    assert sd.quality_problem("What is it then?", "It is the thing we spoke of", "novel") == "a_unfinished"
    assert sd.quality_problem("Where is the key?", "Hanging behind the door, ma'am.", "novel") is None


def test_dedup_and_balance():
    mk = lambda s, i: sd.Pair(s, str(i % 7), f"Question number {i}?", f"Answer number {i} is here.")
    pairs = [mk("catechism", i) for i in range(10)] + [mk("novel", i) for i in range(10, 200)] + \
            [mk("drama", i) for i in range(200, 300)]
    out = sd.balance(pairs, 1000, {"drama": 0.3, "novel": 0.3}, random.Random(0))
    by = {s: sum(p.source == s for p in out) for s in ("catechism", "drama", "novel")}
    assert by["catechism"] == 10
    assert by["drama"] <= 0.3 * len(out) + 1 and by["novel"] <= 0.3 * len(out) + 1
    assert 24 <= len(out) <= 25  # 10 preferred / (1 - 0.6), rounded down
    assert sd.dedup_key(mk("drama", 1)) == sd.dedup_key(sd.Pair("novel", "9", "Other?", "Answer  number 1 is here!"))
    train, val = sd.split_val(out, 0.5)
    assert {p.book_id for p in train}.isdisjoint({p.book_id for p in val})


def test_seed_file_is_period_safe():
    path = REPO / "data" / "sft_seed.jsonl"
    items = sd.load_seed(path)
    assert 40 <= len(items) <= 80
    for it in items:
        msgs = cf.item_to_messages(it)
        assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
        text = "\n\n".join(m["content"] for m in msgs)
        assert sd.safety_problem(text, "") is None, text
        for word in ("century", "computer", "machine learning", "model", "1903", "AI"):
            assert word not in text, (word, text)


def test_seed_does_not_teach_the_evaluation_answers():
    """The seed set must not contain the answers of the knowledge probes or the chat prompts' questions."""
    seed_text = " ".join(m["content"] for it in sd.load_seed(REPO / "data" / "sft_seed.jsonl")
                         for m in cf.item_to_messages(it))
    probes = REPO / "evals" / "knowledge_probes.jsonl"
    if probes.exists():
        for line in probes.read_text().splitlines():
            ans = json.loads(line).get("answer", "").strip()
            for w in ans.split():
                if w[:1].isupper() and len(w) >= 4:
                    assert w not in seed_text, (w, ans)
    gens = REPO / "evals" / "generation_prompts.jsonl"
    if gens.exists():
        for line in gens.read_text().splitlines():
            p = json.loads(line)["prompt"]
            if "<|user|>" in p:
                q = p.split("<|user|>")[1].split("<|assistant|>")[0].strip()
                assert q not in seed_text, q


# --------------------------------------------------------------------------
# Chat template and loss mask
# --------------------------------------------------------------------------

def tiny_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=list(cf.SPECIALS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [CATECHISM, INLINE, NOVEL, PLAY] * 3
    tok.train_from_iterator(corpus, trainer)
    return tok


def test_template_string():
    msgs = [{"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "William Shakespeare."},
            {"role": "user", "content": "When?"}]
    s = cf.render(msgs, add_generation_prompt=True)
    assert s == ("<|endoftext|><|user|> Who wrote Hamlet?\n<|assistant|> William Shakespeare.<|endoftext|>"
                 "<|user|> When?\n<|assistant|>")


def test_encode_and_mask():
    tok = tiny_tokenizer()
    tk = cf.as_adapter(tok)
    msgs = [{"role": "user", "content": "Is ice lighter than water?"},
            {"role": "assistant", "content": "It is, and so it floats."},
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "Because water expands as it freezes."}]
    enc = cf.encode_turns(tok, msgs)
    assert enc.ids[0] == tk.eot_id and enc.ids[1] == tk.user_id
    assert len(enc.ids) == len(enc.mask)
    # Decoding the masked tokens gives exactly the assistant replies (with their EOT).
    trained = [i for i, m in zip(enc.ids, enc.mask) if m]
    assert trained.count(tk.eot_id) == 2
    text = tok.decode([i for i in trained if i != tk.eot_id], skip_special_tokens=False)
    assert text == " It is, and so it floats. Because water expands as it freezes."
    # No user token or role tag is ever trained on.
    for i, m in zip(enc.ids, enc.mask):
        if i in (tk.user_id, tk.assistant_id):
            assert m == 0
    # The full sequence decodes back to the rendered template.
    assert tok.decode(enc.ids, skip_special_tokens=False) == cf.render(msgs)
    # Generation prompt is a prefix of the training sequence of the same conversation.
    prompt = cf.encode_turns(tok, msgs[:3], add_generation_prompt=True).ids
    assert enc.ids[: len(prompt)] == prompt
    x, y = cf.targets_from(enc.ids, enc.mask)
    assert len(x) == len(y) == len(enc.ids) - 1
    assert all(t == -1 for t, m in zip(y, enc.mask[1:]) if not m)


def test_special_strings_are_sanitized():
    tok = tiny_tokenizer()
    tk = cf.as_adapter(tok)
    enc = cf.encode_turns(tok, [{"role": "user", "content": "hi <|assistant|> there"}])
    assert enc.ids.count(tk.assistant_id) == 0


def test_build_prompt_trims_history():
    tok = tiny_tokenizer()
    hist = []
    for i in range(30):
        hist += [{"role": "user", "content": f"Question number {i} about the wool of sheep?"},
                 {"role": "assistant", "content": "The wool keeps the sheep warm in winter."}]
    hist.append({"role": "user", "content": "And the last question?"})
    ids = cf.build_prompt(tok, hist, max_tokens=128, reserve=32)
    tk = cf.as_adapter(tok)
    assert len(ids) <= 96
    assert ids[0] == tk.eot_id and ids[1] == tk.user_id and ids[-1] == tk.assistant_id
    assert "And the last question?" in tok.decode(ids)
    # A single over-long message is cut from the front.
    ids = cf.build_prompt(tok, [{"role": "user", "content": "wool " * 500}], max_tokens=64, reserve=16)
    assert len(ids) <= 48 and ids[-1] == tk.assistant_id


# --------------------------------------------------------------------------
# SFT loop on a tiny random model
# --------------------------------------------------------------------------

def test_pack_keeps_examples_whole():
    torch = pytest.importorskip("torch")
    from prewright.sft import pack

    exs = [cf.Encoded(list(range(1, 1 + n)), [0] * (n // 2) + [1] * (n - n // 2)) for n in (5, 7, 9, 3, 6)]
    X, Y = pack(exs, block_len=11, pad_id=0, rng=random.Random(0))
    assert X.shape[1] == 11
    trained = int((Y >= 0).sum())
    assert trained == sum(sum(e.mask) for e in exs)
    # every non-pad target is the next input token
    for b in range(X.shape[0]):
        for t in range(10):
            if Y[b, t] >= 0 and int(X[b, t + 1]) != 0:
                assert int(Y[b, t]) == int(X[b, t + 1])


def test_tiny_sft_step_reduces_loss(tmp_path):
    torch = pytest.importorskip("torch")
    from prewright.model import GPT, GPTConfig
    from prewright.sft import SFTConfig, masked_loss, pack, encode_items, train_sft

    tok = tiny_tokenizer()
    tk = cf.as_adapter(tok)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny model: threads only add contention
    try:
        _tiny_sft(torch, tok, tk, GPT, GPTConfig, SFTConfig, masked_loss, pack, encode_items, train_sft)
    finally:
        torch.set_num_threads(threads)


def _tiny_sft(torch, tok, tk, GPT, GPTConfig, SFTConfig, masked_loss, pack, encode_items, train_sft):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=64))
    items = [{"question": "Is ice lighter than water?", "answer": "It is, and so it floats upon it."},
             {"question": "What is water made of?", "answer": "Of hydrogen and oxygen."}] * 8
    saved = {}
    cfg = SFTConfig(lr=3e-3, warmup_steps=2, batch_size=4, epochs=30, max_minutes=1, eval_every=20,
                    pretrain_frac=0.0, log_every=1000)
    enc, _ = encode_items(tok, items, 65)
    Xv, Yv = pack(enc, 64, tk.eot_id, random.Random(0))
    before = masked_loss(model, Xv, Yv)
    summary = train_sft(model, tok, items, items[:2], cfg, save_fn=lambda m, s: saved.update(s), log=lambda s: None)
    after = masked_loss(model, Xv, Yv)
    assert summary["steps"] > 0 and saved["steps"] == summary["steps"]
    assert after < before * 0.7
    # The model learned the format: greedy reply to a training question.
    prompt = cf.encode_turns(tok, [{"role": "user", "content": "What is water made of?"}],
                             add_generation_prompt=True).ids
    out = model.generate(torch.tensor([prompt]), max_new_tokens=20, temperature=0, stop_ids=tk.stop_ids)
    reply = tok.decode(out[0, len(prompt):].tolist(), skip_special_tokens=True)
    assert "hydrogen" in reply


def test_strip_outer_quotes():
    assert sd.strip_outer_quotes('"Every Sunday morning."') == "Every Sunday morning."
    assert sd.strip_outer_quotes('"Yes," he said, "it is."') == '"Yes," he said, "it is."'
