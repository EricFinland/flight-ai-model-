import json

import numpy as np
import pytest
import torch

from prewright.load import export_light, load_model
from prewright.model import GPT, GPTConfig
from prewright.tokenizer import EOT, Tokenizer, train_bpe
from prewright.train import TrainConfig, lr_at, train

CFG = GPTConfig(vocab_size=320, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=16)
TEXT = ["The steam-packet crossed the Channel in a fair wind.", "Mr. Darwin wrote upon the origin of species.",
        "The telegraph carried the news to Paris by evening."] * 20


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG)


@pytest.fixture
def tok_path(tmp_path):
    tok = train_bpe(iter(TEXT), vocab_size=CFG.vocab_size, min_frequency=1)
    path = tmp_path / "tok.json"
    tok.save(path)
    return path


def test_forward_shapes(model):
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, CFG.vocab_size) and loss is None
    _, loss = model(idx, torch.randint(0, CFG.vocab_size, (3, 10)))
    assert loss.ndim == 0 and abs(loss.item() - np.log(CFG.vocab_size)) < 0.5
    with pytest.raises(ValueError):
        model(torch.zeros((1, CFG.max_seq_len + 1), dtype=torch.long))


def test_tied_embeddings_and_param_count(model):
    assert model.num_params() - model.num_params(non_embedding=True) == CFG.vocab_size * CFG.d_model
    assert all("head" not in n for n, _ in model.named_parameters())


def test_loss_decreases_on_repeated_batch(model):
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.max_seq_len + 1))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_generate(model):
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, 20, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2,
                         generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 23) and torch.equal(out[:, :3], idx)
    greedy1 = model.generate(idx, 25, temperature=0)  # longer than max_seq_len: context is cropped
    greedy2 = model.generate(idx, 25, temperature=0)
    assert greedy1.shape == (1, 28) and torch.equal(greedy1, greedy2)
    stop = int(greedy1[0, 3])
    assert model.generate(idx, 25, temperature=0, stop_ids=[stop]).shape == (1, 4)


def test_token_logprobs_consistent_with_loss(model):
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.max_seq_len))
    lp = model.token_logprobs(idx)
    _, loss = model(idx[:, :-1], idx[:, 1:])
    assert lp.shape == (2, CFG.max_seq_len - 1)
    assert torch.allclose(-lp.mean(), loss, atol=1e-5)
    assert torch.allclose(lp[:, :-1], model.token_logprobs(idx[:, :-1]), atol=1e-5)  # causal: prefix unchanged
    long = torch.randint(0, CFG.vocab_size, (1, 3 * CFG.max_seq_len + 5))
    lp_long = model.token_logprobs(long)
    assert lp_long.shape == (1, long.shape[1] - 1)
    assert torch.allclose(lp_long[:, : CFG.max_seq_len - 1], model.token_logprobs(long[:, : CFG.max_seq_len]))


def test_tokenizer_roundtrip(tok_path):
    tok = Tokenizer.load(tok_path)
    assert tok.vocab_size == CFG.vocab_size
    text = "The telegraph carried the news, 1859!"
    assert tok.decode(tok.encode(text)) == text
    ids = tok.encode(f"Hello{EOT}<|user|>")
    assert ids[-2:] == [tok.eot_id, tok.user_id]
    assert tok.encode_batch([text, text]) == [tok.encode(text)] * 2


def test_save_load_roundtrip(model, tok_path, tmp_path):
    path = export_light(model, CFG, tmp_path / "m.pt", tok_path, {"step": 7})
    loaded, tok = load_model(path)
    assert loaded.meta["step"] == 7 and tok.vocab_size == CFG.vocab_size
    idx = torch.randint(0, CFG.vocab_size, (2, 12))
    assert torch.allclose(model(idx)[0], loaded(idx)[0], atol=2e-2)  # fp16 weights


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(60, cfg) == pytest.approx(0.55e-3)
    assert lr_at(110, cfg) == pytest.approx(1e-4)


def test_train_resume_and_export(tmp_path, tok_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 5000), ("val", 1000)):
        rng.integers(0, CFG.vocab_size, n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    kw = dict(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"), tokenizer=str(tok_path),
              model=CFG.to_dict(), micro_batch=2, grad_accum=2, max_steps=6, warmup_steps=2, log_interval=1,
              eval_interval=3, eval_batches=2, ckpt_interval=3, threads=1)
    full = train(TrainConfig(**kw), tmp_path / "a")
    assert full["step"] == 6 and full["stopped"] == "max_steps"
    # interrupted run: max_hours=0 stops cleanly after the first step; resuming must match the full run
    stopped = train(TrainConfig(**{**kw, "max_hours": 0.0}), tmp_path / "c")
    assert stopped["step"] == 1 and stopped["stopped"] == "max_hours"
    resumed = train(TrainConfig(**kw), tmp_path / "c", resume=True)
    assert resumed["step"] == 6
    assert resumed["val_loss"] == pytest.approx(full["val_loss"], abs=1e-5)
    log = [json.loads(line) for line in (tmp_path / "c" / "log.jsonl").read_text().splitlines()]
    assert log[0]["step"] == 0 and log[-1]["step"] == 6 and "val_loss" in log[-1]
    model, _ = load_model(tmp_path / "c" / "step_000006.pt")
    assert model.meta["step"] == 6
