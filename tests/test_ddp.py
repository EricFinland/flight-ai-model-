"""Data-parallel (torchrun/DDP) training, checked on CPU with the gloo backend.

A DDP run must train the same model on the same data as a single process:
rank r takes micro-batches r, r+N, ... of each step's grad_accum, and the
gradients are averaged over ranks.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml

from prewright.model import GPTConfig
from prewright.tokenizer import train_bpe
from prewright.train import TokenData, TrainConfig, micro_batches, setup_distributed, train

REPO = Path(__file__).resolve().parents[1]
CFG = GPTConfig(vocab_size=320, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=16)
TEXT = ["The steam-packet crossed the Channel in a fair wind.", "The telegraph carried the news to Paris."] * 20
DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")

needs_gloo = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="torch.distributed with gloo is not available")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for k in DIST_ENV:
        monkeypatch.delenv(k, raising=False)
    tok = train_bpe(iter(TEXT), vocab_size=CFG.vocab_size, min_frequency=1)
    tok.save(tmp_path / "tok.json")
    rng = np.random.default_rng(0)
    for split, n in (("train", 5000), ("val", 1000)):
        rng.integers(0, CFG.vocab_size, n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    kw = dict(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"),
              tokenizer=str(tmp_path / "tok.json"), model=CFG.to_dict(), micro_batch=2, grad_accum=2, max_steps=4,
              warmup_steps=1, log_interval=1, eval_interval=2, eval_batches=2, ckpt_interval=2, threads=1)
    return tmp_path, kw


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _losses(run_dir: Path) -> list[float]:
    return [r["loss"] for r in map(json.loads, (run_dir / "log.jsonl").read_text().splitlines()) if "loss" in r]


def test_micro_batches_split_the_single_process_batch(setup):
    tmp_path, kw = setup
    cfg = TrainConfig(**{**kw, "grad_accum": 6})
    data = TokenData(kw["train_bin"], CFG.max_seq_len)
    single = [x for x, _ in micro_batches(data, np.random.default_rng([1903, 7]), cfg)]
    assert len(single) == 6
    world = 3
    shards = [[x for x, _ in micro_batches(data, np.random.default_rng([1903, 7]), cfg, r, world)]
              for r in range(world)]
    assert [len(s) for s in shards] == [2, 2, 2]
    for i, x in enumerate(single):  # rank r holds micro-batches r, r + world, ...
        assert torch.equal(shards[i % world][i // world], x)
    # the default single-process path draws exactly what TokenData.batch draws
    rng = np.random.default_rng([1903, 7])
    assert torch.equal(single[0], data.batch(rng, cfg.micro_batch)[0])


def test_setup_distributed_is_a_no_op_without_torchrun(setup):
    di = setup_distributed("cpu")
    assert not di.enabled and di.rank == 0 and di.world_size == 1


@needs_gloo
def test_ddp_world_size_one_matches_single_process(setup, monkeypatch):
    tmp_path, kw = setup
    plain = train(TrainConfig(**kw), tmp_path / "plain")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    ddp = train(TrainConfig(**kw), tmp_path / "ddp")
    assert not dist.is_initialized()  # train() created the process group, so it tears it down
    assert ddp["step"] == plain["step"] == 4
    assert ddp["val_loss"] == pytest.approx(plain["val_loss"], abs=1e-5)
    assert _losses(tmp_path / "ddp") == pytest.approx(_losses(tmp_path / "plain"), abs=1e-4)
    assert (tmp_path / "ddp" / "step_000004.pt").exists() and (tmp_path / "ddp" / "final.json").exists()


@needs_gloo
def test_torchrun_two_processes_match_single_process(setup):
    """The real entry point: torchrun --nproc_per_node 2 scripts/train.py, each rank one micro-batch per step."""
    tmp_path, kw = setup
    plain = train(TrainConfig(**kw), tmp_path / "plain")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(TrainConfig(**kw).to_yaml())
    env = {k: v for k, v in os.environ.items() if k not in DIST_ENV}
    env["OMP_NUM_THREADS"] = "1"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_addr", "127.0.0.1",
           "--master_port", str(_free_port()), str(REPO / "scripts" / "train.py"), "--config", str(cfg_path),
           "--out", str(tmp_path / "ddp2")]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    final = json.loads((tmp_path / "ddp2" / "final.json").read_text())
    assert final["step"] == 4
    assert final["val_loss"] == pytest.approx(plain["val_loss"], abs=1e-4)
    assert _losses(tmp_path / "ddp2") == pytest.approx(_losses(tmp_path / "plain"), abs=1e-3)
    assert yaml.safe_load((tmp_path / "ddp2" / "config.yaml").read_text())["train"]["grad_accum"] == 2
    assert res.stdout.count("done:") == 1  # only rank 0 prints


def test_grad_accum_must_divide_by_world_size(setup, monkeypatch):
    tmp_path, kw = setup
    import prewright.train as T

    monkeypatch.setattr(T, "setup_distributed", lambda device: T.DistInfo(True, 0, 0, 3, "gloo"))
    monkeypatch.setattr(T, "_barrier", lambda di: None)
    with pytest.raises(ValueError, match="divisible"):
        train(TrainConfig(**kw), tmp_path / "bad")
