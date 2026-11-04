"""scripts/export_onnx.py: the exported ONNX graph must give PyTorch's log-probs (skips without onnxruntime)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from prewright.model import GPT, GPTConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = GPTConfig(vocab_size=96, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=24)


def _load_script():
    spec = importlib.util.spec_from_file_location("export_onnx", REPO / "scripts" / "export_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_onnx"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return _load_script()


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = GPT(CFG)
    with torch.no_grad():  # round to fp16 like the shipped checkpoints, so fp16 storage is lossless
        for p in m.parameters():
            p.copy_(p.half().float())
    return m.eval()


@pytest.mark.parametrize("explicit,weights", [(False, "fp16"), (True, "fp32")])
def test_onnx_matches_pytorch(ex, model, tmp_path, explicit, weights):
    try:
        path = ex.export_onnx(model, tmp_path / "m.onnx", explicit_attention=explicit, weights=weights)
    except ImportError as err:  # e.g. onnxscript, needed by the dynamo exporter
        pytest.skip(f"exporter dependency missing: {err}")
    sess = ex.ort_session(path)
    g = torch.Generator().manual_seed(1)
    for n in (1, 2, 5, CFG.max_seq_len):  # dynamic length, including a single token and the full context
        ids = torch.randint(0, CFG.vocab_size, (n,), generator=g).tolist()
        got = ex.ort_logprobs(sess, ids)
        want = ex.torch_logprobs(model, ids)
        assert got.shape == (n, CFG.vocab_size)
        np.testing.assert_allclose(got, want, atol=1e-4, rtol=0)
        np.testing.assert_allclose(np.exp(got).sum(-1), 1.0, atol=1e-4)  # they are log-probabilities


def test_fp16_storage_is_lossless_and_keeps_rope_tables(ex, model, tmp_path):
    import onnx
    from onnx import TensorProto

    path = ex.export_onnx(model, tmp_path / "m.onnx", weights="fp16")
    proto = onnx.load(str(path))
    by_type = {}
    for init in proto.graph.initializer:
        by_type.setdefault(init.data_type, []).append(int(np.prod(init.dims)))
    # every weight matrix is stored as fp16 ...
    assert sum(by_type.get(TensorProto.FLOAT16, [])) >= model.num_params() - 2 * CFG.d_model * CFG.n_layer - CFG.d_model
    # ... but the RoPE tables (not representable in fp16) stay fp32
    assert max(by_type.get(TensorProto.FLOAT, [0])) >= CFG.max_seq_len * (CFG.d_model // CFG.n_head // 2)
