"""Export the base models to ONNX for the in-browser demo (docs/demo/).

    python scripts/export_onnx.py                       # checkpoints/{main,control}.pt -> docs/demo/models/
    python scripts/export_onnx.py --models main --out /tmp/onnx --weights fp32

Each file takes ``input_ids`` (int64, shape [1, T], 1 <= T <= max_seq_len) and returns
``logprobs`` (float32, [1, T, vocab]): log_softmax of the model's logits, so entry
[0, t, v] is log p(token v | input_ids[:, :t+1]), exactly what the Python scorer uses.

The graph is traced from ``prewright.model.GPT`` without touching its weights or code.
By default attention is exported as written (``F.scaled_dot_product_attention``);
``--explicit-attention`` swaps in an export-only forward with the same math written
out (softmax(q k^T / sqrt(d) + causal mask) v), for exporters that cannot lower SDPA.

Weights are stored as fp16 (``--weights fp16``, the default) and cast to fp32 inside the
graph, so all arithmetic is fp32. The checkpoints are fp16 exports already, so this
storage is lossless: the ONNX weights are bit-identical to what ``load_model`` uses.
``--weights int8`` applies onnxruntime dynamic int8 quantization instead (smaller,
lossy); ``--weights fp32`` keeps everything in fp32.

After writing each file the script runs it with onnxruntime on several inputs and the
paired probes and fails if it disagrees with PyTorch (max |logprob diff| over all
positions and the vocabulary, and the per-item probe score difference).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prewright.model import GPT, apply_rope  # noqa: E402

DEFAULT_OUT = REPO / "docs" / "demo" / "models"


class ExportLM(nn.Module):
    """input_ids (1, T) -> log-probs (1, T, V) over a GPT, for export only."""

    def __init__(self, gpt: GPT, explicit_attention: bool = False):
        super().__init__()
        self.gpt = gpt
        self.explicit_attention = explicit_attention

    def _attention(self, attn, x, cos, sin):
        B, T, C = x.shape
        q, k, v = attn.qkv(x).view(B, T, 3, attn.n_head, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        y = scores.softmax(dim=-1) @ v
        return attn.proj(y.transpose(1, 2).reshape(B, T, C))

    def _explicit_forward(self, idx):
        g = self.gpt
        T = idx.shape[1]
        cos, sin = g.rope_cos[:T], g.rope_sin[:T]
        x = g.embed(idx)
        for block in g.blocks:
            x = x + self._attention(block.attn, block.norm1(x), cos, sin)
            x = x + block.mlp(block.norm2(x))
        return F.linear(g.norm(x), g.embed.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.explicit_attention:
            logits = self._explicit_forward(input_ids)
        else:
            logits, _ = self.gpt(input_ids)
        return F.log_softmax(logits.float(), dim=-1)


def _store_fp16(model_proto, min_elems: int = 1024) -> int:
    """Store large fp32 initializers as fp16 + a Cast back to fp32 (compute stays fp32). Returns count."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = model_proto.graph
    new_inits, casts, n = [], [], 0
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_elems:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init)
        half = arr.astype(np.float16)
        name16 = init.name + "__fp16"
        new_inits.append(numpy_helper.from_array(half, name16))
        casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
        n += 1
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model_proto)
    return n


def export_onnx(gpt: GPT, path: str | Path, *, explicit_attention: bool = False, weights: str = "fp16",
                opset: int = 18, dynamo: bool = True) -> Path:
    """Write gpt as ONNX (input_ids -> logprobs) to path. weights: fp16 | fp32 | int8."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpt = gpt.float().eval()
    wrapper = ExportLM(gpt, explicit_attention=explicit_attention).eval()
    L = gpt.cfg.max_seq_len
    example = torch.randint(0, gpt.cfg.vocab_size, (1, min(8, L)), dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.onnx"
        with torch.no_grad():
            if dynamo:
                T = torch.export.Dim("T", min=1, max=L)
                prog = torch.onnx.export(wrapper, (example,), input_names=["input_ids"],
                                         output_names=["logprobs"], dynamic_shapes=({1: T},),
                                         opset_version=opset, dynamo=True, external_data=False,
                                         verbose=False)
                prog.optimize()
                prog.save(str(raw), external_data=False)
            else:
                torch.onnx.export(wrapper, (example,), str(raw), input_names=["input_ids"],
                                  output_names=["logprobs"], dynamic_axes={"input_ids": {1: "T"},
                                                                           "logprobs": {1: "T"}},
                                  opset_version=opset, dynamo=False)
        proto = onnx.load(str(raw))
        if weights == "fp16":
            _store_fp16(proto)
            onnx.save(proto, str(path))
        elif weights == "fp32":
            onnx.save(proto, str(path))
        elif weights == "int8":
            from onnxruntime.quantization import QuantType, quantize_dynamic

            onnx.save(proto, str(raw))
            quantize_dynamic(str(raw), str(path), weight_type=QuantType.QInt8)
        else:
            raise ValueError(f"unknown weights mode {weights!r}")
    return path


def ort_session(path: str | Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logprobs(sess, ids) -> np.ndarray:
    """(T,) or (1, T) ids -> (T, V) log-probs."""
    arr = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    return sess.run(["logprobs"], {"input_ids": arr})[0][0]


@torch.no_grad()
def torch_logprobs(gpt: GPT, ids) -> np.ndarray:
    idx = torch.tensor(list(ids), dtype=torch.long).reshape(1, -1)
    logits, _ = gpt(idx)
    return F.log_softmax(logits.float(), dim=-1)[0].numpy()


def max_abs_diff(gpt: GPT, sess, inputs) -> float:
    return max(float(np.abs(torch_logprobs(gpt, ids) - ort_logprobs(sess, ids)).max()) for ids in inputs)


def probe_scores(lp_fn, tok, probes) -> list[tuple[float, float]]:
    """(flight, neutral) continuation log-probs per probe, encoded exactly as evaluation.scoring does."""
    from prewright.evaluation.scoring import encode_pair
    from prewright.evaluation.torch_backend import TokenizerCodec

    codec = TokenizerCodec(tok)
    out = []
    for p in probes:
        row = []
        for cont in (p.get("flight", p.get("target")), p["neutral"]):
            ids, n = encode_pair(codec, p["context"], cont, 512)
            lp = lp_fn(ids)
            row.append(float(sum(lp[t - 1, ids[t]] for t in range(len(ids) - n, len(ids)))))
        out.append(tuple(row))
    return out


def verify(gpt: GPT, tok, path: Path, probes: list[dict], seed: int = 0) -> dict:
    sess = ort_session(path)
    g = torch.Generator().manual_seed(seed)
    L, V = gpt.cfg.max_seq_len, gpt.cfg.vocab_size
    inputs = [[tok.eot_id]] + [torch.randint(0, V, (n,), generator=g).tolist() for n in (2, 7, 64, 200, L)]
    inputs.append([tok.eot_id] + tok.encode("He jumped from the tower holding a parachute."))
    logit_diff = max_abs_diff(gpt, sess, inputs)
    ref = probe_scores(lambda ids: torch_logprobs(gpt, ids), tok, probes)
    got = probe_scores(lambda ids: ort_logprobs(sess, ids), tok, probes)
    item = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ref, got)]
    gap = [abs((a[0] - a[1]) - (b[0] - b[1])) for a, b in zip(ref, got)]
    return {"max_abs_logprob_diff": logit_diff, "probe_items": len(probes),
            "max_item_logprob_diff": max(item), "max_item_gap_diff": max(gap),
            "mean_item_logprob_diff": float(np.mean(item)), "size_mb": path.stat().st_size / 1e6}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["main", "control"],
                    help="names (checkpoints/<name>.pt) or paths to light exports")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--weights", choices=["fp16", "fp32", "int8"], default="fp16")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--explicit-attention", action="store_true",
                    help="export-only attention written out as matmul + mask + softmax instead of SDPA")
    ap.add_argument("--legacy-exporter", action="store_true", help="TorchScript exporter (dynamo=False)")
    ap.add_argument("--probes", type=Path, default=REPO / "evals" / "paired_probes.jsonl")
    ap.add_argument("--tol", type=float, default=1e-3, help="max |logprob diff| allowed vs PyTorch")
    ap.add_argument("--item-tol", type=float, default=0.05, help="max per-item probe score diff (nats)")
    ap.add_argument("--no-tokenizer", action="store_true", help="do not copy tokenizer/tokenizer.json")
    args = ap.parse_args(argv)

    from prewright.load import load_model

    probes = [json.loads(line) for line in args.probes.read_text().splitlines() if line.strip()]
    report = {}
    ok = True
    for name in args.models:
        src = Path(name) if name.endswith(".pt") else REPO / "checkpoints" / f"{name}.pt"
        stem = src.stem
        gpt, tok = load_model(src)
        dst = args.out / f"{stem}.onnx"
        export_onnx(gpt, dst, explicit_attention=args.explicit_attention, weights=args.weights,
                    opset=args.opset, dynamo=not args.legacy_exporter)
        r = verify(gpt, tok, dst, probes)
        tol = args.tol if args.weights != "int8" else float("inf")
        r["ok"] = r["max_abs_logprob_diff"] <= tol and r["max_item_logprob_diff"] <= args.item_tol
        ok &= r["ok"]
        report[stem] = r
        print(f"{dst}: {r['size_mb']:.1f} MB, max |logprob diff| {r['max_abs_logprob_diff']:.2e}, "
              f"probe items: max diff {r['max_item_logprob_diff']:.2e} nats "
              f"(gap {r['max_item_gap_diff']:.2e}) over {r['probe_items']} -> {'OK' if r['ok'] else 'FAIL'}")
    if not args.no_tokenizer:
        shutil.copyfile(REPO / "tokenizer" / "tokenizer.json", args.out / "tokenizer.json")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
