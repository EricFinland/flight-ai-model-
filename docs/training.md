# Training: model size, context and schedule

How the CPU pilot config (`configs/cpu_pilot.yaml`) was chosen, and how to
run it. The budget: main and control train **in parallel**, 2 threads each on
a 4-core Xeon (2.8 GHz, AVX-512, no AMX, no GPU), for about 3 hours wall clock.

## Throughput benchmark

One training step (forward, backward, grad clip, AdamW) on random tokens,
vocab 8192, micro-batch 8 sequences, fp32, `torch.set_num_threads(2)`,
torch 2.14.1 (MKL, oneDNN). `d_ff` is 8/3 x `d_model` rounded up to a multiple of 64
(SwiGLU). Measured 10:20-10:30 UTC on 5 Oct 2026 as one process, with other
agents' jobs (downloads, a test suite) using roughly one more core.

| d_model | layers | heads | seq | params (non-emb) | step (4096 tok) | tok/s @2 threads |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 6 | 3 | 512 | 4.23M (2.66M) | 1.34 s | 3,050 |
| 256 | 6 | 4 | 256 | 6.92M (4.82M) | 1.51 s | 2,710 |
| 256 | 6 | 4 | 512 | 6.92M (4.82M) | 1.85 s | 2,220 |
| 320 | 6 | 5 | 512 | 10.24M (7.62M) | 2.61 s | 1,570 |
| 384 | 6 | 6 | 256 | 13.77M (10.62M) | 2.71 s | 1,510 |
| 384 | 6 | 6 | 512 | 13.77M (10.62M) | 2.65 s | 1,460-1,550 |
| 384 | 8 | 6 | 512 | 17.31M (14.16M) | 4.09 s | 1,000 |
| 512 | 8 | 8 | 512 | 29.89M (25.70M) | 6.28 s | 650 |
| **256** | **8** | **4** | **512** | **8.52M (6.43M)** | **~2.2 s (est.)** | **~1,800 (est.)** |

Variants of the 384/6/512 config:

| variant | tok/s | verdict |
|---|---:|---|
| fp32, eager (baseline) | 1,460-1,550 | used |
| bf16 autocast | 520 | 3x slower: no AMX, so bf16 matmuls are emulated |
| `torch.compile` (inductor) | 1,600 | +3%, 40 s compile; not worth the risk, left off (`compile: true` works) |
| fused AdamW | 1,550 | no change |
| math SDPA backend instead of flash | 1,260 | slower; default flash kernel kept |
| micro-batch 4 / 16 | 1,220 / 1,280 | 8 is the sweet spot |

**Noise caveat.** Other agents were using the CPU throughout, and from 10:34
the corpus builder took three or more cores (load average 5-10). Benchmarks
from then on, including a run of two processes in parallel, swung by 2x
between repeats and are not in the table. The chosen 256/8 config could only
be measured under that load. Its row is a FLOP-based estimate: the 256-wide
models ran at ~92 GFLOP/s (6 x params x tokens), so 51 MFLOP/token gives
~550 us/token. Under load it was the fastest of {256/6, 256/8, 320/6, 384/6}
in one round and the second fastest in the other. The profile (384/6/512) is
56% matmul (~200 GFLOP/s, near the 2-core peak), 18% CPU flash attention and
~10% elementwise (RoPE, RMSNorm, SiLU). So this box is compute-bound and the
numbers will not get much better.

Two runs in parallel should each get about 90% of the single-run speed,
since the cores are not hyperthreads and the matmuls are cache-resident. The
plan assumes **~1,600 tok/s per run**. Check this on launch (see below).

## Choosing the size

At ~1,600 tok/s, 3 h is roughly **16M training tokens per model**. That is a
tiny budget: about 1e15 FLOPs. The scaling laws all point to small models:

- **Chinchilla** (about 20 tokens/param): C = 6ND = 1e15 gives N ~ 3M params
  and D ~ 60M tokens. Every candidate is over-sized. The 384/6 model would see
  ~1.2 tokens/param (4-5x too big), while 256/6 and 256/8 see ~2-3.5.
  IsoFLOP curves put a 4x over-size at roughly +0.1 nats and 2x at about
  +0.02.
- **Kaplan L(N, D)** with each candidate's own token count gives 192/6 4.08,
  256/6 4.11, 320/6 4.20, 384/6 4.20 and 384/8 4.35 nats. The curve is flat,
  but it favours the smaller models.
- **Chinchilla's parametric fit** comes out level (7.24 / 7.24 / 7.28 for
  256/6, 384/6 and 192/6). It is far outside its fitted range here.
- Below ~250 wide, efficiency drops (192/6 reached only 77 GFLOP/s), and the
  8192 x d embedding becomes most of the model.
- At small scale, depth beats width for the same parameters (MobileLLM,
  2024). Samples have to be coherent for the demo, and the 6-layer 256 model
  is shallow.

**Choice: d_model 256, 8 layers, 4 heads (head_dim 64), d_ff 704, context
512. That is 8.5M parameters, 6.4M of them non-embedding.** It sits between the
Chinchilla optimum (smaller) and the capacity we want for prose (bigger). It
is deeper than the 6-layer options and should see ~16M tokens (~2
tokens/param), about 20% more than 384/6 would. Context 512 costs only ~20%
over 256 at this width (the 384-wide model ran no slower at 512 than at 256),
and prose and chat need it.

## Schedule

- **Batch: 16,384 tokens per optimizer step** (micro-batch 8 x 512, grad
  accumulation 4), so ~950 steps. That is below the 32-64K first suggested,
  on purpose. Kaplan's critical batch at losses of 4-5 nats is ~100-300K
  tokens, so any batch well under that is equally token-efficient. A smaller
  batch then just buys more optimizer steps (950 rather than 475) out of the
  same 16M tokens. nanoGPT's small CPU configs use the same 16K tokens/step.
- **LR 2e-3** peak (AdamW 0.9/0.95, wd 0.1 on 2-D weights only, so
  norm gains are not decayed). Warmup 50 steps, cosine to 2e-4, grad clip 1.0.
  2e-3 is standard for ~10M-param models at this batch size.
- **max_steps 950**: 950 x 16,384 = 15.6M tokens. At 1,600 tok/s that is 2.7 h,
  a 10% margin inside 3 h. `--max-hours 3` is the backstop: if the machine is
  slower, the run stops cleanly at 3 h. At that point the cosine is at most a
  few percent from its end and the LR is already near the floor, so cutting it
  short costs little.
- **Eval and checkpoint every 50 steps** (~8.5 min). The eval covers a fixed
  16 x 4096 validation windows (65K tokens, ~15 s, under 3% of the time).
  These windows are the same for every eval and both runs. Checkpoints land
  at the same steps in main and control, so they can be compared step for
  step.
- **Data**: random 513-token windows from the token file. Window positions
  depend only on (seed, step), so main and control draw from the same
  positions, and a resumed run sees exactly the batches of an uninterrupted one.
  The same seed means the same initial weights. At 16M tokens the model sees
  ~10% of the 150M-token training file, so repeats are rare.

## Data as built (5 Oct 2026)

| file | books | tokens |
|---|---:|---:|
| main/train.bin | 1,415 of 16,168 | 150.1M (capped) |
| control/train.bin | 1,436 (the same 1,415 + 21 flight books main dropped) | 152.7M |
| main/val.bin | 324 (all) | 31.9M |
| control/val.bin | 328 (all) | 32.2M |

The tokenizer was trained on 757 main/train books (289 MB) in 92 s. It
averages ~3.8 characters per token on the corpus. " balloon" splits into
`Ġball|oon` and " aeronaut" into `Ġa|er|on|aut`; " steam", " flying" and
" Napoleon" are single tokens. Tokenizing everything took 15 min on the
shared CPU (~0.4M tokens/s with 4 workers).

## Commands

```bash
cd /home/user/flight-ai-model-

# 1. tokenizer: byte-level BPE, 8192 vocab, from ~300 MB of main/train (about 3-5 min)
python3 scripts/train_tokenizer.py --corpus /home/user/data/corpus/main/train.jsonl \
    --out tokenizer/tokenizer.json --sample-mb 300

# 2. token files: main first (sets the book cutoff), then control on the same books (about 5-10 min)
python3 scripts/tokenize_corpus.py --corpus /home/user/data/corpus \
    --out /home/user/data/tokenized --max-train-tokens 150000000 --workers 4

# 3. main and control in parallel, 2 threads each
mkdir -p /home/user/data/runs/main /home/user/data/runs/control
for c in main control; do
  nohup python3 scripts/train.py --config configs/cpu_pilot.yaml \
      --data-dir /home/user/data/tokenized/$c --out /home/user/data/runs/$c --max-hours 3 \
      > /home/user/data/runs/$c/stdout.log 2>&1 &
done
tail -f /home/user/data/runs/main/stdout.log

# 4. compact fp16 files for git (~17 MB each)
python3 scripts/export_checkpoint.py /home/user/data/runs/main    checkpoints/main.pt
python3 scripts/export_checkpoint.py /home/user/data/runs/control checkpoints/control.pt
# or a matched intermediate step: .../runs/main/step_000500.pt -> checkpoints/main_step500.pt
```

**Check the speed at launch.** After about 5 minutes (step 15-20), read
`tok_per_s` in `runs/*/log.jsonl`. If it is more than ~10% below 1,600, kill
both runs, delete both run directories, and relaunch both with the same
`--max-steps N`, where
N = `python3 -c "print(int(0.9 * 3 * 3600 * TOKS / 16384))"` and TOKS is the
measured rate. The two runs must never have different `max_steps`.

**Resume** after a crash: re-run the same command with `--resume` (with
`--max-hours` set to the time remaining). The schedule, data order and
optimizer state continue exactly. SIGTERM or Ctrl-C saves a checkpoint and
exits cleanly.

**Smoke test** (seconds): `python3 scripts/train.py --config configs/tiny_test.yaml
--data-dir /home/user/data/tokenized/main --out /tmp/prewright_tiny`.

## Run directory

`config.yaml` holds the resolved config. `log.jsonl` has step, tokens, loss,
val_loss, lr, grad_norm, tok_per_s and elapsed. `ckpt.pt` is the full state
for resume. `step_XXXXXX.pt` are light fp16 exports with the tokenizer
embedded, and `final.json` is written at the end. Load any of them with:

```python
from prewright.load import load_model
model, tok = load_model("checkpoints/main.pt")  # or a run dir, ckpt.pt, step_*.pt
```

## Scaling up

`configs/gpu_125m.yaml` (~120M params, 1024 context, 2.6B tokens) and
`configs/gpu_1b.yaml` (~1.2B, 2048 context, 8.4B tokens) run the same code
with `device: cuda`, bf16 autocast, fused AdamW and `torch.compile`. They
have not been tested here, since the build box has no GPU. Re-tokenize without
a cap first (`--max-train-tokens 0`). At that scale data is the limit: PG-19's
pre-1903 English is only a couple of billion tokens.
