# Scaling up: training Prewright on rented GPUs

This is a step-by-step guide to running the whole pipeline (data, main and
control pretraining, chat fine-tuning, evaluation) on a rented GPU machine,
with the full ~1.7B-token corpus instead of the CPU pilot's 150M-token slice.

**Status, plainly.** None of this has been run on a GPU yet: the build machine
had no GPU. The CUDA code path was reviewed line by line (see
[What was checked](#what-was-checked-in-the-gpu-code-path)), and the multi-GPU
(DDP) path is tested on CPU, including a real two-process `torchrun` run that
reproduces the single-process result. Expect to fix something small the first
time. The smoke tests in step 4 exist to find it in minutes, not hours.

What exists and what does not:

| feature | status |
|---|---|
| one GPU, bf16 autocast, fused AdamW, TF32, `torch.compile` | built in (`device: cuda` in the GPU configs) |
| several GPUs on **one** machine (DDP via `torchrun`) | built in, tested on CPU with gloo, not on GPUs |
| several machines (multi-node) | not built |
| FSDP / ZeRO (sharding weights or optimizer state) | not built, not needed up to ~1.2B params on 80 GB cards |
| gradient (activation) checkpointing | not built; lower `micro_batch` instead (below) |
| GPU for `sft.py`, `evaluate.py`, `chat.py`, `generate.py` | `--device cuda` (and `--dtype bf16` for `sft.py`) |
| GPU for `eval_curve.py`, `chat_showcase.py` | CPU only; they score short probes and are fast enough |

## 1. Pick a config

| config | params | context | tokens trained | GPU memory | GPU you need |
|---|---:|---:|---:|---:|---|
| [`configs/cpu_pilot.yaml`](../configs/cpu_pilot.yaml) | 8.5M | 512 | 13.9M | n/a | none (this repo's build) |
| [`configs/gpu_125m.yaml`](../configs/gpu_125m.yaml) | 119.6M | 1,024 | 2.62B | ~28 GB | one A100 40/80 GB or H100; a 24 GB card with a smaller micro-batch |
| [`configs/gpu_1b.yaml`](../configs/gpu_1b.yaml) | 1.23B | 2,048 | 8.39B | ~65-70 GB | one 80 GB A100/H100 per process; 8 x H100 to finish in hours |

Start with **gpu_125m**. It is cheap (tens of dollars for both models), it
finishes in an afternoon, and it tells you everything about the setup before
you spend hundreds on the 1B run.

## 2. Token budgets: how much to train

The rule of thumb from Chinchilla (Hoffmann et al., 2022) is **about 20
training tokens per parameter** for the best model per unit of compute.

| model | Chinchilla tokens | corpus (main, full) | config trains | epochs over the corpus |
|---|---:|---:|---:|---:|
| 8.5M (pilot) | 0.17B | 0.15B slice | 13.9M | 0.09 |
| 119.6M | 2.4B | ~1.7B | 2.62B | ~1.5 |
| 1.23B | 24.6B | ~1.7B | 8.39B | ~4.9 |

The corpus is the limit. Main/train holds 16,165 books, about 6.6 billion
characters, which is ~1.73B tokens at this tokenizer's ~3.8 characters per
token (control has 240 more books, ~1.76B). For 125M that is fine: 1.5 passes
over the data loses almost nothing. For 1.2B it is far too little data:
Chinchilla would want 24.6B tokens, 14 passes. Repeating data up to about 4
epochs is nearly as good as fresh data; beyond that the returns fall quickly
(Muennighoff et al., 2023, *Scaling Data-Constrained Language Models*). The 1B
config's 8.4B tokens (~5 epochs) is already at that edge, so **do not train
the 1B model longer**; a bigger 1B result needs more pre-1903 text (see
[data-sources.md](data-sources.md)), not more steps.

To change the budget, change `max_steps`:

```
tokens per step = micro_batch x grad_accum x max_seq_len
max_steps       = tokens you want / tokens per step
```

`gpu_125m`: 32 x 16 x 1024 = 524,288 tokens/step, so 5,000 steps = 2.62B
tokens. `gpu_1b`: 8 x 64 x 2048 = 1,048,576 tokens/step, 8,000 steps = 8.39B.
Also scale `warmup_steps` (about 5% of `max_steps`), `eval_interval` and
`ckpt_interval` with it. **Main and control must always use the same config
and the same `max_steps`**, or the comparison between them means nothing.

## 3. Time and cost

Training compute is about `6 x params x tokens` FLOPs, plus attention
(`6 x layers x context x d_model` per token):

| config | FLOPs per model | FLOPs per token |
|---|---:|---:|
| gpu_125m | 2.1e18 | 0.79 GFLOP |
| gpu_1b | 6.7e19 | 8.0 GFLOP |

A plain PyTorch loop like this one typically reaches 30-40% of a GPU's peak
bf16 throughput ("MFU"). An H100 SXM peaks at ~990 TFLOP/s and an A100 at
~312, so assume ~300-400 TFLOP/s on an H100 and ~100-130 on an A100. That
gives, **per model** (you train two, main and control):

| config | hardware | tokens/s (watch `tok_per_s`) | time per model | both models |
|---|---|---:|---:|---:|
| gpu_125m | 1 x H100 | ~350-450K | ~1.5-2.5 h | ~3-5 h |
| gpu_125m | 1 x A100 | ~120-160K | ~5-6 h | ~10-12 h |
| gpu_1b | 1 x H100 | ~40-50K | ~2-2.5 days | ~4-5 days |
| gpu_1b | 8 x H100 (DDP) | ~300-380K | ~7-9 h | ~14-18 h |
| gpu_1b | 1 x A100 80 GB | ~13-16K | ~6-7 days | ~2 weeks (not recommended) |

Rough rental prices (on-demand, 2026; spot or community clouds are cheaper):
H100 80 GB ~$2-3.5/h, A100 80 GB ~$1.2-2/h, 8 x H100 node ~$16-28/h. So:

- **gpu_125m, both models: ~$10-25** on one H100 or A100.
- **gpu_1b, both models: ~$250-500**, about the same on 1 x H100 (5 days) or
  8 x H100 (under a day). Fewer days means fewer chances for something to go
  wrong; pick the 8-GPU node if you can.

Add an hour or two for data preparation and evaluation. Data preparation is CPU
work: do it on a cheap CPU machine and copy the results over (step 3), or
accept paying GPU prices for a while.

After a few minutes of training, compute your real MFU from the log and redo
the arithmetic: `MFU = tok_per_s x FLOPs per token / peak FLOP/s`.

## 4. Memory: what fits where

Per GPU process, the model needs:

- **Weights, gradients and AdamW state**: 16 bytes per parameter (fp32
  weights 4, gradients 4, two AdamW moments 8). 125M: ~1.9 GB. 1.2B: ~19.7 GB.
  DDP adds one more gradient-sized buffer (125M: 0.5 GB; 1.2B: 4.9 GB).
- **Activations** saved for the backward pass, which grow with `micro_batch x
  max_seq_len`. Measured for this model under bf16 autocast: ~40 KB per token
  per layer at d_model 768 and ~104 KB at d_model 2048, plus ~50 KB per token
  for the logits. 125M at micro-batch 32 x 1024: ~23 GB. 1.2B at micro-batch
  8 x 2048: ~42 GB. `torch.compile` fuses the small operations and usually
  saves part of that.

So **gpu_125m needs ~28 GB** (any 40 or 80 GB card) and **gpu_1b ~65-70 GB**
(an 80 GB card; tight with DDP).

**If you run out of memory** ("CUDA out of memory"), halve `micro_batch` and
double `grad_accum`. The tokens per step, and therefore the training, stay the
same; only the speed drops a little:

```bash
python3 scripts/train.py --config configs/gpu_1b.yaml ... --set micro_batch=4 --set grad_accum=128
python3 scripts/train.py --config configs/gpu_125m.yaml ... --set micro_batch=16 --set grad_accum=32   # 24 GB cards
```

There is no gradient checkpointing (recomputing activations in the backward
pass to save memory). It would let a 1.2B model train on a 40 GB card, at
~30% more compute. It is not needed for the configs above on 80 GB cards.

## 5. Step by step

### 5.1 Rent the machine

- A Linux image with a recent CUDA build of PyTorch (2.2 or newer; most
  providers have a "PyTorch" image).
- Disk: **~100 GB** for gpu_125m, **~300 GB** for gpu_1b. Raw books 6.8 GB,
  corpora 13.6 GB, token files 7 GB; a 1.2B `ckpt.pt` with optimizer state is
  ~15 GB and every `step_XXXXXX.pt` export 2.5 GB, for each of two runs.
- RAM: 32 GB or more (64 GB for gpu_1b: exporting a 1.2B checkpoint loads it
  twice); CPU: 8+ cores if you prepare data there.
- Use `tmux` or `screen` so a dropped SSH connection does not kill a run.

### 5.2 Install and test

```bash
git clone https://github.com/EricFinland/flight-ai-model- && cd flight-ai-model-
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
#   must print True and your GPU count. If not, install a CUDA build of torch first
#   (https://pytorch.org/get-started/locally/); do not let pip replace it with a CPU build.
pip install -e '.[dev]'
make test                      # or: python3 -m pytest -q   (about a minute)
nvidia-smi                     # GPU model and free memory
```

### 5.3 Build the data (the full corpus)

Set where the data goes, then run steps 1-4 of the README's "Reproduce from
scratch", with **no token cap** in the last one:

```bash
export DATA=/workspace/data    # any disk with ~30 GB free
pip install gsutil              # download_pg19.sh needs it (and curl)
make download DATA=$DATA                       # ~6.8 GB of PG-19 books dated <= 1902
make corpus   DATA=$DATA WORKERS=8             # ~16 min on 3 cores; faster with more
make tokenize DATA=$DATA WORKERS=16 MAX_TRAIN_TOKENS=0
```

Keep the committed `tokenizer/tokenizer.json` (trained on main only). Do not
run `make tokenizer` unless you mean to change the tokenizer; if you do, the
committed checkpoints no longer match it.

Tokenizing ~3.5B tokens (main + control) runs at ~0.1M tokens/s per worker:
~40 min with 16 workers, ~2.5 h with 4. Check the result:

```bash
cat $DATA/tokenized/main/train.meta.json      # "tokens": ~1.7e9, "books": 16165, "cutoff": null
ls -lh $DATA/tokenized/*/train.bin            # ~3.5 GB each (2 bytes per token)
```

`"cutoff": null` means no cap: control then holds all of its books, the same
books as main plus the 240 books main dropped as being about flight.

**Cheaper alternative**: build `$DATA/corpus` and `$DATA/tokenized` on a CPU
machine (it is exactly the CPU pilot pipeline with `MAX_TRAIN_TOKENS=0`) and
copy them to the GPU machine with `rsync -a`. Training needs only
`tokenized/`; chat fine-tuning and evaluation also read `corpus/` and `sft/`.

### 5.4 Smoke tests (minutes; do not skip)

```bash
# a. the loop on the GPU in seconds: tiny model, bf16
python3 scripts/train.py --config configs/tiny_test.yaml --data-dir $DATA/tokenized/main \
    --out /tmp/smoke_tiny --device cuda --set dtype=bf16

# b. the real config for 30 steps: compile time, memory, speed
python3 scripts/train.py --config configs/gpu_125m.yaml --data-dir $DATA/tokenized/main \
    --out /tmp/smoke_125m --max-steps 30 --set log_interval=5 --set eval_interval=30 --set ckpt_interval=30
#    meanwhile, in another terminal: watch -n 5 nvidia-smi

# c. resume works: start (b) again in a fresh directory, press Ctrl-C after ~15 steps (it saves
#    and exits), then re-run the same command with --resume: it must continue from that step
```

The first steps of (b) take a minute or two while `torch.compile` compiles;
then read `tok_per_s` from the printed lines and compare with the table in
section 3. If `torch.compile` fails, add `--set compile=false` (slower, ~10-30%,
but otherwise identical). Delete `/tmp/smoke_*` afterwards.

### 5.5 Train main and control

The same command twice, once per corpus, with the same config. Everything is
deterministic given the seed: both runs start from the same weights and draw
their windows from the same positions.

**One GPU** (the runs one after the other):

```bash
make train-gpu DATA=$DATA CONFIG=configs/gpu_125m.yaml
# which runs, for c in main control:
#   python3 -m torch.distributed.run --standalone --nproc_per_node 1 scripts/train.py \
#       --config configs/gpu_125m.yaml --data-dir $DATA/tokenized/$c --out $DATA/runs_gpu/$c --resume
```

(Plain `python3 scripts/train.py ...` without `torchrun` works just as well on
one GPU.)

**Two GPUs**: run main and control at the same time, one per GPU. This needs no
DDP at all:

```bash
mkdir -p $DATA/runs_gpu/main $DATA/runs_gpu/control
for c in main control; do
  i=$([ $c = main ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$i nohup python3 scripts/train.py --config configs/gpu_125m.yaml \
      --data-dir $DATA/tokenized/$c --out $DATA/runs_gpu/$c --resume > $DATA/runs_gpu/$c/stdout.log 2>&1 &
done
tail -f $DATA/runs_gpu/main/stdout.log
```

**Eight GPUs** (DDP, for gpu_1b): each run uses all eight; run them one after
the other:

```bash
make train-gpu DATA=$DATA CONFIG=configs/gpu_1b.yaml NPROC=8
# for each c: torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/gpu_1b.yaml \
#                 --data-dir $DATA/tokenized/$c --out $DATA/runs_gpu/$c --resume
```

`--resume` makes every command safe to re-run: with no `ckpt.pt` in the output
directory it starts fresh, otherwise it continues exactly where it stopped
(same schedule, same data order, same optimizer state). On resume the config
comes from the checkpoint, not from the YAML; only `--max-hours` is taken from
the new command.

**Monitoring.** Each run writes `log.jsonl` (step, loss, val_loss, lr,
grad_norm, tok_per_s, elapsed) and prints the same. Healthy signs: the loss
falls fast in the first few hundred steps, `grad_norm` settles below ~1, and
`val_loss` tracks the training loss. A loss that jumps up and stays up, or a
`nan`, means stop and lower the learning rate (`--set lr=...`), then restart
from scratch (with `--resume` it would keep the old config).

**Stopping cleanly.** Ctrl-C, or `kill -TERM` on a training process, saves a
checkpoint at the end of the current step and exits. Under DDP, signalling any
one rank stops all of them at the same step. `--max-hours H` does the same
after H hours. Avoid killing `torchrun` itself with SIGKILL; the last
checkpoint is still safe (saves are atomic), but the steps since then are lost.

### 5.6 After pretraining: export, chat, evaluate

The same steps 6-10 as in the README, with the GPU flags:

```bash
R=$DATA/runs_gpu
# 6. fp16 exports. 125M: ~240 MB each, 1.2B: ~2.5 GB, too big for plain git
#    (GitHub's limit is 100 MB per file): keep them out of checkpoints/ or use Git LFS
#    or the Hugging Face Hub.
python3 scripts/export_checkpoint.py $R/main    exports/main_125m.pt --max-mb 5000
python3 scripts/export_checkpoint.py $R/control exports/control_125m.pt --max-mb 5000

# 7. chat data (CPU, ~18 min, once) and chat fine-tuning on the GPU (minutes)
python3 scripts/build_sft_data.py --corpus $DATA/corpus/main/train.jsonl --out $DATA/sft --workers 8
for c in main control; do
  python3 scripts/sft.py --init $R/$c --data $DATA/sft --out $R/${c}_chat \
      --pretrain-bin $DATA/tokenized/$c/train.bin --export exports/${c}_chat_125m.pt \
      --device cuda --dtype bf16 --max-minutes 60
done
#    1.2B: add --batch-size 4 if memory runs out.

# 8. evaluation on the GPU
python3 scripts/evaluate.py --main exports/main_125m.pt --control exports/control_125m.pt \
    --out evals/results_125m --data-dir $DATA/corpus --tokenized-dir $DATA/tokenized --device cuda

# 9. flight gap over training and loss curves (CPU; one point per saved step)
python3 scripts/eval_curve.py --main-run $R/main --control-run $R/control --out evals/results_125m --threads 8

# talk to it
python3 scripts/chat.py --model exports/main_chat_125m.pt --device cuda
```

`exports/` is in `.gitignore`, so `git add .` cannot pick up these
gigabyte files; `checkpoints/` is not ignored, because the CPU pilot's ~17 MB
files are committed there.

Step 10, `fill_readme_results.py`, writes the numbers from
`evals/results/eval.json` into README.md and MODEL_CARD.md. With results in
another directory, pass `--eval evals/results_125m/eval.json --curve
evals/results_125m/curve.json`, and use `--out-dir /tmp/filled` first to see
what would change: the README describes the CPU pilot, and the scaled model
deserves its own section, not silently replaced numbers.

## 6. How the multi-GPU (DDP) training works

`scripts/train.py` notices when `torchrun` started it (the `RANK`,
`WORLD_SIZE` and `LOCAL_RANK` environment variables) and then:

- joins the process group (NCCL on GPUs, gloo on CPU) and uses GPU
  `cuda:LOCAL_RANK`;
- wraps the model in `DistributedDataParallel` (after `torch.compile`), which
  averages gradients across GPUs during the backward pass;
- keeps the **global batch unchanged**: each optimizer step still has
  `grad_accum` micro-batches drawn from `(seed, step)`, and GPU `r` of `N`
  takes micro-batches `r, r+N, r+2N, ...`. So `grad_accum` must be divisible
  by `N`, and 8 GPUs train the same model on the same data as one GPU, about
  8 times faster. Gradients are only exchanged on each GPU's last micro-batch
  of a step;
- makes every GPU stop at the same step (a `--max-hours` limit or a signal on
  any one of them), so none is left waiting;
- runs the validation on every GPU (identical, and short) and lets only rank
  0 write files: `log.jsonl`, `ckpt.pt`, `step_XXXXXX.pt`, `final.json`.
  Checkpoints have exactly the single-GPU format; a run can be resumed with a
  different number of GPUs.

It is single-machine only (`--standalone`). The tests
(`tests/test_ddp.py`) check, on CPU, that the micro-batch split covers exactly
the single-process batch, that a one-process DDP run matches a plain run, and
that a two-process `torchrun` run reproduces the single-process losses.

## What was checked in the GPU code path

For the curious, and for whoever debugs the first GPU run:

- **Device placement.** Training and validation batches are moved to
  `cfg.device`; training batches go through pinned memory, so the copy to the
  GPU overlaps with compute. The RoPE tables are buffers and move with the
  model. Checkpoints are written from CPU copies and loaded with
  `map_location="cpu"`, then moved; AdamW state follows its parameters' device
  on load.
- **Precision.** `dtype: bf16` uses `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)`: matmuls in bf16, master weights, residual stream,
  RMSNorm and the loss in fp32. bf16 has fp32's exponent range, so no
  `GradScaler` is needed (that is only for fp16). TF32 is switched on for the
  remaining fp32 matmuls.
- **Optimizer.** AdamW uses `fused=True` on CUDA (one kernel for all
  parameters).
- **Attention.** `F.scaled_dot_product_attention(is_causal=True)` picks the
  FlashAttention kernel on A100/H100 in bf16.
- **`torch.compile`** is on in both GPU configs; checkpoints are saved from the
  uncompiled module, so their parameter names never carry the `_orig_mod.`
  prefix. Turn it off with `--set compile=false` if it misbehaves.
- **Syncs.** The training loss is summed on the GPU and read once per log
  line, not after every micro-batch, so the CPU can queue work ahead.
- **Sampling on a GPU.** `generate.py` and `chat.py` create their random
  generator on the model's device (a CPU generator with a CUDA model is an
  error in `torch.multinomial`).
- **Fixed in this review:** the per-micro-batch `loss.item()` sync; no
  `--device` for chat fine-tuning (`sft.py` batches stayed on the CPU); no
  `--device` and a CPU-only random generator in `generate.py`/`chat.py`; no
  `--device` in `evaluate.py`; no multi-GPU support.
