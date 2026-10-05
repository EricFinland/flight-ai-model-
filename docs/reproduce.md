# Reproduce from scratch

Every step of the pipeline, in order: the long form of the [README](../README.md)'s short version.

All commands run from the repository root. Raw data and built corpora live outside git, in
`/home/user/data` by default. To use another place, set `DATA` and keep the explicit flags below.
Times are for a shared 4-core CPU.

```bash
DATA=/home/user/data
pip install -e '.[dev]' && pytest            # unit tests (seconds)
pip install -e '.[plots]'                    # matplotlib, for the plots in step 9

# 1. Download the PG-19 books published in or before 1902 (needs curl and gsutil, which comes
#    with the Google Cloud CLI; ~6.8 GB, 17,350 files)
scripts/download_pg19.sh $DATA

# 2. Build the main and control corpora: date, anachronism, English and flight filters (~16 min)
python3 scripts/build_corpus.py --raw $DATA/pg19_raw --metadata $DATA/metadata.csv \
    --out $DATA/corpus --workers 3

# 2b. Re-derive main from control with the current lexicon (no re-download, ~11 min). Needed
#     after a change to src/prewright/flight.py, and once after a fresh build to match the
#     released corpus: the same-title rule (other editions of a dropped flight book) is
#     applied only here.
python3 scripts/rescrub_corpus.py --corpus $DATA/corpus --workers 2

# 3. Tokenizer: byte-level BPE, 8,192 tokens, from ~300 MB of main/train only (~2-5 min)
python3 scripts/train_tokenizer.py --corpus $DATA/corpus/main/train.jsonl \
    --out tokenizer/tokenizer.json --sample-mb 300

# 4. Token files: the same books in the same order for both corpora, 150M-token cap (~15 min).
#    Main sets the book cutoff and control reuses it.
python3 scripts/tokenize_corpus.py --corpus $DATA/corpus --out $DATA/tokenized \
    --max-train-tokens 150000000 --workers 4
#    After a re-scrub (2b), keep the old book set: pass the cutoff from the existing
#    $DATA/tokenized/control/train.meta.json ("cutoff") and rebuild main only:
# python3 scripts/tokenize_corpus.py --corpus $DATA/corpus --out $DATA/tokenized \
#     --corpora main --train-cutoff <cutoff hex> --workers 4

# 5. Train main and control in parallel, 2 threads each, 850 steps (about 3-3.6 h at the
#    1,050-1,300 tokens/s per run measured here; the control took 2.9 h. The released
#    flight-free model was retrained alone on 4 threads, --threads 4, in about 71 min). Resumable: re-run the same command after a
#    crash and each run continues from its ckpt.pt. Exports step_XXXXXX.pt every 25 steps.
#    (The launcher's own DATA means the token-file directory, hence DATA=$DATA/tokenized.)
DATA=$DATA/tokenized RUNS=$DATA/runs scripts/launch_cpu_pilot.sh   # returns at once; logs in runs/*/stdout.log

# Steps 6-10 (apart from building the chat data, which it expects in $DATA/sft) are what
# scripts/finish_pilot.sh (make finish) runs, in order, once both runs have written
# final.json. They are listed here so each can be run on its own. finish_pilot.sh reads the
# chat data and corpus statistics from /home/user/data whatever DATA is; with another data
# directory, run the steps below one by one.

# 6. Compact fp16 exports for git (~17 MB each)
python3 scripts/export_checkpoint.py $DATA/runs/main    checkpoints/main.pt
python3 scripts/export_checkpoint.py $DATA/runs/control checkpoints/control.pt

# 7. Chat data mined from the scrubbed books (~18 min), then a chat fine-tune of each model
#    on the same pairs (2 epochs, at most 40 min each)
python3 scripts/build_sft_data.py --corpus $DATA/corpus/main/train.jsonl --out $DATA/sft --workers 2
for c in main control; do
  python3 scripts/sft.py --init $DATA/runs/$c --data $DATA/sft --out $DATA/runs/${c}_chat \
      --pretrain-bin $DATA/tokenized/$c/train.bin --export checkpoints/${c}_chat.pt \
      --epochs 2 --max-minutes 40 --threads 2
done

# 8. Evaluation: main vs control (~30 min)
python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt \
    --out evals/results --data-dir $DATA/corpus --tokenized-dir $DATA/tokenized --threads 2

# 9. The same questions to both chat models, then the flight gap over training
#    (~5 s per matched step) and the loss curves
python3 scripts/chat_showcase.py --main checkpoints/main_chat.pt --control checkpoints/control_chat.pt \
    --out evals/results/chat_showcase.md --threads 4
python3 scripts/eval_curve.py --main-run $DATA/runs/main --control-run $DATA/runs/control \
    --out evals/results --threads 1

# 10. Copy the numbers into the README and the model card (add
#     --files README.md MODEL_CARD.md docs/writeup.md docs/social.md to fill the write-up and the
#     social drafts as well; finish_pilot.sh and make fill do)
python3 scripts/fill_readme_results.py --data $DATA

# Optional: rebuild the showcase page from the results (docs/site/README.md)
python3 scripts/build_site.py --results evals/results --data $DATA --out docs/site/index.html

# Optional: red-team search of the scrubbed text for flight the lexicon missed
python3 scripts/audit_leaks.py $DATA/corpus/main/train.jsonl \
    --compare $DATA/corpus/control/train.jsonl --rg --out evals/leak_audit
```

Step 5 is deterministic given the seed. The batch for each step depends only on (seed, step), so
main and control see text from the same positions, and a resumed run sees exactly the batches of
an uninterrupted one. `launch_cpu_pilot.sh` runs `scripts/train.py --config
configs/cpu_pilot.yaml --max-steps 850 --max-hours 3.6 --set ckpt_interval=25` for both corpora
(`FRESH=1` wipes both runs and starts over). On a different machine, measure the speed after a few
minutes (`tok_per_s` in `log.jsonl`) and pick `--max-steps` so that both runs finish inside your
budget, always the same for both; see [docs/training.md](training.md). A smoke test of the
evaluation pipeline with no checkpoints: `python3 scripts/evaluate.py --fake --quick --out
/tmp/eval_fake`.

## Make targets

The [Makefile](../Makefile) wraps the commands above; every target runs one documented script.
`make help` lists them:

| target | what it runs |
|---|---|
| `make help` | list the targets |
| `make install`, `make test`, `make smoke` | `pip install -e '.[dev]'`; the unit tests; the evaluation pipeline on a fake model |
| `make download corpus rescrub tokenizer tokenize` | steps 1-4, with `rescrub` as step 2b |
| `make train` | step 5, the CPU pilot (`launch_cpu_pilot.sh`) |
| `make train-gpu` | step 5 on GPUs: `CONFIG=configs/gpu_125m.yaml`, `NPROC=` GPUs per run ([docs/scaling.md](scaling.md)) |
| `make finish` | steps 6-10 as the pilot runs them (`finish_pilot.sh`; run `make sft-data` first) |
| `make export`, `make sft-data`, `make sft`, `make eval`, `make curve`, `make fill` | steps 6-10 one at a time (`make sft` tunes main, then control, each with a 40-minute budget) |
| `make showcase`, `make site` | the side-by-side chat answers (step 9); the optional showcase page |
| `make chat`, `make generate` | talk to `checkpoints/main_chat.pt`; complete `PROMPT="..."` with both base models |

Paths and settings are variables: `make tokenize DATA=/mnt/data WORKERS=16 MAX_TRAIN_TOKENS=0`.
