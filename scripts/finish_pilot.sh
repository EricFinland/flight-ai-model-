#!/bin/bash
# Everything after pretraining, in order, for the CPU pilot:
#   both runs done -> export both, chat fine-tune both in parallel (2 threads each)
#   then           -> full evaluation (4 threads), chat showcase, training curves,
#                     README/model card/write-up numbers, showcase page
# Waits on files (a final.json that says the run reached max_steps), never on
# process names. Safe to re-run: finished stages are skipped.
#   scripts/finish_pilot.sh > $RUNS/finish.log 2>&1
case "${1:-}" in -h|--help) sed -n '2,10p' "$0"; exit 0;; esac
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-/home/user/data/runs}
TOK=${TOK:-/home/user/data/tokenized}
SFT_ARGS=(--data /home/user/data/sft --epochs 2 --max-minutes 40 --threads 2)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# A run that stopped on a signal or the time budget also writes final.json (and is
# resumed later), so wait for one that says it completed all steps.
wait_done() {
  until [ -f "$1" ] && python3 -c "import json,sys; sys.exit(json.load(open('$1')).get('stopped') != 'max_steps')"; do
    sleep 30
  done
}
mkdir -p checkpoints evals/results

chat_tune() {  # $1 = main|control
  local c=$1
  if [ ! -f "checkpoints/${c}_chat.pt" ]; then
    log "$c: chat fine-tune"
    python3 scripts/sft.py --init "$RUNS/$c" --out "$RUNS/${c}_chat" --export "checkpoints/${c}_chat.pt" \
      --pretrain-bin "$TOK/$c/train.bin" "${SFT_ARGS[@]}" > "$RUNS/${c}_chat.log" 2>&1
    log "$c: chat fine-tune done"
  fi
}

export_base() {  # $1 = main|control
  local c=$1
  wait_done "$RUNS/$c/final.json"
  [ -f "checkpoints/$c.pt" ] || python3 scripts/export_checkpoint.py "$RUNS/$c" "checkpoints/$c.pt"
  log "$c: exported checkpoints/$c.pt"
}

# Stages never overlap more threads than the 4 cores: oversubscribed OpenMP
# threads spin and slow every job several-fold.
log "waiting for control"
export_base control
log "waiting for main"
export_base main
chat_tune control &
CONTROL_CHAT=$!
chat_tune main &
MAIN_CHAT=$!
wait "$CONTROL_CHAT"
wait "$MAIN_CHAT"
if [ ! -f evals/results/eval.json ] || [ "${REEVAL:-0}" = 1 ]; then
  log "full evaluation"
  python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt \
    --out evals/results --threads 4 > "$RUNS/evaluate.log" 2>&1
  log "evaluation done"
fi

log "chat showcase"
python3 scripts/chat_showcase.py --main checkpoints/main_chat.pt --control checkpoints/control_chat.pt \
  --out evals/results/chat_showcase.md --threads 4 > "$RUNS/showcase.log" 2>&1
log "training curves"
python3 scripts/eval_curve.py --main-run "$RUNS/main" --control-run "$RUNS/control" --out evals/results \
  --threads 4 > "$RUNS/curve.log" 2>&1
log "filling README and model card"
python3 scripts/fill_readme_results.py --data /home/user/data \
  --files README.md MODEL_CARD.md docs/writeup.md docs/social.md > "$RUNS/fill.log" 2>&1
log "showcase page"
python3 scripts/build_site.py --results evals/results --data /home/user/data --out docs/site/index.html \
  > "$RUNS/site.log" 2>&1
log "ALL DONE"
