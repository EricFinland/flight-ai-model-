#!/bin/bash
# Everything after pretraining, in order, for the CPU pilot:
#   control done -> export, chat fine-tune (control_chat)
#   main done    -> export, chat fine-tune (main_chat) alongside the full evaluation
#   then         -> chat showcase, training curves, README/model card numbers
# Waits on files (final.json), never on process names. Safe to re-run: finished
# stages are skipped. Logs go to $RUNS/finish.log.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-/home/user/data/runs}
TOK=${TOK:-/home/user/data/tokenized}
SFT_ARGS=(--data /home/user/data/sft --epochs 2 --max-minutes 40 --threads 2)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for() { until [ -f "$1" ]; do sleep 30; done; }
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
  wait_for "$RUNS/$c/final.json"
  [ -f "checkpoints/$c.pt" ] || python3 scripts/export_checkpoint.py "$RUNS/$c" "checkpoints/$c.pt"
  log "$c: exported checkpoints/$c.pt"
}

log "waiting for control"
export_base control
chat_tune control &
CONTROL_CHAT=$!

log "waiting for main"
export_base main
wait "$CONTROL_CHAT"
chat_tune main &
MAIN_CHAT=$!
if [ ! -f evals/results/eval.json ] || [ "${REEVAL:-0}" = 1 ]; then
  log "full evaluation"
  python3 scripts/evaluate.py --main checkpoints/main.pt --control checkpoints/control.pt \
    --out evals/results --threads 2 > "$RUNS/evaluate.log" 2>&1
  log "evaluation done"
fi
wait "$MAIN_CHAT"

log "chat showcase"
python3 scripts/chat_showcase.py --main checkpoints/main_chat.pt --control checkpoints/control_chat.pt \
  --out evals/results/chat_showcase.md --threads 4 > "$RUNS/showcase.log" 2>&1
log "training curves"
python3 scripts/eval_curve.py --main-run "$RUNS/main" --control-run "$RUNS/control" --out evals/results \
  --threads 4 > "$RUNS/curve.log" 2>&1
log "filling README and model card"
python3 scripts/fill_readme_results.py --data /home/user/data > "$RUNS/fill.log" 2>&1
log "ALL DONE"
