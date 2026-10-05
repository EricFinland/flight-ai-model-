#!/bin/bash
# Launch (or resume) the matched main + control pretraining runs.
# Identical config, seed and step count for both; safe to re-run after a crash
# or container restart: a run with an existing ckpt.pt resumes from it.
#   FRESH=1 scripts/launch_cpu_pilot.sh   # wipe both runs and start over
# Env: DATA (token files, default /home/user/data/tokenized), RUNS (default /home/user/data/runs).
# Returns at once; logs in $RUNS/{main,control}/stdout.log.
set -euo pipefail
case "${1:-}" in -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
cd "$(dirname "$0")/.."
RUNS=${RUNS:-/home/user/data/runs}
DATA=${DATA:-/home/user/data/tokenized}
for c in main control; do
  if pgrep -f -- "--out $RUNS/$c( |$)" > /dev/null; then
    echo "$c: already running"; continue
  fi
  [ "${FRESH:-0}" = 1 ] && rm -rf "$RUNS/$c"
  mkdir -p "$RUNS/$c"
  extra=()
  [ -f "$RUNS/$c/ckpt.pt" ] && extra=(--resume) && echo "$c: resuming"
  nohup python3 scripts/train.py --config configs/cpu_pilot.yaml \
    --data-dir "$DATA/$c" --out "$RUNS/$c" \
    --max-steps 850 --max-hours 3.6 --set ckpt_interval=25 "${extra[@]}" \
    >> "$RUNS/$c/stdout.log" 2>&1 &
  echo "$c: started (pid $!)"
done
