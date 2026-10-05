#!/bin/bash
# Launch matched main + control pretraining runs (identical config, seed, steps).
cd /home/user/flight-ai-model-
for c in main control; do
  rm -rf /home/user/data/runs/$c
  mkdir -p /home/user/data/runs/$c
  nohup python3 scripts/train.py --config configs/cpu_pilot.yaml \
    --data-dir /home/user/data/tokenized/$c --out /home/user/data/runs/$c \
    --max-steps 850 --max-hours 3.6 > /home/user/data/runs/$c/stdout.log 2>&1 &
done
