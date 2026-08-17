#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit all 20 per-category training jobs

SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh

for i in $(seq -w 0 19); do
    DATASET="percat_r1_model_0${i}"
    echo "Submitting: $DATASET"
    sbatch $SCRIPT $DATASET
done

echo ""
echo "All 20 jobs submitted. Check status with: squeue -u honjar"
