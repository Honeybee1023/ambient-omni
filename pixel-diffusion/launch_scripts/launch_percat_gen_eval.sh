#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit all 20 per-category gen+eval jobs
# Run this AFTER all training jobs have finished

SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_percat_gen_eval.sh

# First check how many 1000 kimg checkpoints exist
READY=$(ls ${AMBIENT_BASE}/train_outputs/*-percat_r1_model_*-*/network-snapshot-001000.pkl 2>/dev/null | wc -l)
echo "Found $READY / 20 models with 1000 kimg checkpoints."

if [ "$READY" -lt 20 ]; then
    echo "WARNING: Not all models finished training yet. Submitting only for completed models."
fi

for i in $(seq -w 0 19); do
    DATASET="percat_r1_model_0${i}"
    CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${DATASET}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "Submitting: $DATASET"
        sbatch $SCRIPT $DATASET
    else
        echo "SKIPPING: $DATASET (no 1000 kimg checkpoint yet)"
    fi
done

echo ""
echo "Check status with: squeue -u honjar"
