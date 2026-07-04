#!/bin/bash
# Submit gen+eval for all CelebA models that have finished training but don't have metrics yet

REF_STATS=/data/scratch/honjar/celeba_processed/celeba_holdout_ref_stats.npz
if [ ! -f "$REF_STATS" ]; then
    echo "ERROR: Reference stats not found. Wait for precompute job to finish."
    exit 1
fi

SCRIPT=/data/scratch/honjar/ambient-omni/pixel-diffusion/run_celeba_gen_eval.sh
COUNT=0

for b in 1 2 3 4 5 6 7; do
    for suffix in 000 0125 025 0375 050 0625 075 080 085 090 095 097 099 100; do
        NAME="celeba_2d_b${b}_T${suffix}"
        JSON="/data/scratch/honjar/generated/metrics_${NAME}_1000kimg.json"
        CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ] && [ ! -f "$JSON" ]; then
            sbatch $SCRIPT $NAME
            COUNT=$((COUNT + 1))
        fi
    done
done

echo "Submitted $COUNT gen+eval jobs"
