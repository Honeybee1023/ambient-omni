#!/bin/bash
# Submit all 20 per-category training jobs

SCRIPT=/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_percat.sh

for i in $(seq -w 0 19); do
    DATASET="percat_r1_model_0${i}"
    echo "Submitting: $DATASET"
    sbatch $SCRIPT $DATASET
done

echo ""
echo "All 20 jobs submitted. Check status with: squeue -u honjar"
