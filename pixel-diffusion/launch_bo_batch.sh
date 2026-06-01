#!/bin/bash
# Submit training + gen+eval for a BO round.
# Usage: bash launch_bo_batch.sh <round_number>

ROUND=$1
if [ -z "$ROUND" ]; then
    echo "Usage: bash launch_bo_batch.sh <round_number>"
    exit 1
fi

MANIFEST="/data/scratch/honjar/generated/bo_round${ROUND}_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    echo "Run bo_suggest.py --round $ROUND first."
    exit 1
fi

# Disk check
DISK_USAGE=$(du -sh /data/scratch/honjar/train_outputs/ | awk '{print $1}')
echo "Disk usage: $DISK_USAGE (threshold: 600G)"
echo ""

# Parse dataset names from manifest
DATASETS=$(/data/scratch/honjar/miniconda3/envs/ambient/bin/python -c \
    "import json; d=json.load(open('$MANIFEST')); print(' '.join(d['datasets']))")

COUNT=$(echo $DATASETS | wc -w)
echo "=== BO Round $ROUND: $COUNT datasets ==="
echo ""

for DS in $DATASETS; do
    echo "Submitting: $DS"

    # Training (2k kimg)
    TRAIN_JOB=$(sbatch --exclude=aia-h200-7 --parsable \
        /data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k.sh "$DS")
    echo "  Train: $TRAIN_JOB"

    # Gen+eval chained after training completes
    EVAL_JOB=$(sbatch --exclude=aia-h200-7 --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        /data/scratch/honjar/ambient-omni/pixel-diffusion/run_celeba_gen_eval_2k.sh "$DS")
    echo "  Eval:  $EVAL_JOB (after $TRAIN_JOB)"
done

echo ""
echo "=== Submitted $COUNT train+eval pairs ==="
echo "Monitor: squeue -u honjar"
echo "When done, run: python bo_suggest.py --round $((ROUND+1)) --batch-size 15"
