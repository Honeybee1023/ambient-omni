#!/bin/bash
# Submit training + gen+eval for a 2D BO round (v2b, MIND).
# Usage: bash launch_bo_2d.sh <round_number>

ROUND=$1
if [ -z "$ROUND" ]; then
    echo "Usage: bash launch_bo_2d.sh <round_number>"
    exit 1
fi

MANIFEST="/data/scratch/honjar/generated/bo2d_round${ROUND}_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    echo "Run bo_suggest_2d.py --round $ROUND first."
    exit 1
fi

TRAIN_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
EVAL_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
LOG_DIR="/data/scratch/honjar/train_logs"

# Disk check
DISK_USAGE=$(du -sh /data/scratch/honjar/train_outputs/ | awk '{print $1}')
echo "Disk usage: $DISK_USAGE (threshold: 600G)"
echo ""

# Parse dataset names from manifest
DATASETS=$(/data/scratch/honjar/miniconda3/envs/ambient/bin/python -c \
    "import json; d=json.load(open('$MANIFEST')); print(' '.join(d['datasets']))")

COUNT=$(echo $DATASETS | wc -w)
echo "=== 2D BO Round $ROUND: $COUNT datasets ==="
echo ""

for DS in $DATASETS; do
    echo "Submitting: $DS"

    TRAIN_JOB=$(sbatch --exclude=aia-h200-7 --parsable \
        --job-name="${DS}" \
        --output="${LOG_DIR}/%j_${DS}.out" \
        "$TRAIN_SCRIPT" "$DS" 0)
    echo "  Train: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --exclude=aia-h200-7 --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="eval_${DS}" \
        --output="${LOG_DIR}/%j_eval_${DS}.out" \
        "$EVAL_SCRIPT" "$DS")
    echo "  Eval:  $EVAL_JOB (after $TRAIN_JOB)"
done

echo ""
echo "=== Submitted $COUNT train+eval pairs ==="
echo "Monitor: squeue -u honjar"
echo "When done: python bo_suggest_2d.py --round $((ROUND+1)) --batch-size 10"
