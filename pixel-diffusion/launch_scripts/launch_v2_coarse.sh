#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Launch all 50 training jobs for v2 coarse 2D sweep.
# 1 baseline (all T=1) + 7 buckets x 7 T values = 50 jobs total.
# All use seed=0 for controlled comparison.

TRAIN_SCRIPT="${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
DATASET_DIR="${AMBIENT_BASE}/annotated_datasets"
SEED=0
EXCLUDE="aia-h200-7"
LOG_DIR="${AMBIENT_BASE}/train_logs"

submitted=0
missing=0

submit_job() {
    local NAME=$1
    local DATASET_PATH="${DATASET_DIR}/${NAME}"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "MISSING: $DATASET_PATH"
        missing=$((missing + 1))
        return
    fi

    sbatch --exclude=$EXCLUDE \
        --job-name="${NAME}" \
        --output="${LOG_DIR}/%j_${NAME}.out" \
        "$TRAIN_SCRIPT" "$NAME" $SEED

    submitted=$((submitted + 1))
}

echo "============================================"
echo "CelebA v2 Coarse Sweep — Training Launcher"
echo "============================================"
echo "Seed: $SEED"
echo ""

# Check disk first
DISK_USAGE=$(du -s ${AMBIENT_BASE}/train_outputs/ 2>/dev/null | awk '{print $1}')
DISK_GB=$((DISK_USAGE / 1048576))
echo "Disk usage: ~${DISK_GB}GB"
if [ "$DISK_GB" -gt 600 ]; then
    echo "WARNING: Disk over 600GB. Clean intermediate checkpoints first!"
    echo "Aborting."
    exit 1
fi
echo ""

# Baseline
echo "--- Baseline ---"
submit_job "celeba_v2_baseline"

# Coarse sweep: 7 buckets x 7 T values
echo ""
echo "--- Coarse 2D sweep ---"
for b in 1 2 3 4 5 6 7; do
    for t in 000 020 040 060 080 090 095; do
        submit_job "celeba_v2_b${b}_T${t}"
    done
done

echo ""
echo "============================================"
echo "Submitted: $submitted jobs"
echo "Missing:   $missing datasets"
echo "============================================"
