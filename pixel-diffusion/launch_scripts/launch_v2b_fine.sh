#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Launch fine-sweep + T-convergence + baseline re-seed training jobs.
# 23 fine sweep + 4 T-convergence + 1 baseline re-seed = 28 jobs total.

TRAIN_SCRIPT="${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
DATASET_DIR="${AMBIENT_BASE}/annotated_datasets"
SEED=0
EXCLUDE="aia-h200-7"
LOG_DIR="${AMBIENT_BASE}/train_logs"

submitted=0
missing=0

submit_job() {
    local NAME=$1
    local JOB_SEED=${2:-$SEED}
    local DATASET_PATH="${DATASET_DIR}/${NAME}"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "MISSING: $DATASET_PATH"
        missing=$((missing + 1))
        return
    fi

    local JOB_LABEL="${NAME}"
    if [ "$JOB_SEED" != "0" ]; then
        JOB_LABEL="${NAME}_s${JOB_SEED}"
    fi

    sbatch --exclude=$EXCLUDE \
        --job-name="${JOB_LABEL}" \
        --output="${LOG_DIR}/%j_${JOB_LABEL}.out" \
        "$TRAIN_SCRIPT" "$NAME" $JOB_SEED

    submitted=$((submitted + 1))
}

echo "============================================"
echo "CelebA v2b Fine Sweep — Training Launcher"
echo "============================================"
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

# --- Fine sweep around minima ---
echo "--- Fine sweep: B1, B2 (around T=0.4 min) ---"
for b in 1 2; do
    for t in 030 035 045 050; do
        submit_job "celeba_v2b_b${b}_T${t}"
    done
done

echo ""
echo "--- Fine sweep: B3 (full 0.3-0.7 range) ---"
for t in 030 035 045 050 055 065 070; do
    submit_job "celeba_v2b_b3_T${t}"
done

echo ""
echo "--- Fine sweep: B4, B5 (around T=0.6 min) ---"
for b in 4 5; do
    for t in 050 055 065 070; do
        submit_job "celeba_v2b_b${b}_T${t}"
    done
done

# --- T -> 1 convergence check ---
echo ""
echo "--- T->1 convergence: B1, B5 ---"
for b in 1 5; do
    for t in 099 100; do
        submit_job "celeba_v2b_b${b}_T${t}"
    done
done

# --- Baseline re-seed ---
echo ""
echo "--- Baseline re-seed (seed=1) ---"
if [ ! -d "${DATASET_DIR}/celeba_v2b_baseline_s1" ]; then
    ln -s "${DATASET_DIR}/celeba_v2b_baseline" "${DATASET_DIR}/celeba_v2b_baseline_s1"
    echo "Created symlink: celeba_v2b_baseline_s1 -> celeba_v2b_baseline"
fi
submit_job "celeba_v2b_baseline_s1" 1

echo ""
echo "============================================"
echo "Submitted: $submitted jobs"
echo "Missing:   $missing datasets"
echo "============================================"
