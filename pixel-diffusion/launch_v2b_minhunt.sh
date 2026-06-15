#!/bin/bash
# Launch min hunt with dependency-chained eval.
# Each training job auto-triggers gen+eval on completion.

TRAIN_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
EVAL_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
DATASET_DIR="/data/scratch/honjar/annotated_datasets"
EXCLUDE="aia-h200-7"
LOG_DIR="/data/scratch/honjar/train_logs"

submitted=0
missing=0

submit_chained() {
    local NAME=$1
    local DATASET_PATH="${DATASET_DIR}/${NAME}"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "MISSING: $DATASET_PATH"
        missing=$((missing + 1))
        return
    fi

    TRAIN_ID=$(sbatch --parsable --exclude=$EXCLUDE \
        --job-name="${NAME}" \
        --output="${LOG_DIR}/%j_${NAME}.out" \
        "$TRAIN_SCRIPT" "$NAME" 0)

    sbatch --exclude=$EXCLUDE \
        --dependency=afterok:${TRAIN_ID} \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME"

    echo "  ${NAME}: train=$TRAIN_ID -> eval chained"
    submitted=$((submitted + 1))
}

echo "============================================"
echo "CelebA v2b Min Hunt — Chained Train+Eval"
echo "============================================"
DISK_USAGE=$(du -s /data/scratch/honjar/train_outputs/ 2>/dev/null | awk '{print $1}')
DISK_GB=$((DISK_USAGE / 1048576))
echo "Disk usage: ~${DISK_GB}GB"
if [ "$DISK_GB" -gt 600 ]; then
    echo "WARNING: Disk over 600GB. Clean first!"
    exit 1
fi
echo ""

echo "--- B1 (1 point) ---"
submit_chained "celeba_v2b_b1_T0475"

echo "--- B2 (3 points) ---"
for t in 0425 0475 0525; do
    submit_chained "celeba_v2b_b2_T${t}"
done

echo "--- B3 (2 points) ---"
for t in 0475 0525; do
    submit_chained "celeba_v2b_b3_T${t}"
done

echo "--- B4 (3 points) ---"
for t in 035 045 0475; do
    submit_chained "celeba_v2b_b4_T${t}"
done

echo "--- B5 (6 points) ---"
for t in 030 035 0375 0425 045 0475; do
    submit_chained "celeba_v2b_b5_T${t}"
done

echo "--- B6 (7 points) ---"
for t in 030 035 045 050 055 065 070; do
    submit_chained "celeba_v2b_b6_T${t}"
done

echo "--- B7 (7 points) ---"
for t in 030 035 045 050 055 065 070; do
    submit_chained "celeba_v2b_b7_T${t}"
done

echo ""
echo "============================================"
echo "Submitted: $submitted chained pairs (train+eval)"
echo "Missing:   $missing datasets"
echo "============================================"
