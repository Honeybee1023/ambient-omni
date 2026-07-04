#!/bin/bash
TRAIN_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
EVAL_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
DATASET_DIR="/data/scratch/honjar/annotated_datasets"
EXCLUDE="aia-h200-7"
LOG_DIR="/data/scratch/honjar/train_logs"
submitted=0
missing=0

submit_chained() {
    local NAME=$1
    if [ ! -d "${DATASET_DIR}/${NAME}" ]; then
        echo "MISSING: $NAME"
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
    echo "  ${NAME}: train=${TRAIN_ID} -> eval chained"
    submitted=$((submitted + 1))
}

echo "============================================"
echo "CelebA v2b Min Hunt (17 chained train+eval)"
echo "============================================"
DISK_USAGE=$(du -s /data/scratch/honjar/train_outputs/ 2>/dev/null | awk '{print $1}')
DISK_GB=$((DISK_USAGE / 1048576))
echo "Disk: ~${DISK_GB}GB"
if [ "$DISK_GB" -gt 600 ]; then echo "Disk over 600GB!"; exit 1; fi

echo ""
echo "--- B1: 0.475, 0.525, 0.55 ---"
for t in 0475 0525 055; do submit_chained "celeba_v2b_b1_T${t}"; done

echo "--- B2: 0.425, 0.475 ---"
for t in 0425 0475; do submit_chained "celeba_v2b_b2_T${t}"; done

echo "--- B3: 0.475, 0.525 ---"
for t in 0475 0525; do submit_chained "celeba_v2b_b3_T${t}"; done

echo "--- B4: 0.45, 0.475, 0.525 ---"
for t in 045 0475 0525; do submit_chained "celeba_v2b_b4_T${t}"; done

echo "--- B5: 0.45, 0.475, 0.525 ---"
for t in 045 0475 0525; do submit_chained "celeba_v2b_b5_T${t}"; done

echo "--- B6: 0.5, 0.7 ---"
for t in 050 070; do submit_chained "celeba_v2b_b6_T${t}"; done

echo "--- B7: 0.5, 0.7 ---"
for t in 050 070; do submit_chained "celeba_v2b_b7_T${t}"; done

echo ""
echo "============================================"
echo "Submitted: $submitted chained pairs"
echo "Missing:   $missing"
echo "============================================"
