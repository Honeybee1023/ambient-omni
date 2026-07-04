#!/bin/bash
# launch_v2b_independence.sh
# Submits 13 training+eval pairs with dependency chaining.
# Order: Test 1 (pairwise best) → Test 2 (shifts) → Test 3 (conditional sweep)

set -e

PYTHON="/data/scratch/honjar/miniconda3/envs/ambient/bin/python"
SCRIPT_DIR="/data/scratch/honjar/ambient-omni/pixel-diffusion"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_2k_seeded.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/run_v2_gen_eval.sh"
LOG_DIR="/data/scratch/honjar/train_logs"
EXCLUDE="aia-h200-7"
SEED=0

# === Pre-flight checks ===
echo "=== Disk check ==="
du -sh /data/scratch/honjar/train_outputs/
echo ""

for f in "$TRAIN_SCRIPT" "$EVAL_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing script: $f"
        exit 1
    fi
done

# === Create datasets ===
echo "=== Creating datasets ==="
$PYTHON ${SCRIPT_DIR}/create_v2b_indep_datasets.py
echo ""

# === Submit jobs ===
echo "=== Submitting jobs ==="
TOTAL=0

submit_pair() {
    local NAME=$1
    local LABEL=$2
    TRAIN_ID=$(sbatch --parsable --exclude=$EXCLUDE \
        --job-name="train_${NAME}" \
        --output="${LOG_DIR}/%j_train_${NAME}.out" \
        "$TRAIN_SCRIPT" "$NAME" $SEED)
    EVAL_ID=$(sbatch --parsable --exclude=$EXCLUDE \
        --dependency=afterok:${TRAIN_ID} \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME")
    echo "  ${LABEL}: ${NAME}  (train=${TRAIN_ID}, eval=${EVAL_ID})"
    TOTAL=$((TOTAL + 1))
}

# --- Test 1: pairwise best (B1=0.5, B2=0.55) ---
submit_pair "celeba_v2b_cond_b1_T050" "Test1"

# --- Test 2: shifts ---
submit_pair "celeba_v2b_shift_bothup" "Test2"
submit_pair "celeba_v2b_shift_bothdn" "Test2"
submit_pair "celeba_v2b_shift_apart"  "Test2"
submit_pair "celeba_v2b_shift_close"  "Test2"

# --- Test 3: conditional sweep (skip T050, already submitted as Test 1) ---
for T in 000 020 040 045 055 060 080 095; do
    submit_pair "celeba_v2b_cond_b1_T${T}" "Test3"
done

echo ""
echo "=== Submitted ${TOTAL} train+eval pairs (26 jobs total) ==="
echo ""
echo "Monitor:  squeue -u honjar"
echo "Logs:     ls -lt ${LOG_DIR}/ | head -30"
