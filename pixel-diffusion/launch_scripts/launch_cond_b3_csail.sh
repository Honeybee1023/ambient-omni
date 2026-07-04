#!/bin/bash
# Launch 5 cond_b3 train+eval pairs on CSAIL SLURM
# Fine points near B3 minimum: T=0.4, 0.45, 0.5, 0.55, 0.6

TRAIN_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
EVAL_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
LOG_DIR="/data/scratch/honjar/train_logs"
EXCLUDE="aia-h200-7"

DATASETS=(
    "celeba_v2b_cond_b3_T040"
    "celeba_v2b_cond_b3_T045"
    "celeba_v2b_cond_b3_T050"
    "celeba_v2b_cond_b3_T055"
    "celeba_v2b_cond_b3_T060"
)

for NAME in "${DATASETS[@]}"; do
    echo "Submitting $NAME ..."
    TRAIN_ID=$(sbatch --parsable --exclude=$EXCLUDE \
        --job-name="${NAME}" \
        --output="${LOG_DIR}/%j_${NAME}.out" \
        "$TRAIN_SCRIPT" "$NAME" 0)
    echo "  Train job: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable --exclude=$EXCLUDE \
        --dependency=afterok:${TRAIN_ID} \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME")
    echo "  Eval job: $EVAL_ID (depends on $TRAIN_ID)"
done

echo ""
echo "=== Submitted 5 train+eval pairs ==="
