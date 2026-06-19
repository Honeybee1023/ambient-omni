#!/bin/bash
# Launch conditional sweep: B1 fixed at T=0.5, sweep B2
# 9 train+eval pairs with dependency chaining

SCRIPT_DIR="/data/scratch/honjar/ambient-omni/pixel-diffusion"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_2k_seeded.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/run_v2_gen_eval.sh"
LOG_DIR="/data/scratch/honjar/train_logs"

DATASETS=(
    celeba_v2b_cond_b2_T040
    celeba_v2b_cond_b2_T045
    celeba_v2b_cond_b2_T050
    celeba_v2b_cond_b2_T0525
    celeba_v2b_cond_b2_T055
    celeba_v2b_cond_b2_T0575
    celeba_v2b_cond_b2_T060
    celeba_v2b_cond_b2_T065
    celeba_v2b_cond_b2_T070
)

echo "Submitting ${#DATASETS[@]} train+eval pairs..."
echo ""

for NAME in "${DATASETS[@]}"; do
    TRAIN_ID=$(sbatch --parsable --exclude=aia-h200-7 \
        --job-name="${NAME}" \
        --output="${LOG_DIR}/%j_${NAME}.out" \
        "$TRAIN_SCRIPT" "$NAME" 0)

    EVAL_ID=$(sbatch --parsable --exclude=aia-h200-7 \
        --dependency=afterok:${TRAIN_ID} \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME")

    echo "  ${NAME}: train=${TRAIN_ID} -> eval=${EVAL_ID}"
done

echo ""
echo "Done! ${#DATASETS[@]} pairs submitted."
