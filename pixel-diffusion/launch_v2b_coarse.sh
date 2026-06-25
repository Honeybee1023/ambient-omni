#!/bin/bash
TRAIN_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_2k_seeded.sh"
DATASET_DIR="/data/scratch/honjar/annotated_datasets"
SEED=0
EXCLUDE="aia-h200-7"
LOG_DIR="/data/scratch/honjar/train_logs"
submitted=0; missing=0

echo "============================================"
echo "CelebA v2b Coarse Sweep — Training Launcher"
echo "============================================"

DISK_GB=$(( $(du -s /data/scratch/honjar/train_outputs/ 2>/dev/null | awk '{print $1}') / 1048576 ))
echo "Disk: ~${DISK_GB}GB"
if [ "$DISK_GB" -gt 600 ]; then echo "Disk over 600GB, aborting."; exit 1; fi

submit_job() {
    local NAME=$1
    if [ ! -d "${DATASET_DIR}/${NAME}" ]; then
        echo "MISSING: $NAME"; missing=$((missing+1)); return; fi
    sbatch --exclude=$EXCLUDE --job-name="${NAME}" \
        --output="${LOG_DIR}/%j_${NAME}.out" "$TRAIN_SCRIPT" "$NAME" $SEED
    submitted=$((submitted+1))
}

echo "--- Baseline ---"
submit_job "celeba_v2b_baseline"

echo "--- Coarse sweep ---"
for b in 1 2 3 4 5 6 7; do
    for t in 000 020 040 060 080 090 095; do
        submit_job "celeba_v2b_b${b}_T${t}"
    done
done

echo ""; echo "Submitted: $submitted | Missing: $missing"
