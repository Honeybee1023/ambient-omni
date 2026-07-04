#!/bin/bash
# Launch all CelebA training jobs in priority order:
# 1. Stability test (1 job, longer duration)
# 2. Coarse sweep (35 jobs)
# 3. Fine sweep (63 jobs)

SCRIPT_DIR=/data/scratch/honjar/ambient-omni/pixel-diffusion
ANNOTATED=/data/scratch/honjar/annotated_datasets

echo "=== CelebA Training Launcher ==="

# 1. Stability test: bucket 3 (sigma_blur=2.0) at T=0.5
echo ""
echo "--- Submitting stability test ---"
STAB_NAME="celeba_2d_b3_T050"
if [ -d "$ANNOTATED/$STAB_NAME" ]; then
    sbatch $SCRIPT_DIR/run_train_celeba_stability.sh $STAB_NAME
    echo "  Submitted: $STAB_NAME (5k kimg)"
else
    echo "  ERROR: Dataset $STAB_NAME not found!"
fi

# 2. Coarse sweep
echo ""
echo "--- Submitting coarse sweep (35 models) ---"
COARSE_COUNT=0
for b in 1 2 3 4 5 6 7; do
    for suffix in 000 025 050 075 100; do
        NAME="celeba_2d_b${b}_T${suffix}"
        # Skip the stability test model (already submitted with longer duration)
        if [ "$NAME" = "$STAB_NAME" ]; then
            echo "  SKIP (stability): $NAME"
            continue
        fi
        CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "  SKIP (done): $NAME"
            continue
        fi
        if [ -d "$ANNOTATED/$NAME" ]; then
            sbatch $SCRIPT_DIR/run_train_percat.sh $NAME
            COARSE_COUNT=$((COARSE_COUNT + 1))
        else
            echo "  ERROR: Dataset $NAME not found!"
        fi
    done
done
echo "  Submitted: $COARSE_COUNT coarse jobs"

# 3. Fine sweep
echo ""
echo "--- Submitting fine sweep (63 models) ---"
FINE_COUNT=0
for b in 1 2 3 4 5 6 7; do
    for suffix in 0125 0375 0625 080 085 090 095 097 099; do
        NAME="celeba_2d_b${b}_T${suffix}"
        CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "  SKIP (done): $NAME"
            continue
        fi
        if [ -d "$ANNOTATED/$NAME" ]; then
            sbatch $SCRIPT_DIR/run_train_percat.sh $NAME
            FINE_COUNT=$((FINE_COUNT + 1))
        else
            echo "  ERROR: Dataset $NAME not found!"
        fi
    done
done
echo "  Submitted: $FINE_COUNT fine jobs"

echo ""
echo "=== Total: 1 stability + $COARSE_COUNT coarse + $FINE_COUNT fine ==="
echo "Monitor with: squeue -u honjar"
