#!/bin/bash
# Submit gen+eval for ALL 2-domain pilot models (old + new T values).
SCRIPT=/data/scratch/honjar/ambient-omni/pixel-diffusion/run_percat_gen_eval.sh

READY=0
SKIP=0
TOTAL=0

for cat in dog cat tiger lion fox leopard cheetah; do
    echo "=== $cat ==="
    for suffix in 000 0125 025 0375 050 0625 075 080 085 090 095 097 099 100; do
        NAME="pilot2d_${cat}_T${suffix}"
        TOTAL=$((TOTAL+1))

        # Check if dataset exists
        if [ ! -d "/data/scratch/honjar/annotated_datasets/${NAME}" ]; then
            continue
        fi

        # Check for checkpoint
        CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            # Check if metrics already computed
            OUTJSON="/data/scratch/honjar/generated/metrics_${NAME}_1000kimg.json"
            if [ -f "$OUTJSON" ]; then
                echo "  DONE: $NAME (metrics exist)"
            else
                echo "  Submitting: $NAME"
                sbatch $SCRIPT $NAME
                READY=$((READY+1))
            fi
        else
            echo "  SKIP: $NAME (no checkpoint yet)"
            SKIP=$((SKIP+1))
        fi
    done
done

echo ""
echo "$READY submitted, $SKIP waiting for training"
