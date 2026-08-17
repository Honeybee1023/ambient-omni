#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit gen+eval for completed 2-domain pilot models
SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_percat_gen_eval.sh

READY=0
TOTAL=0
for cat in dog cat tiger lion fox leopard cheetah; do
    for t in 000 025 050 075 100; do
        NAME="pilot2d_${cat}_T${t}"
        TOTAL=$((TOTAL+1))
        CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "Submitting: $NAME"
            sbatch $SCRIPT $NAME
            READY=$((READY+1))
        else
            echo "SKIPPING: $NAME (no checkpoint)"
        fi
    done
done
echo ""
echo "$READY / $TOTAL models ready for gen+eval"
