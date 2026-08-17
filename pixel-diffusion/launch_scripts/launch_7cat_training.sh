#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit 7-category one-at-a-time training jobs.
# Priority: dog/cat/fox first (most signal), fine-grained T values first.
SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh
COUNT=0

echo "=== 7-category one-at-a-time experiment ==="
for cat in dog cat fox tiger lion leopard cheetah; do
    for suffix in 090 091 092 093 094 095 097 099 085 080 075 0625 050 0375 025 0125 000 100; do
        NAME="exp7d_${cat}_T${suffix}"
        if [ -d "${AMBIENT_BASE}/annotated_datasets/${NAME}" ]; then
            # Skip if already has checkpoint
            CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
            if [ -n "$CKPT" ]; then
                echo "DONE: $NAME"
            else
                echo "Submitting: $NAME"
                sbatch $SCRIPT $NAME
                COUNT=$((COUNT+1))
            fi
        fi
    done
done
echo ""
echo "Submitted $COUNT training jobs"
