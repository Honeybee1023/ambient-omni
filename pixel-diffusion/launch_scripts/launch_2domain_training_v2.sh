#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit training for fine-grained 2-domain T values.
# Priority: dogs/cats/fox fine-grained first, then midpoints, then rest.
SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh

COUNT=0

echo "=== Phase 1: Fine-grained (0.80-0.99) for dog, cat, fox ==="
for cat in dog cat fox; do
    for suffix in 080 085 090 095 097 099; do
        NAME="pilot2d_${cat}_T${suffix}"
        if [ -d "${AMBIENT_BASE}/annotated_datasets/${NAME}" ]; then
            echo "Submitting: $NAME"
            sbatch $SCRIPT $NAME
            COUNT=$((COUNT+1))
        else
            echo "MISSING: $NAME"
        fi
    done
done

echo ""
echo "=== Phase 2: Midpoints (0.125, 0.375, 0.625) for all categories ==="
for cat in dog cat tiger lion fox leopard cheetah; do
    for suffix in 0125 0375 0625; do
        NAME="pilot2d_${cat}_T${suffix}"
        if [ -d "${AMBIENT_BASE}/annotated_datasets/${NAME}" ]; then
            echo "Submitting: $NAME"
            sbatch $SCRIPT $NAME
            COUNT=$((COUNT+1))
        else
            echo "MISSING: $NAME"
        fi
    done
done

echo ""
echo "=== Phase 3: Fine-grained (0.80-0.99) for tiger, lion, leopard, cheetah ==="
for cat in tiger lion leopard cheetah; do
    for suffix in 080 085 090 095 097 099; do
        NAME="pilot2d_${cat}_T${suffix}"
        if [ -d "${AMBIENT_BASE}/annotated_datasets/${NAME}" ]; then
            echo "Submitting: $NAME"
            sbatch $SCRIPT $NAME
            COUNT=$((COUNT+1))
        else
            echo "MISSING: $NAME"
        fi
    done
done

echo ""
echo "Submitted $COUNT training jobs"
