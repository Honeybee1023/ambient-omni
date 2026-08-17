#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit 2-domain pilot training jobs — dogs and cats FIRST
SCRIPT=${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh

echo "=== Submitting dogs (priority 1) ==="
for t in 000 025 050 075 100; do
    echo "  pilot2d_dog_T${t}"
    sbatch $SCRIPT pilot2d_dog_T${t}
done

echo "=== Submitting cats (priority 2) ==="
for t in 000 025 050 075 100; do
    echo "  pilot2d_cat_T${t}"
    sbatch $SCRIPT pilot2d_cat_T${t}
done

echo "=== Submitting remaining categories ==="
for cat in tiger lion fox leopard cheetah; do
    for t in 000 025 050 075 100; do
        echo "  pilot2d_${cat}_T${t}"
        sbatch $SCRIPT pilot2d_${cat}_T${t}
    done
done

echo ""
echo "Submitted 35 jobs. Check: squeue -u honjar"
