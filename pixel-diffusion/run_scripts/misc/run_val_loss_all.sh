#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=val_loss
#SBATCH --output=/data/scratch/honjar/train_logs/%j_val_loss.out
#SBATCH --requeue

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

for NAME in celeba_cleanonly_500 celeba_cleanonly_750 celeba_cleanonly_1000 celeba_cleanonly_2000 celeba_cleanonly_5000 celeba_cleanonly_10000 celeba_cleanonly_22k; do
    CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
    OUT="${AMBIENT_BASE}/generated/val_loss_${NAME}.json"
    if [ -z "$CKPT" ]; then
        echo "No checkpoint for $NAME, skipping"
        continue
    fi
    if [ -f "$OUT" ]; then
        echo "Skipping $NAME (already done)"
        continue
    fi
    echo "=== $NAME ==="
    python eval_val_loss.py --checkpoint=$CKPT --out_path=$OUT
done
echo "=== All done ==="
