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
#SBATCH --job-name=val1k
#SBATCH --output=/data/scratch/honjar/train_logs/%j_val1k.out
#SBATCH --requeue

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-celeba_cleanonly_1000-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint for celeba_cleanonly_1000"
    exit 1
fi
echo "Using checkpoint: $CKPT"
python eval_val_loss.py --checkpoint=$CKPT --out_path=${AMBIENT_BASE}/generated/val_loss_celeba_cleanonly_1000.json
