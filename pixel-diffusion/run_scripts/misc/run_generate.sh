#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=generate
#SBATCH --output=${AMBIENT_BASE}/train_logs/generate_%j.out

CHECKPOINT=$1
OUTDIR=$2

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export WANDB_MODE=disabled

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

echo "Generating from checkpoint: $CHECKPOINT"
echo "Output directory: $OUTDIR"

python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CHECKPOINT \
    --outdir=$OUTDIR \
    --seeds=0-999 \
    --batch=64

echo "Exit code: $?"