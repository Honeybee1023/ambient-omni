#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_diffusion
#SBATCH --output=/data/scratch/honjar/train_logs/%j.out

DATASET_PATH=$1
EXPR_ID=$2

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_API_KEY=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

mkdir -p ${AMBIENT_BASE}/train_logs
mkdir -p ${AMBIENT_BASE}/train_outputs

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

echo "Job started on $(hostname)"
echo "Dataset: $DATASET_PATH"
echo "Experiment: $EXPR_ID"

python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=${AMBIENT_BASE}/train_outputs \
    --data=$DATASET_PATH \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=20 \
    --batch=64 \
    --batch-gpu=64 \
    --dump=40 \
    --corruption_probability=0.0 \
    --workers=4 \
    --expr_id=$EXPR_ID \
    --s_max=4

echo "Exit code: $?"