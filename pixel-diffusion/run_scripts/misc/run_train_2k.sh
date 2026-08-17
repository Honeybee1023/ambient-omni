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
#SBATCH --job-name=2k_train
#SBATCH --output=${AMBIENT_BASE}/train_logs/%j_2k_train.out
#SBATCH --requeue

DATASET_NAME=$1

if [ -z "$DATASET_NAME" ]; then
    echo "Error: No dataset name provided."
    exit 1
fi

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_API_KEY=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

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

LATEST_STATE=$(ls ${AMBIENT_BASE}/train_outputs/*-${DATASET_NAME}-*/training-state-*.pt 2>/dev/null | while read f; do echo "$(basename $f) $f"; done | sort | tail -1 | awk '{print $2}')

RESUME_FLAG=""
if [ -n "$LATEST_STATE" ]; then
    echo "Resuming from: $LATEST_STATE"
    RESUME_FLAG="--resume=$LATEST_STATE"
fi

echo "=== Training 2k kimg: $DATASET_NAME ==="
echo "Start time: $(date)"

python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=${AMBIENT_BASE}/train_outputs \
    --data=${AMBIENT_BASE}/annotated_datasets/${DATASET_NAME} \
    --cond=0 \
    --arch=ddpmpp \
    --batch=64 \
    --tick=50 \
    --snap=5 \
    --dump=5 \
    --corruption_probability=0.0 \
    --noise_config=identity \
    --s_max=4 \
    --cache=False \
    --duration=2 \
    $RESUME_FLAG

echo "End time: $(date)"
echo "Exit code: $?"
