#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=celeba_stab
#SBATCH --output=/data/scratch/honjar/train_logs/%j_celeba_stab.out
#SBATCH --requeue

DATASET_NAME=$1
if [ -z "$DATASET_NAME" ]; then
    echo "Error: No dataset name provided."
    exit 1
fi

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion
export HF_HOME=/data/scratch/honjar/.cache/huggingface
export TORCH_HOME=/data/scratch/honjar/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_API_KEY=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

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

LATEST_STATE=$(ls /data/scratch/honjar/train_outputs/*-${DATASET_NAME}-*/training-state-*.pt 2>/dev/null | while read f; do echo "$(basename $f) $f"; done | sort | tail -1 | awk '{print $2}')
RESUME_FLAG=""
if [ -n "$LATEST_STATE" ]; then
    echo "Resuming from: $LATEST_STATE"
    RESUME_FLAG="--resume=$LATEST_STATE"
fi

echo "=== Stability Test: $DATASET_NAME (5k kimg) ==="
echo "Start time: $(date)"

python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=/data/scratch/honjar/train_outputs \
    --data=/data/scratch/honjar/annotated_datasets/${DATASET_NAME} \
    --cond=0 \
    --arch=ddpmpp \
    --batch=64 \
    --tick=50 \
    --snap=2 \
    --dump=2 \
    --corruption_probability=0.0 \
    --noise_config=identity \
    --s_max=4 \
    --cache=False \
    --duration=5 \
    $RESUME_FLAG

echo "End time: $(date)"
echo "Exit code: $?"
