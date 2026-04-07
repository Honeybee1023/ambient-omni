#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_diffusion
#SBATCH --output=/data/scratch/honjar/train_logs/%j.out
#SBATCH --requeue

DATASET_NAME=$1

if [ -z "$DATASET_NAME" ]; then
    echo "Error: No dataset name provided."
    echo "Usage: sbatch run_train_all.sh <dataset_name>"
    exit 1
fi

DATASET_PATH="/data/scratch/honjar/annotated_datasets/${DATASET_NAME}"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_API_KEY=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

mkdir -p /data/scratch/honjar/train_logs
mkdir -p /data/scratch/honjar/train_outputs

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

# --- Auto-resume logic (FIXED: searches ALL matching dirs, not just the last one) ---
RESUME_FLAG=""
LATEST_STATE=$(ls -t /data/scratch/honjar/train_outputs/*-${DATASET_NAME}-*/training-state-*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_STATE" ]; then
    echo "Resuming from: $LATEST_STATE"
    RESUME_FLAG="--resume=$LATEST_STATE"
else
    echo "No training states found. Starting fresh."
fi

echo "Job started on $(hostname)"
echo "Dataset: $DATASET_PATH"
echo "Experiment: $DATASET_NAME"

python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=/data/scratch/honjar/train_outputs \
    --data=$DATASET_PATH \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=20 \
    --batch=64 \
    --batch-gpu=64 \
    --snap=10 \
    --dump=10 \
    --corruption_probability=0.0 \
    --workers=4 \
    --expr_id=$DATASET_NAME \
    --s_max=4 \
    $RESUME_FLAG

echo "Exit code: $?"
