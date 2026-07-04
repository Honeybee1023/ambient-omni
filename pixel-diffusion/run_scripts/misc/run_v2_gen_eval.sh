#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=v2_eval
#SBATCH --output=/data/scratch/honjar/train_logs/%j_v2_eval.out
#SBATCH --requeue

# Gen 5K images + MIND + Val Loss for a v2 model.
# Usage: sbatch run_v2_gen_eval.sh <dataset_name>

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

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

GEN_BASE=/data/scratch/honjar/generated
HOLDOUT_DIR=/data/scratch/honjar/celeba_processed/holdout_64
MIND_REF_CACHE=/data/scratch/honjar/celeba_processed/inception_holdout_feats.npz

# Find 2k kimg checkpoint
CKPT=$(ls /data/scratch/honjar/train_outputs/*-${DATASET_NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No 2k kimg checkpoint found for $DATASET_NAME"
    exit 1
fi

OUTDIR="${GEN_BASE}/${DATASET_NAME}_5k_gen"
MIND_JSON="${GEN_BASE}/mind_${DATASET_NAME}_2000kimg.json"
VALLOSS_JSON="${GEN_BASE}/val_loss_${DATASET_NAME}_2000kimg.json"

echo "=== V2 Gen+Eval: $DATASET_NAME ==="
echo "Checkpoint: $CKPT"
echo "Start time: $(date)"

# --- Step 1: Generate 5K images ---
if [ -d "$OUTDIR" ] && [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -ge 5000 ]; then
    echo "Images already generated (5K found), skipping."
else
    echo "Generating 5000 images..."
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT \
        --outdir=$OUTDIR \
        --seeds=0-4999 \
        --batch=64
    if [ $? -ne 0 ]; then echo "ERROR: Generation failed"; exit 1; fi
    echo "Generated $(ls "$OUTDIR"/*.png | wc -l) images"
fi

# --- Step 2: MIND ---
if [ -f "$MIND_JSON" ]; then
    echo "MIND already computed, skipping."
else
    echo "Computing MIND..."
    python eval_mind.py \
        --gen_path=$OUTDIR \
        --ref_path=$HOLDOUT_DIR \
        --ref_cache=$MIND_REF_CACHE \
        --out_path=$MIND_JSON
    if [ $? -ne 0 ]; then echo "ERROR: MIND eval failed"; exit 1; fi
fi

# --- Step 3: Val Loss ---
if [ -f "$VALLOSS_JSON" ]; then
    echo "Val loss already computed, skipping."
else
    echo "Computing val loss..."
    python eval_val_loss.py \
        --checkpoint=$CKPT \
        --holdout_dir=$HOLDOUT_DIR \
        --out_path=$VALLOSS_JSON
    if [ $? -ne 0 ]; then echo "ERROR: Val loss eval failed"; exit 1; fi
fi

echo ""
echo "=== Results: $DATASET_NAME ==="
if [ -f "$MIND_JSON" ]; then
    echo "MIND: $(python -c "import json; print(json.load(open('$MIND_JSON'))['mind'])")"
fi
if [ -f "$VALLOSS_JSON" ]; then
    echo "Val loss: $(python -c "import json; print(json.load(open('$VALLOSS_JSON'))['weighted_loss_mean'])")"
fi
echo "End time: $(date)"
