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
#SBATCH --job-name=bs_eval
#SBATCH --output=${AMBIENT_BASE}/train_logs/%j_bs_eval.out

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

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

GEN_BASE=${AMBIENT_BASE}/generated

# Find the 1000 kimg checkpoint
CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${DATASET_NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No 1000 kimg checkpoint found for $DATASET_NAME"
    exit 1
fi

OUTDIR="${GEN_BASE}/${DATASET_NAME}_001000kimg"
OUTJSON="${GEN_BASE}/metrics_${DATASET_NAME}_1000kimg.json"

echo "=== Generate + Evaluate: $DATASET_NAME ==="
echo "Checkpoint: $CKPT"
echo "Output dir: $OUTDIR"

# Step 1: Generate 1000 images (skip if already done)
if [ -d "$OUTDIR" ] && [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -ge 1000 ]; then
    echo "Images already generated, skipping."
else
    echo "Generating 1000 images..."
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT \
        --outdir=$OUTDIR \
        --seeds=0-999 \
        --batch=64

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed"
        exit 1
    fi
fi

# Step 2: Evaluate metrics (skip if already done)
if [ -f "$OUTJSON" ]; then
    echo "Metrics already computed, skipping."
else
    echo "Computing PickScore + Aesthetic + Vendi..."
    python eval_new_metrics.py \
        --image_dir $OUTDIR \
        --prompt "a photograph of a wolf" \
        --max_images 1000 \
        --output_json $OUTJSON
fi

echo ""
echo "=== Done: $DATASET_NAME ==="
echo "Results: $OUTJSON"
