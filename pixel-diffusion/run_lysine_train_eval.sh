#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Run train (2k kimg) + eval (gen 5K + MIND + val loss) on lysine for one dataset.
# Usage: bash run_lysine_train_eval.sh <dataset_name> <gpu_id> [seed]

DATASET_NAME=$1
GPU_ID=$2
TRAIN_SEED=${3:-0}

if [ -z "$DATASET_NAME" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash run_lysine_train_eval.sh <dataset_name> <gpu_id> [seed]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_API_KEY=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

PYTHON=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
BASE=${AMBIENT_BASE}
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

# --- Auto-resume ---
LATEST_STATE=$(ls ${BASE}/train_outputs/*-${DATASET_NAME}-*/training-state-*.pt 2>/dev/null | while read f; do echo "$(basename $f) $f"; done | sort | tail -1 | awk '{print $2}')
RESUME_FLAG=""
if [ -n "$LATEST_STATE" ]; then
    echo "Resuming from: $LATEST_STATE"
    RESUME_FLAG="--resume=$LATEST_STATE"
fi

echo "=== Training 2k kimg: $DATASET_NAME (seed=$TRAIN_SEED, GPU=$GPU_ID) ==="
echo "Start time: $(date)"

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=${BASE}/train_outputs \
    --data=${BASE}/annotated_datasets/${DATASET_NAME} \
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
    --seed=$TRAIN_SEED \
    $RESUME_FLAG

TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"
if [ $TRAIN_EXIT -ne 0 ]; then echo "ERROR: Training failed"; exit 1; fi

# --- Eval ---
TRAIN_DIR=$(ls -td ${BASE}/train_outputs/*-${DATASET_NAME}-* 2>/dev/null | head -1)
CKPT=$(ls ${TRAIN_DIR}/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No 2k checkpoint found in $TRAIN_DIR"
    exit 1
fi

OUTDIR="${BASE}/generated/${DATASET_NAME}_2000kimg"
HOLDOUT_DIR="${BASE}/celeba_processed_v2b/holdout_64"
MIND_REF_CACHE="${BASE}/generated/mind_ref_cache.npz"
MIND_JSON="${BASE}/generated/mind_${DATASET_NAME}_2000kimg.json"
VALLOSS_JSON="${BASE}/generated/valloss_${DATASET_NAME}_2000kimg.json"

echo ""
echo "=== Eval: $DATASET_NAME ==="
echo "Checkpoint: $CKPT"

# Gen 5K images
mkdir -p "$OUTDIR"
EXISTING=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
if [ "$EXISTING" -ge 5000 ]; then
    echo "Images already generated (${EXISTING} found), skipping."
else
    echo "Generating 5000 images..."
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT \
        --outdir=$OUTDIR \
        --seeds=0-4999 \
        --batch=64
    if [ $? -ne 0 ]; then echo "ERROR: Generation failed"; exit 1; fi
    echo "Generated $(ls "$OUTDIR"/*.png | wc -l) images"
fi

# MIND
if [ -f "$MIND_JSON" ]; then
    echo "MIND already computed, skipping."
else
    echo "Computing MIND..."
    $PYTHON eval_mind.py \
        --gen_path=$OUTDIR \
        --ref_path=$HOLDOUT_DIR \
        --ref_cache=$MIND_REF_CACHE \
        --out_path=$MIND_JSON
    if [ $? -ne 0 ]; then echo "ERROR: MIND eval failed"; exit 1; fi
fi

# Val Loss
if [ -f "$VALLOSS_JSON" ]; then
    echo "Val loss already computed, skipping."
else
    echo "Computing val loss..."
    $PYTHON eval_val_loss.py \
        --checkpoint=$CKPT \
        --holdout_dir=$HOLDOUT_DIR \
        --out_path=$VALLOSS_JSON
    if [ $? -ne 0 ]; then echo "ERROR: Val loss eval failed"; exit 1; fi
fi

echo ""
echo "=== Results: $DATASET_NAME ==="
if [ -f "$MIND_JSON" ]; then
    echo "MIND: $($PYTHON -c "import json; print(json.load(open('$MIND_JSON'))['mind'])")"
fi
if [ -f "$VALLOSS_JSON" ]; then
    echo "Val loss: $($PYTHON -c "import json; print(json.load(open('$VALLOSS_JSON'))['weighted_loss_mean'])")"
fi
echo "End time: $(date)"
