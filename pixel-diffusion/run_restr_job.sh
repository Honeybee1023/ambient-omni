#!/bin/bash
# One restricted-bucket experiment: train 2000 kimg, generate 5K, MIND + val loss.
# Usage: bash run_restr_job.sh <dataset_name> <gpu_id> [seed]
#
# Modelled on run_lysine_train_eval.sh, with one deliberate difference: the
# wandb key is read from $AMBIENT_BASE/.wandb_key instead of being baked in.
# The key hardcoded in the older run_scripts is committed to a PUBLIC repo and
# needs rotating; nothing new should carry it.
#
# Every stage is skip-if-done, so re-running after an interruption resumes
# rather than redoing work. That is what makes the queue safe to restart.

set -u

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

DATASET_NAME=${1:-}
GPU_ID=${2:-}
TRAIN_SEED=${3:-0}

if [ -z "$DATASET_NAME" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash run_restr_job.sh <dataset_name> <gpu_id> [seed]"
    exit 2
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
# Derived from the GPU id so four concurrent jobs cannot collide on a port.
export MASTER_PORT=$((29500 + GPU_ID * 7 + RANDOM % 5))

if [ -f "${AMBIENT_BASE}/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "${AMBIENT_BASE}/.wandb_key")
else
    export WANDB_MODE=offline
fi

PYTHON=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
BASE=${AMBIENT_BASE}
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion || exit 1

HOLDOUT_DIR="${BASE}/celeba_processed_v2b/holdout_64"
MIND_REF_CACHE="${BASE}/generated/mind_ref_cache.npz"
OUTDIR="${BASE}/generated/${DATASET_NAME}_2000kimg"
MIND_JSON="${BASE}/generated/mind_${DATASET_NAME}_2000kimg.json"
VALLOSS_JSON="${BASE}/generated/valloss_${DATASET_NAME}_2000kimg.json"

echo "=== $DATASET_NAME | GPU $GPU_ID | seed $TRAIN_SEED | $(date) ==="

# --- Train (auto-resume from the newest training-state if one exists) ---
if [ -f "$MIND_JSON" ] && [ -f "$VALLOSS_JSON" ]; then
    echo "Both metrics already present, nothing to do."
    exit 0
fi

LATEST_STATE=$(ls ${BASE}/train_outputs/*-${DATASET_NAME}-*/training-state-*.pt 2>/dev/null \
    | while read f; do echo "$(basename $f) $f"; done | sort | tail -1 | awk '{print $2}')
RESUME_FLAG=""
if [ -n "$LATEST_STATE" ]; then
    echo "Resuming from: $LATEST_STATE"
    RESUME_FLAG="--resume=$LATEST_STATE"
fi

TRAIN_DIR=$(ls -td ${BASE}/train_outputs/*-${DATASET_NAME}-* 2>/dev/null | head -1)
CKPT=$(ls ${TRAIN_DIR}/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)

if [ -z "$CKPT" ]; then
    echo "--- Training 2000 kimg ---"
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
    if [ $? -ne 0 ]; then echo "ERROR: training failed for $DATASET_NAME"; exit 1; fi

    TRAIN_DIR=$(ls -td ${BASE}/train_outputs/*-${DATASET_NAME}-* 2>/dev/null | head -1)
    CKPT=$(ls ${TRAIN_DIR}/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
fi

if [ -z "$CKPT" ]; then echo "ERROR: no 2k checkpoint for $DATASET_NAME"; exit 1; fi
echo "Checkpoint: $CKPT"

# --- Generate 5K ---
mkdir -p "$OUTDIR"
EXISTING=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
if [ "$EXISTING" -ge 5000 ]; then
    echo "Images already generated ($EXISTING), skipping."
else
    echo "--- Generating 5000 images ---"
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT --outdir=$OUTDIR --seeds=0-4999 --batch=64
    if [ $? -ne 0 ]; then echo "ERROR: generation failed for $DATASET_NAME"; exit 1; fi
fi

# --- Metrics ---
if [ ! -f "$MIND_JSON" ]; then
    echo "--- MIND ---"
    $PYTHON eval_mind.py --gen_path=$OUTDIR --ref_path=$HOLDOUT_DIR \
        --ref_cache=$MIND_REF_CACHE --out_path=$MIND_JSON
    if [ $? -ne 0 ]; then echo "ERROR: MIND failed for $DATASET_NAME"; exit 1; fi
fi

if [ ! -f "$VALLOSS_JSON" ]; then
    echo "--- Val loss ---"
    $PYTHON eval_val_loss.py --checkpoint=$CKPT --holdout_dir=$HOLDOUT_DIR \
        --out_path=$VALLOSS_JSON
    if [ $? -ne 0 ]; then echo "ERROR: val loss failed for $DATASET_NAME"; exit 1; fi
fi

# --- Reclaim disk: intermediate snapshots are ~7G per run and never reused. ---
if [ -n "$TRAIN_DIR" ]; then
    find "$TRAIN_DIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" \
        -o -name "training-state-*.pt" \) -delete
fi

echo "=== DONE $DATASET_NAME | MIND=$($PYTHON -c "import json;print(json.load(open('$MIND_JSON'))['mind'])") | $(date) ==="
