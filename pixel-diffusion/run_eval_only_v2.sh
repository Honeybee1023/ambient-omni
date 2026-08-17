#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

GPU_ID=$1; shift; DATASETS=("$@")
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export WANDB_MODE=disabled
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + GPU_ID))
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion
HOLDOUT=${AMBIENT_BASE}/celeba_processed_v2b/holdout_64
MIND_CACHE=${AMBIENT_BASE}/generated/inception_holdout_feats.npz
for NAME in "${DATASETS[@]}"; do
  echo "=== EVAL $NAME on GPU $GPU_ID ($(date)) ==="
  CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then echo "NO CKPT: $NAME"; continue; fi
  OUTDIR=${AMBIENT_BASE}/generated/${NAME}_5k_gen
  MIND_JSON=${AMBIENT_BASE}/generated/mind_${NAME}_2000kimg.json
  if [ ! -d "$OUTDIR" ] || [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
    rm -rf "$OUTDIR"
    python generate.py --network=$CKPT --outdir=$OUTDIR --seeds=0-4999 --batch=64
    if [ $? -ne 0 ]; then echo "GEN FAILED: $NAME"; continue; fi
  fi
  if [ ! -f "$MIND_JSON" ]; then
    python eval_mind.py --gen_path=$OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  fi
  echo "EVAL DONE: $NAME ($(date))"
done
echo "GPU $GPU_ID EVALS COMPLETE ($(date))"
