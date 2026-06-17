#!/bin/bash
GPU_ID=$1; shift; DATASETS=("$@")
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=/data/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/honjar/ambient-omni/pixel-diffusion
export TORCH_HOME=/data/honjar/.cache/torch
export HF_HOME=/data/honjar/.cache/huggingface
export WANDB_MODE=disabled
export MASTER_ADDR=localhost
cd /data/honjar/ambient-omni/pixel-diffusion
HOLDOUT=/data/honjar/celeba_processed_v2b/holdout_64
MIND_CACHE=/data/honjar/generated/inception_holdout_feats.npz

for NAME in "${DATASETS[@]}"; do
  echo ""; echo "==============================="
  echo "GPU $GPU_ID: $NAME ($(date))"
  echo "==============================="
  export MASTER_PORT=$((RANDOM % 5000 + 10000))
  # --- TRAIN ---
  python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=/data/honjar/train_outputs \
    --data=/data/honjar/annotated_datasets/${NAME} \
    --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=2 --seed=0
  if [ $? -ne 0 ]; then echo "TRAIN FAILED: $NAME"; continue; fi
  # --- EVAL ---
  CKPT=$(ls /data/honjar/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then echo "NO CHECKPOINT: $NAME"; continue; fi
  OUTDIR=/data/honjar/generated/${NAME}_5k_gen
  MIND_JSON=/data/honjar/generated/mind_${NAME}_2000kimg.json
  VALLOSS_JSON=/data/honjar/generated/val_loss_${NAME}_2000kimg.json
  # Generate 5K
  if [ ! -d "$OUTDIR" ] || [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
    export MASTER_PORT=$((RANDOM % 5000 + 10000))
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
      --network=$CKPT --outdir=$OUTDIR --seeds=0-4999 --batch=64
  fi
  # MIND
  if [ ! -f "$MIND_JSON" ]; then
    python eval_mind.py --gen_path=$OUTDIR --ref_path=$HOLDOUT \
      --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  fi
  # Val loss
  if [ ! -f "$VALLOSS_JSON" ]; then
    python eval_val_loss.py --checkpoint=$CKPT \
      --holdout_dir=$HOLDOUT --out_path=$VALLOSS_JSON
  fi
  # Cleanup intermediate checkpoints
  TDIR=$(dirname "$CKPT")
  find "$TDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
  echo "GPU $GPU_ID: DONE $NAME ($(date))"
done
echo ""; echo "GPU $GPU_ID: ALL JOBS COMPLETE ($(date))"
