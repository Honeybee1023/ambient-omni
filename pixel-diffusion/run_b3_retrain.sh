#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/honjar/ambient-omni/pixel-diffusion
export TORCH_HOME=/data/honjar/.cache/torch
export WANDB_MODE=disabled
export MASTER_ADDR=localhost
cd /data/honjar/ambient-omni/pixel-diffusion
HOLDOUT=/data/honjar/celeba_processed_v2b/holdout_64
MIND_CACHE=/data/honjar/generated/inception_holdout_feats.npz
for NAME in celeba_v2b_b3_T0475 celeba_v2b_b3_T0525; do
  echo "=== TRAIN $NAME ($(date)) ==="
  export MASTER_PORT=$((RANDOM % 5000 + 20000))
  python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=/data/honjar/train_outputs --data=/data/honjar/annotated_datasets/${NAME} \
    --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=2 --seed=0
  if [ $? -ne 0 ]; then echo "TRAIN FAILED: $NAME"; continue; fi
  CKPT=$(ls /data/honjar/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
  OUTDIR=/data/honjar/generated/${NAME}_5k_gen
  MIND_JSON=/data/honjar/generated/mind_${NAME}_2000kimg.json
  VALLOSS_JSON=/data/honjar/generated/val_loss_${NAME}_2000kimg.json
  rm -rf "$OUTDIR"
  python generate.py --network=$CKPT --outdir=$OUTDIR --seeds=0-4999 --batch=64
  python eval_mind.py --gen_path=$OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  python eval_val_loss.py --checkpoint=$CKPT --holdout_dir=$HOLDOUT --out_path=$VALLOSS_JSON
  find "$(dirname $CKPT)" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
  echo "DONE: $NAME ($(date))"
done
echo "B3 RETRAIN COMPLETE ($(date))"
