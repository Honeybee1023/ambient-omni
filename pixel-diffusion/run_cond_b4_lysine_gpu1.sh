#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

export CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=localhost
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion
PY=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
HOLDOUT=${AMBIENT_BASE}/celeba_processed_v2b/holdout_64
MIND_CACHE=${AMBIENT_BASE}/generated/inception_holdout_feats.npz
NAME=celeba_v2b_cond_b4_T080

echo ""; echo "==============================="; echo "GPU 1: $NAME ($(date))"; echo "==============================="
export MASTER_PORT=10201
$PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
  --outdir=${AMBIENT_BASE}/train_outputs \
  --data=${AMBIENT_BASE}/annotated_datasets/${NAME} \
  --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
  --corruption_probability=0.0 --noise_config=identity --s_max=4 \
  --cache=False --duration=2 --seed=0
if [ $? -ne 0 ]; then echo "TRAIN FAILED: $NAME"; exit 1; fi
CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then echo "NO CHECKPOINT: $NAME"; exit 1; fi
OUTDIR=${AMBIENT_BASE}/generated/${NAME}_5k_gen
MIND_JSON=${AMBIENT_BASE}/generated/mind_${NAME}_2000kimg.json
VALLOSS_JSON=${AMBIENT_BASE}/generated/val_loss_${NAME}_2000kimg.json
if [ ! -d "$OUTDIR" ] || [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
  export MASTER_PORT=10211
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT --outdir=$OUTDIR --seeds=0-4999 --batch=64
fi
if [ ! -f "$MIND_JSON" ]; then
  $PY eval_mind.py --gen_path=$OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
fi
if [ ! -f "$VALLOSS_JSON" ]; then
  $PY eval_val_loss.py --checkpoint=$CKPT --holdout_dir=$HOLDOUT --out_path=$VALLOSS_JSON
fi
TDIR=$(dirname "$CKPT")
find "$TDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
echo "GPU 1: DONE $NAME ($(date))"
