#!/bin/bash
# Run generation + MIND evaluation for completed dynamic T experiments
# GPU 2: first 4 experiments, GPU 3: next 4
export CUDA_VISIBLE_DEVICES=$1
export MASTER_ADDR=localhost
cd /var/local/honjar/ambient-omni/pixel-diffusion
PY=/var/local/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/var/local/honjar/celeba_processed_v2b/holdout_64
MIND_CACHE=/var/local/honjar/generated/inception_holdout_feats.npz

eval_checkpoint() {
  local NAME=$1
  local CKPT=$2
  local PORT=$3
  
  echo "Evaluating: $NAME"
  GEN_OUTDIR=/var/local/honjar/generated/dynamic_t_${NAME}_gen
  MIND_JSON=/var/local/honjar/generated/mind_dynamic_t_${NAME}.json
  
  if [ ! -d "$GEN_OUTDIR" ] || [ $(ls "$GEN_OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
    export MASTER_PORT=$PORT
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
      --network=$CKPT --outdir=$GEN_OUTDIR --seeds=0-4999 --batch=64
  fi
  if [ ! -f "$MIND_JSON" ]; then
    $PY eval_mind.py --gen_path=$GEN_OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  fi
  echo "DONE: $NAME"
}

shift  # remove GPU arg

# Process all name=ckpt pairs passed as arguments
PORT=10300
while [ $# -ge 2 ]; do
  eval_checkpoint "$1" "$2" "$PORT"
  PORT=$((PORT+1))
  shift 2
done

echo "ALL EVAL COMPLETE ($(date))"
