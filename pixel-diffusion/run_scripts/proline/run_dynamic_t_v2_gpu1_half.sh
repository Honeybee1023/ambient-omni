#!/bin/bash
# GPU 1: second half of experiments (9-16)
# GPU 3 is running experiments sequentially from 1. These are the tail end it won't reach for days.
set -e
export CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=localhost
cd /var/local/honjar/ambient-omni/pixel-diffusion
PY=/var/local/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/var/local/honjar/celeba_processed_v2b/holdout_64
DATA=/var/local/honjar/annotated_datasets/celeba_dynamic_t_v2
OUTDIR=/var/local/honjar/train_outputs
GENDIR=/var/local/honjar/generated
SEED=0
PORT_CTR=10800

run_one() {
  local NAME="v2_${1}_s${SEED}"
  local DATASET=$2
  local SCHEDULE=$3
  PORT_CTR=$((PORT_CTR + 1))
  local PORT=$PORT_CTR

  FID_JSON=${GENDIR}/fid_${NAME}.json
  if [ -f "$FID_JSON" ]; then
    echo "SKIP (done): $NAME"
    return
  fi

  echo ""; echo "=== $NAME ($(date)) ==="; echo ""
  export MASTER_PORT=$PORT
  CMD="$PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=$OUTDIR --data=$DATASET --expr_id=$NAME \
    --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=2 --seed=$SEED --workers=8"
  if [ -n "$SCHEDULE" ]; then
    CMD="$CMD --t_schedule='$SCHEDULE'"
  fi
  eval $CMD
  if [ $? -ne 0 ]; then echo "TRAIN FAILED: $NAME"; return; fi
  sleep 5

  CKPT=$(find $OUTDIR -name "network-snapshot-002000.pkl" -newer /var/local/honjar/annotated_datasets/celeba_dynamic_t_v2/annotations.jsonl 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "NO CHECKPOINT: $NAME"; return; fi

  GEN_OUT=${GENDIR}/v2_${NAME}_gen
  rm -rf "$GEN_OUT"
  export MASTER_PORT=$((PORT + 30))
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT --outdir=$GEN_OUT --seeds=0-4999 --batch=64
  if [ $? -ne 0 ]; then echo "GEN FAILED: $NAME"; return; fi
  sleep 3

  $PY eval_fid.py --gen_path=$GEN_OUT --ref_path=$HOLDOUT --out_path=fid_result.json
  mv ${GEN_OUT}/fid_result.json $FID_JSON 2>/dev/null || true
  MIND_JSON=${GENDIR}/mind_${NAME}.json
  $PY eval_mind.py --gen_path=$GEN_OUT --ref_path=$HOLDOUT \
    --ref_cache=${GENDIR}/inception_holdout_feats.npz --out_path=$MIND_JSON

  TDIR=$(dirname "$CKPT")
  find "$TDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
  echo "DONE: $NAME ($(date))"
}

echo "GPU 1 second-half experiments starting ($(date))"

# Experiments 9-16 (GPU 3 does 1-8)
run_one "step_095to0" "$DATA" '{"type":"step","t_start":0.95,"t_end":0.0,"switch_point":0.5}'
run_one "cosine_0to095" "$DATA" '{"type":"cosine","t_start":0.0,"t_end":0.95}'
run_one "warmup_0to095" "$DATA" '{"type":"warmup_linear","t_start":0.0,"t_end":0.95,"warmup_frac":0.25}'
run_one "linear_0to050" "$DATA" '{"type":"linear","t_start":0.0,"t_end":0.5}'
run_one "linear_050to0" "$DATA" '{"type":"linear","t_start":0.5,"t_end":0.0}'
run_one "step_0to050" "$DATA" '{"type":"step","t_start":0.0,"t_end":0.5,"switch_point":0.5}'
run_one "cosine_0to050" "$DATA" '{"type":"cosine","t_start":0.0,"t_end":0.5}'
run_one "twophase_0_050_095" "$DATA" '{"type":"two_phase","t_start":0.0,"t_mid":0.5,"t_end":0.95}'

echo "GPU 1: ALL COMPLETE ($(date))"
