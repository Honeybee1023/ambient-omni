#!/bin/bash
# Dynamic T Scheduling Experiments - GPU 3
# Experiments: cosine_0_to_075, warmup_linear
export CUDA_VISIBLE_DEVICES=3
export MASTER_ADDR=localhost
cd /var/local/honjar/ambient-omni/pixel-diffusion
PY=/var/local/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/var/local/honjar/celeba_processed_v2b/holdout_64
MIND_CACHE=/var/local/honjar/generated/inception_holdout_feats.npz
DATA=/var/local/honjar/annotated_datasets/celeba_dynamic_t
OUTDIR=/var/local/honjar/train_outputs

run_experiment() {
  local NAME=$1
  local DATASET=$2
  local SCHEDULE=$3
  local PORT=$4
  
  echo ""; echo "==============================="; echo "GPU 3: $NAME ($(date))"; echo "==============================="
  export MASTER_PORT=$PORT
  
  CMD="$PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --outdir=$OUTDIR \
    --data=$DATASET \
    --expr_id=$NAME \
    --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=2 --seed=0"
  
  if [ -n "$SCHEDULE" ]; then
    CMD="$CMD --t_schedule='$SCHEDULE'"
  fi
  
  eval $CMD
  if [ $? -ne 0 ]; then echo "TRAIN FAILED: $NAME"; return; fi
  
  CKPT=$(find $OUTDIR -name "network-snapshot-002*.pkl" -path "*${NAME}*" 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then echo "NO CHECKPOINT: $NAME"; return; fi
  
  GEN_OUTDIR=/var/local/honjar/generated/dynamic_t_${NAME}_gen
  MIND_JSON=/var/local/honjar/generated/mind_dynamic_t_${NAME}.json
  
  if [ ! -d "$GEN_OUTDIR" ] || [ $(ls "$GEN_OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
    export MASTER_PORT=$((PORT+10))
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
      --network=$CKPT --outdir=$GEN_OUTDIR --seeds=0-4999 --batch=64
  fi
  if [ ! -f "$MIND_JSON" ]; then
    $PY eval_mind.py --gen_path=$GEN_OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  fi
  
  TDIR=$(dirname "$CKPT")
  find "$TDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
  echo "GPU 3: DONE $NAME ($(date))"
}

# Experiment 9: Cosine 0 → 0.75
run_experiment "dyn_cosine_0to075" "$DATA" '{"type":"cosine","t_start":0.0,"t_end":0.75}' 10230

# Experiment 10: Warmup then linear 0 → 0.75
run_experiment "dyn_warmup_linear" "$DATA" '{"type":"warmup_linear","t_start":0.0,"t_end":0.75,"warmup_frac":0.25}' 10231

echo ""; echo "GPU 3: ALL JOBS COMPLETE ($(date))"
