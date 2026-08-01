#!/bin/bash
# Fixed eval script - runs generation + MIND for all experiments that need it
# Usage: bash run_eval_all.sh <gpu_id>
set -e
export CUDA_VISIBLE_DEVICES=$1
export MASTER_ADDR=localhost
cd /var/local/honjar/ambient-omni/pixel-diffusion
PY=/var/local/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/var/local/honjar/celeba_processed_v2b/holdout_64
MIND_CACHE=/var/local/honjar/generated/inception_holdout_feats.npz

eval_one() {
  local NAME=$1
  local CKPT=$2
  local PORT=$3

  GEN_OUTDIR=/var/local/honjar/generated/dynamic_t_${NAME}_gen
  MIND_JSON=/var/local/honjar/generated/mind_dynamic_t_${NAME}.json

  if [ -f "$MIND_JSON" ]; then
    echo "SKIP (MIND exists): $NAME"
    return
  fi

  echo "=== Evaluating: $NAME ($(date)) ==="

  # Generate if needed
  if [ ! -d "$GEN_OUTDIR" ] || [ $(ls "$GEN_OUTDIR"/*.png 2>/dev/null | wc -l) -lt 5000 ]; then
    rm -rf "$GEN_OUTDIR"
    export MASTER_PORT=$PORT
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
      --network=$CKPT --outdir=$GEN_OUTDIR --seeds=0-4999 --batch=64
    sleep 5
  fi

  # MIND eval
  $PY eval_mind.py --gen_path=$GEN_OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  echo "DONE: $NAME -> $(cat $MIND_JSON)"
}

# All experiments with their checkpoints
eval_one "cosine_0to075" "/var/local/honjar/train_outputs/00037-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-328yv/network-snapshot-002000.pkl" 10400
eval_one "static_T050" "/var/local/honjar/train_outputs/00038-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-n4oqK/network-snapshot-002000.pkl" 10401
eval_one "step_0to075" "/var/local/honjar/train_outputs/00039-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-WMcsK/network-snapshot-002000.pkl" 10402
eval_one "static_T000" "/var/local/honjar/train_outputs/00044-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-q9sgc/network-snapshot-002000.pkl" 10403
eval_one "warmup_linear" "/var/local/honjar/train_outputs/00045-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-DaVYt/network-snapshot-002000.pkl" 10404
eval_one "linear_0to075" "/var/local/honjar/train_outputs/00046-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-kmbCz/network-snapshot-002000.pkl" 10405
eval_one "step_075to0" "/var/local/honjar/train_outputs/00047-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-sd3EX/network-snapshot-002000.pkl" 10406
eval_one "linear_075to0" "/var/local/honjar/train_outputs/00049-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-T253g/network-snapshot-002000.pkl" 10407

echo "ALL EVAL COMPLETE ($(date))"
