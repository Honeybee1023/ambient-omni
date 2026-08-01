#!/bin/bash
# Eval remaining experiments - designed to survive SSH disconnect
# Run with: setsid bash run_scripts/proline/run_eval_remaining.sh &
export CUDA_VISIBLE_DEVICES=1
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
    echo "SKIP (exists): $NAME"
    return
  fi

  echo "=== Evaluating: $NAME ($(date)) ==="
  rm -rf "$GEN_OUTDIR"
  export MASTER_PORT=$PORT
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT --outdir=$GEN_OUTDIR --seeds=0-4999 --batch=64
  if [ $? -ne 0 ]; then echo "GEN FAILED: $NAME"; return; fi
  sleep 3
  $PY eval_mind.py --gen_path=$GEN_OUTDIR --ref_path=$HOLDOUT --ref_cache=$MIND_CACHE --out_path=$MIND_JSON
  echo "DONE: $NAME -> $(cat $MIND_JSON)"
}

eval_one "step_0to075" "/var/local/honjar/train_outputs/00039-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-WMcsK/network-snapshot-002000.pkl" 10410
eval_one "linear_0to075" "/var/local/honjar/train_outputs/00046-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-kmbCz/network-snapshot-002000.pkl" 10411
eval_one "step_075to0" "/var/local/honjar/train_outputs/00047-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-sd3EX/network-snapshot-002000.pkl" 10412
eval_one "linear_075to0" "/var/local/honjar/train_outputs/00049-celeba_dynamic_t-uncond-ddpmpp-edm-gpus1-batch64-fp32-T253g/network-snapshot-002000.pkl" 10413

# Now train static_T075 fresh (since resume failed and only GPUs 1,3 work)
echo "=== Training static_T075 ($(date)) ==="
export MASTER_PORT=10420
$PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
  --outdir=/var/local/honjar/train_outputs \
  --data=/var/local/honjar/annotated_datasets/celeba_dynamic_t \
  --expr_id=dyn_static_T075 \
  --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
  --corruption_probability=0.0 --noise_config=identity --s_max=4 \
  --cache=False --duration=2 --seed=0 \
  --t_schedule='{"type":"static","t_start":0.75}'
if [ $? -ne 0 ]; then echo "TRAIN FAILED: static_T075"; exit 1; fi

CKPT=$(find /var/local/honjar/train_outputs -name "network-snapshot-002000.pkl" -newer /var/local/honjar/logs_eval_remaining.log -path "*dynamic*" 2>/dev/null | sort | tail -1)
if [ -n "$CKPT" ]; then
  eval_one "static_T075" "$CKPT" 10421
fi

echo "ALL REMAINING COMPLETE ($(date))"
