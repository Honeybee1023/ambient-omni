#!/bin/bash
# Round 3: is warmup winning because of its SHAPE, or because it feeds the model
# more corrupt data?
#
# Round 2 left these tangled. Warmup holds T at 0 early, so its T curve sits below
# linear's for the whole run, so more corrupt images are eligible throughout:
# warmup25 averaged 60.5% corrupt batches vs 48.0% for linear. Warmup won. But
# "better shape" and "more corrupt data" predict that equally well.
#
# The arms below are ordered so that the first four vary warmup length and
# endpoint (what we want to know), and the last two are the controls that can
# actually tell the two explanations apart:
#
#   linear_0to0713 and cosine_0to0713 have the SAME mean T as warmup25 (0.3563),
#   so they feed the model the same amount of corrupt data with a different shape.
#
#     - if they match warmup25  -> exposure explains it, shape is a red herring,
#                                  and the real knob is just "how much corrupt data"
#     - if warmup25 still wins  -> the shape genuinely matters, and holding T=0
#                                  early is doing something a smooth ramp cannot
#
# Either outcome is publishable; not knowing which is not.
#
#   tmux new-session -d -s r3g0 'GPU=0 ./run_scripts/lysine/run_warmup_sweep_v3.sh'
#   ... one per GPU, they coordinate through the same mkdir locks as v2.

set -u

GPU=${GPU:-0}
SEED=${SEED:-0}
BASE_PORT=${BASE_PORT:-11400}
ONLY=${ONLY:-}

export CUDA_VISIBLE_DEVICES=$GPU
export MASTER_ADDR=localhost
export WANDB_MODE=disabled

cd /data/honjar/ambient-omni/pixel-diffusion

PY=/data/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/data/honjar/celeba_processed_v2b/holdout_64
DATA=/data/honjar/annotated_datasets/celeba_dynamic_t_v2
OUTDIR=/data/honjar/train_outputs/dynamic_t_v2
GENDIR=/data/honjar/generated
REF_CACHE=${GENDIR}/inception_holdout_feats.npz
LOGDIR=/data/honjar/train_logs/dynamic_t_v2
LOCKDIR=/data/honjar/train_logs/dynamic_t_v2/locks

mkdir -p "$OUTDIR" "$LOGDIR" "$LOCKDIR"
PORT_CTR=$((BASE_PORT + GPU * 100 + SEED * 10))

MY_LOCKS=()
cleanup() { for l in "${MY_LOCKS[@]:-}"; do [ -n "$l" ] && rmdir "$l" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

run_one() {
  local TAG=$1 SCHEDULE=$2
  local NAME="v2_${TAG}_s${SEED}"

  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $TAG "* ]]; then return 0; fi

  local RUNDIR="${OUTDIR}/${NAME}"
  local CKPT="${RUNDIR}/network-snapshot-002000.pkl"
  local GEN_OUT="${GENDIR}/${NAME}_5k_gen"
  local MIND_JSON="${GENDIR}/mind_${NAME}.json"
  local FID_JSON="${GENDIR}/fid_${NAME}.json"
  local LOG="${LOGDIR}/${NAME}.log"
  local LOCK="${LOCKDIR}/${NAME}"

  if [ -f "$MIND_JSON" ]; then echo "SKIP (already done): $NAME"; return 0; fi
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "SKIP (claimed by GPU $(cat "$LOCK/gpu" 2>/dev/null || echo '?')): $NAME"; return 0
  fi
  echo "$GPU" > "$LOCK/gpu"; echo "$$" > "$LOCK/pid"; MY_LOCKS+=("$LOCK")
  PORT_CTR=$((PORT_CTR + 1))

  ( _body "$TAG" "$SCHEDULE" "$NAME" "$RUNDIR" "$CKPT" "$GEN_OUT" "$MIND_JSON" "$FID_JSON" "$LOG" )
  local RC=$?
  rm -f "$LOCK/gpu" "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null
  return $RC
}

_body() {
  local TAG=$1 SCHEDULE=$2 NAME=$3 RUNDIR=$4 CKPT=$5 GEN_OUT=$6 MIND_JSON=$7 FID_JSON=$8 LOG=$9

  echo ""
  echo "=================================================================="
  echo "  $NAME   (GPU $GPU, seed $SEED)   started $(date '+%F %T')"
  echo "  schedule: $SCHEDULE"
  echo "  log: $LOG"
  echo "=================================================================="

  if [ ! -f "$CKPT" ]; then
    local RESUME=""
    local STATE
    STATE=$(ls -1t "${RUNDIR}"/training-state-*.pt 2>/dev/null | head -1)
    if [ -n "$STATE" ]; then
      echo "resuming from $STATE"
      RESUME="--resume=$STATE"
    fi
    $PY -u train.py --outdir="$RUNDIR" --nosubdir --data="$DATA" \
        --arch=ddpmpp --batch=64 --duration=2 --seed=$SEED \
        --lr=10e-4 --dropout=0.13 --augment=0.12 --ema=0.5 \
        --t_schedule="$SCHEDULE" \
        --port=$PORT_CTR $RESUME >> "$LOG" 2>&1 || { echo "TRAIN FAILED: $NAME"; return 1; }
  fi

  if [ ! -d "$GEN_OUT" ] || [ "$(ls -1 "$GEN_OUT" 2>/dev/null | wc -l)" -lt 5000 ]; then
    $PY -u generate.py --network="$CKPT" --outdir="$GEN_OUT" \
        --seeds=0-4999 --batch=64 >> "$LOG" 2>&1 || { echo "GEN FAILED: $NAME"; return 1; }
  fi

  $PY -u eval_mind.py --gen_dir="$GEN_OUT" --ref_dir="$HOLDOUT" \
      --out_path="$MIND_JSON" >> "$LOG" 2>&1 || { echo "MIND FAILED: $NAME"; return 1; }
  $PY -u eval_fid.py --gen_dir="$GEN_OUT" --ref_cache="$REF_CACHE" \
      --out_path="$FID_JSON" >> "$LOG" 2>&1 || echo "FID failed (non-fatal): $NAME"

  echo "DONE $NAME  $(date '+%F %T')  MIND=$($PY -c "import json;print(json.load(open('$MIND_JSON'))['mind'])" 2>/dev/null)"
}

echo "###################################################################"
echo "# Warmup sweep v3 | GPU $GPU | seed $SEED | $(date)"
echo "###################################################################"

# --- 1. Does warmup length matter? (warmup25 = 0.02962 is the incumbent) -----
run_one warmup15_0to095 '{"type":"warmup_linear","t_start":0.0,"t_end":0.95,"warmup_frac":0.15}'
run_one warmup40_0to095 '{"type":"warmup_linear","t_start":0.0,"t_end":0.95,"warmup_frac":0.40}'

# --- 2. Does the final endpoint matter? (shallower slope, lower ceiling) -----
run_one linear_0to085   '{"type":"linear","t_start":0.0,"t_end":0.85}'
run_one warmup25_0to085 '{"type":"warmup_linear","t_start":0.0,"t_end":0.85,"warmup_frac":0.25}'

# --- 3. THE CONTROLS: same corrupt-data exposure as warmup25, different shape -
#        mean T = 0.3563 for all three, solved numerically.
run_one linear_0to0713  '{"type":"linear","t_start":0.0,"t_end":0.7126}'
run_one cosine_0to0713  '{"type":"cosine","t_start":0.0,"t_end":0.7126}'

echo ""
echo "queue drained on GPU $GPU at $(date)"
