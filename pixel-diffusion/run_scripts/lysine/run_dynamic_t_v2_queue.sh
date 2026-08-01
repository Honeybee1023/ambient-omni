#!/bin/bash
# Dynamic-T v2 experiment queue, lysine, single GPU (default: GPU 0).
#
# Runs train -> generate 5k -> FID + MIND for each experiment, sequentially.
# Safe to re-run: any experiment whose MIND json already exists is skipped, so
# you can kill this and restart, or point it at another GPU to run the tail of
# the queue in parallel.
#
# ALWAYS LAUNCH THIS DETACHED FROM THE TERMINAL, e.g.
#
#     tmux new-session -d -s dynt 'GPU=0 ./run_scripts/lysine/run_dynamic_t_v2_queue.sh'
#     tmux attach -t dynt        # to watch
#
# `nohup ... &` is NOT sufficient: torch.distributed.run installs its own
# SIGHUP handler which overrides the disposition nohup inherits, so an SSH
# disconnect kills training mid-run (this cost us a 3.5h run on 2026-07-31).
# tmux/setsid put the job in a new session with no controlling terminal.
#
#   GPU=1 ...        # choose GPU
#   SEED=1 ...       # additional seed
#
# Each experiment is ~9.2h of training on an A100 (16.5 sec/kimg x 2000 kimg)
# plus ~20 min for generation and eval.

set -u

GPU=${GPU:-0}
SEED=${SEED:-0}
BASE_PORT=${BASE_PORT:-10900}

export CUDA_VISIBLE_DEVICES=$GPU
export MASTER_ADDR=localhost
export WANDB_MODE=disabled

cd /data/honjar/ambient-omni/pixel-diffusion

PY=/data/honjar/miniconda3/envs/ambient/bin/python
HOLDOUT=/data/honjar/celeba_processed_v2b/holdout_64
DATA=/data/honjar/annotated_datasets/celeba_dynamic_t_v2
DATA_CLEAN=/data/honjar/annotated_datasets/celeba_dynamic_t_v2_cleanonly
OUTDIR=/data/honjar/train_outputs/dynamic_t_v2
GENDIR=/data/honjar/generated
REF_CACHE=${GENDIR}/inception_holdout_feats.npz
LOGDIR=/data/honjar/train_logs/dynamic_t_v2

mkdir -p "$OUTDIR" "$LOGDIR"

PORT_CTR=$((BASE_PORT + GPU * 100 + SEED * 10))

run_one() {
  local TAG=$1 DATASET=$2 SCHEDULE=$3
  local NAME="v2_${TAG}_s${SEED}"
  local RUNDIR="${OUTDIR}/${NAME}"
  local CKPT="${RUNDIR}/network-snapshot-002000.pkl"
  local GEN_OUT="${GENDIR}/${NAME}_5k_gen"
  local MIND_JSON="${GENDIR}/mind_${NAME}.json"
  local FID_JSON="${GENDIR}/fid_${NAME}.json"
  local LOG="${LOGDIR}/${NAME}.log"

  if [ -f "$MIND_JSON" ]; then
    echo "SKIP (already done): $NAME"
    return 0
  fi

  PORT_CTR=$((PORT_CTR + 1))
  echo ""
  echo "=================================================================="
  echo "  $NAME   (GPU $GPU, seed $SEED)   started $(date '+%F %T')"
  echo "  schedule: ${SCHEDULE:-<none, clean-only dataset>}"
  echo "  log: $LOG"
  echo "=================================================================="

  # ---- train ----
  if [ ! -f "$CKPT" ]; then
    # Resume from the newest training-state dump if a previous attempt died
    # partway (each run is ~9h, so losing one to a crash is expensive).
    local RESUME=""
    if [ -d "$RUNDIR" ]; then
      local STATE
      STATE=$(ls -1 "$RUNDIR"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
      if [ -n "$STATE" ]; then
        RESUME="--resume=$STATE"
        echo "  resuming from $(basename "$STATE")"
      fi
    fi
    if [ -z "$RESUME" ]; then
      rm -rf "$RUNDIR"
      mkdir -p "$RUNDIR"
    fi
    export MASTER_PORT=$PORT_CTR
    local CMD="$PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
      --outdir=$RUNDIR --nosubdir --data=$DATASET --expr_id=$NAME \
      --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
      --corruption_probability=0.0 --noise_config=identity --s_max=4 \
      --cache=False --duration=2 --seed=$SEED --workers=8 $RESUME"
    if [ -n "$SCHEDULE" ]; then
      CMD="$CMD --t_schedule='$SCHEDULE'"
    fi
    if ! eval $CMD >> "$LOG" 2>&1; then
      echo "TRAIN FAILED: $NAME (see $LOG)"
      return 1
    fi
  else
    echo "  (checkpoint already present, skipping training)"
  fi

  if [ ! -f "$CKPT" ]; then
    echo "NO CHECKPOINT produced: $NAME"
    return 1
  fi

  # ---- generate ----
  if [ ! -f "${GEN_OUT}/.complete" ]; then
    rm -rf "$GEN_OUT"
    export MASTER_PORT=$((PORT_CTR + 500))
    if ! $PY -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network="$CKPT" --outdir="$GEN_OUT" --seeds=0-4999 --batch=64 >> "$LOG" 2>&1; then
      echo "GENERATE FAILED: $NAME"
      return 1
    fi
    touch "${GEN_OUT}/.complete"
  fi

  # ---- eval ----
  $PY eval_mind.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
      --ref_cache="$REF_CACHE" --out_path="$MIND_JSON" >> "$LOG" 2>&1 \
      || echo "MIND FAILED: $NAME"
  $PY eval_fid.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
      --out_path="$FID_JSON" >> "$LOG" 2>&1 \
      || echo "FID FAILED: $NAME"

  echo "  done $(date '+%F %T'): $(cat "$MIND_JSON" 2>/dev/null)"
}

echo "###################################################################"
echo "# Dynamic-T v2 queue | GPU $GPU | seed $SEED | started $(date)"
echo "###################################################################"

# --------------------------------------------------------------------------
# 0. PIPELINE VALIDATION.
#    Same dataset + a static T=0.475 schedule must reproduce the pre-existing
#    static run celeba_v2b_b5_T0475 (MIND = 0.035401). If this lands far from
#    that number, something in the dynamic path is still wrong and everything
#    below is untrustworthy -- check this one before reading any other result.
# --------------------------------------------------------------------------
run_one validate_T0475 "$DATA" '{"type":"static","t_start":0.475}'

# --------------------------------------------------------------------------
# 1. BASELINES: the static curve on this exact setup, incl. the high-T region
#    where we currently have no b5 measurements.
# --------------------------------------------------------------------------
run_one clean_only   "$DATA_CLEAN" ''
run_one static_T000  "$DATA" '{"type":"static","t_start":0.0}'
run_one static_T050  "$DATA" '{"type":"static","t_start":0.5}'
run_one static_T075  "$DATA" '{"type":"static","t_start":0.75}'
run_one static_T095  "$DATA" '{"type":"static","t_start":0.95}'

# --------------------------------------------------------------------------
# 2. THE HYPOTHESIS: ramp up (use everything early, exclude late).
# --------------------------------------------------------------------------
run_one linear_0to095  "$DATA" '{"type":"linear","t_start":0.0,"t_end":0.95}'
run_one step_0to095    "$DATA" '{"type":"step","t_start":0.0,"t_end":0.95,"switch_point":0.5}'
run_one cosine_0to095  "$DATA" '{"type":"cosine","t_start":0.0,"t_end":0.95}'
run_one warmup_0to095  "$DATA" '{"type":"warmup_linear","t_start":0.0,"t_end":0.95,"warmup_frac":0.25}'

# --------------------------------------------------------------------------
# 3. CONTROLS: anti-curriculum. If these beat the ramps, the story is wrong.
# --------------------------------------------------------------------------
run_one linear_095to0  "$DATA" '{"type":"linear","t_start":0.95,"t_end":0.0}'
run_one step_095to0    "$DATA" '{"type":"step","t_start":0.95,"t_end":0.0,"switch_point":0.5}'

# --------------------------------------------------------------------------
# 4. MILDER ENDPOINTS: in case T=0.95 overshoots for the unconditional setup,
#    where the static optimum is nearer T=0.5.
# --------------------------------------------------------------------------
run_one linear_0to050  "$DATA" '{"type":"linear","t_start":0.0,"t_end":0.5}'
run_one cosine_0to050  "$DATA" '{"type":"cosine","t_start":0.0,"t_end":0.5}'
run_one step_0to050    "$DATA" '{"type":"step","t_start":0.0,"t_end":0.5,"switch_point":0.5}'
run_one linear_050to0  "$DATA" '{"type":"linear","t_start":0.5,"t_end":0.0}'
run_one twophase_0_050_095 "$DATA" '{"type":"two_phase","t_start":0.0,"t_mid":0.5,"t_end":0.95}'

echo ""
echo "###################################################################"
echo "# QUEUE FINISHED $(date)"
echo "###################################################################"
