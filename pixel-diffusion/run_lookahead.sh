#!/bin/bash
# Lookahead-oracle experiment: how far ahead must you look to pick the right T?
#
#   bash run_lookahead.sh spine            # train the spine (best schedule, seed 2), keeping
#                                          #   full training state every 250 kimg
#   bash run_lookahead.sh branches         # from each saved state, continue K kimg under a
#                                          #   fixed T, generate 5k, MIND. Skip-if-done.
#
# The spine is sched_hold50_ceil95 (T=0 for the first half, ramp to 0.95), the
# 2-knot schedule that matches the 30-run search winner. A branch is a faithful
# continuation: --resume restores weights, optimizer state and the kimg counter,
# so lr ramp and EMA half-life carry on exactly; only the schedule changes.
#
# Question answered per branch point k and horizon K: which T ∈ {0, 0.5, 0.95}
# gives the best MIND K kimg later? If short K prefers high T early while long
# K prefers T=0 (and all K prefer 0.95 late), a myopic controller is provably
# stuck and the horizon tells us what a lookahead controller must pay.
#
# Env: GPU (uuid, required), DYN_DATASET, KS, TS, HS to restrict the grid.
set -u
for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break; [ -d "$_c" ] && AMBIENT_BASE="$_c"; done
export AMBIENT_BASE; BASE=$AMBIENT_BASE
PY=$BASE/miniconda3/envs/ambient/bin/python
cd $BASE/ambient-omni/pixel-diffusion || exit 1
[ -f "$BASE/.wandb_key" ] && export WANDB_API_KEY=$(cat $BASE/.wandb_key) || export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<uuid>}
DATA="$BASE/annotated_datasets/${DYN_DATASET:-celeba_dynamic_t_v2}"
HOLDOUT="$BASE/celeba_processed_v2b/holdout_64"; MIND_REF="$BASE/generated/mind_ref_cache.npz"
SPINE_SEED=${SPINE_SEED:-2}
SPINE="$BASE/train_outputs/lookahead/spine_s${SPINE_SEED}"
SCHED='{"type":"piecewise","control_points":[[0.0,0.0],[0.5,0.0],[1.0,0.95]]}'
LOG="$BASE/lookahead_state"; mkdir -p "$LOG" "$BASE/train_outputs/lookahead"

train() {  # outdir duration_mimg schedule extra...
    local out=$1 dur=$2 sched=$3; shift 3
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) train.py \
        --outdir="$out" --nosubdir --data="$DATA" --cond=0 --arch=ddpmpp --batch=64 --tick=50 \
        --corruption_probability=0.0 --noise_config=identity --s_max=4 --cache=False --workers=8 \
        --duration=$dur --t_schedule="$sched" "$@"
}
mind() {  # ckpt name -> writes generated/mind_<name>.json
    local ck=$1 name=$2 gen="$BASE/generated/${name}_5k_gen" js="$BASE/generated/mind_${name}.json"
    [ -f "$js" ] && return 0
    if [ ! -f "$gen/.complete" ]; then rm -rf "$gen"; mkdir -p "$gen"
        $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) generate.py \
            --network="$ck" --outdir="$gen" --seeds=0-4999 --batch=64 || return 1
        touch "$gen/.complete"; fi
    $PY eval_mind.py --gen_path="$gen" --ref_path="$HOLDOUT" --ref_cache="$MIND_REF" --out_path="$js" || return 1
    rm -rf "$gen"
}

case ${1:-} in
spine)
    if [ ! -f "$SPINE/network-snapshot-002000.pkl" ]; then
        rm -rf "$SPINE"; mkdir -p "$SPINE"
        train "$SPINE" 2 "$SCHED" --expr_id=lookahead_spine_s$SPINE_SEED --seed=$SPINE_SEED --snap=5 --dump=5 || exit 1
    fi
    mind "$SPINE/network-snapshot-002000.pkl" "dyn_sched_hold50_ceil95_s$SPINE_SEED"
    echo "SPINE_DONE $(date)";;
branches)
    for k in ${KS:-250 500 750 1000 1250 1500 1750}; do
      st=$(ls $SPINE/training-state-$(printf %06d $k).pt 2>/dev/null || ls $SPINE/training-state-$(printf %06d $((k+1))).pt 2>/dev/null)
      [ -z "$st" ] && { echo "no state near $k"; continue; }
      kk=$(basename $st .pt | sed 's/training-state-//' | sed 's/^0*//')
      for T in ${TS:-0 0.5 0.95}; do for K in ${HS:-100 300}; do
        name="la_k${kk}_T${T}_K${K}_s${SPINE_SEED}"; js="$BASE/generated/mind_${name}.json"
        [ -f "$js" ] && continue
        br="$BASE/train_outputs/lookahead/$name"; dur=$(python3 -c "print(($kk+$K)/1000)")
        if ! ls $br/network-snapshot-*.pkl >/dev/null 2>&1 || [ "$(ls $br/network-snapshot-*.pkl | wc -l)" -lt 1 ]; then
            rm -rf "$br"; mkdir -p "$br"
            echo "=== $name  resume $st  duration $dur  $(date)"
            train "$br" $dur "{\"type\":\"static\",\"t_start\":$T}" --expr_id=$name --resume="$st" --snap=1000 --dump=1000 || { echo "TRAIN FAIL $name"; continue; }
        fi
        ck=$(ls $br/network-snapshot-*.pkl | tail -1)
        mind "$ck" "$name" && echo "DONE $name MIND=$($PY -c "import json;print(json.load(open('$js'))['mind'])") $(date)" && rm -f $br/*.pkl $br/*.pt
      done; done
    done
    echo "BRANCHES_DONE $(date)";;
*) echo "usage: $0 spine|branches"; exit 1;;
esac
