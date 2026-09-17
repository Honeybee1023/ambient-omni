#!/bin/bash
# One dynamic-T experiment at an ARBITRARY training budget: train <kimg> -> generate 5k -> MIND + FID.
# Usage: bash run_budget_job.sh <run_name> <gpu_uuid> <total_kimg> [seed] [slot]
#
# Why this exists rather than an edit to run_dyn_job.sh: that script hardcodes --duration=2 and the
# network-snapshot-002000.pkl checkpoint name, and live chains are executing it right now. A script
# a running shell is reading must not be rewritten (bash reads it incrementally), so the budget
# variant is a separate file.
#
# Everything else matches run_dyn_job.sh exactly, so a 1000-kimg run differs from a 2000-kimg run
# in one respect only. NOTE that lr rampup (10,000 kimg) and EMA half-life (500 kimg) are absolute,
# so a shorter run sits at a different point on both -- that applies equally to every arm of the
# budget test, which is why the baselines are rerun at the same length rather than compared across.

set -u

# HARD RULE: never kill a job you did not start. `pkill -f` is banned. See MACHINES.md.

for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break
    [ -d "$_c" ] && AMBIENT_BASE="$_c"
done
AMBIENT_BASE="${AMBIENT_BASE:-/data/scratch/honjar}"
export AMBIENT_BASE

RUN_NAME=${1:-}; GPU_ID=${2:-}; TOTAL_KIMG=${3:-}; TRAIN_SEED=${4:-0}; SLOT=${5:-0}
if [ -z "$RUN_NAME" ] || [ -z "$GPU_ID" ] || [ -z "$TOTAL_KIMG" ]; then
    echo "Usage: bash run_budget_job.sh <run_name> <gpu_uuid> <total_kimg> [seed] [slot]"; exit 2
fi

[ "$GPU_ID" != "slurm" ] && export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((30500 + SLOT * 11 + RANDOM % 7))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f "${AMBIENT_BASE}/.wandb_key" ]; then export WANDB_API_KEY=$(cat "${AMBIENT_BASE}/.wandb_key"); else export WANDB_MODE=offline; fi

PYTHON=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
BASE=${AMBIENT_BASE}
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion || exit 1

MANIFEST="${BASE}/generated/dyn_search_manifest.json"
DATA="${BASE}/annotated_datasets/${DYN_DATASET:-celeba_dynamic_t_v2}"
HOLDOUT="${BASE}/celeba_processed_v2b/holdout_64"
MIND_REF="${BASE}/generated/mind_ref_cache.npz"
NAME="dyn_${RUN_NAME}_s${TRAIN_SEED}"
RUNDIR="${BASE}/train_outputs/dyn_search/${NAME}"
SNAP_NAME=$(printf "network-snapshot-%06d.pkl" "$TOTAL_KIMG")
CKPT="${RUNDIR}/${SNAP_NAME}"
GEN_OUT="${BASE}/generated/${NAME}_5k_gen"
MIND_JSON="${BASE}/generated/mind_${NAME}.json"
FID_JSON="${BASE}/generated/fid_${NAME}.json"
DURATION=$($PYTHON -c "print($TOTAL_KIMG/1000)")

if [ -f "$MIND_JSON" ]; then echo "$NAME already has MIND, nothing to do."; exit 0; fi

SCHEDULE=$($PYTHON -c "
import json,sys
m=json.load(open('$MANIFEST'))
r=[e for e in m['runs'] if e['name']=='$RUN_NAME']
if not r: sys.exit('no such run: $RUN_NAME')
print(json.dumps(r[0]['schedule'],separators=(',',':')))
") || exit 1

echo "=== $NAME | GPU $GPU_ID | seed $TRAIN_SEED | budget ${TOTAL_KIMG} kimg | $(date) ==="
echo "    schedule: $SCHEDULE"
echo "    dataset: $DATA ($(wc -l < "$DATA/annotations.jsonl" 2>/dev/null || echo '?') annotations)"

if [ ! -f "$CKPT" ]; then
    RESUME=""
    if [ -d "$RUNDIR" ]; then
        STATE=$(ls -1 "$RUNDIR"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
        [ -n "$STATE" ] && RESUME="--resume=$STATE" && echo "    resuming from $(basename "$STATE")"
    fi
    [ -z "$RESUME" ] && { rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; }
    echo "--- Training ${TOTAL_KIMG} kimg ---"
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
        --outdir="$RUNDIR" --nosubdir --data="$DATA" --expr_id="$NAME" \
        --cond=0 --arch=ddpmpp --batch=64 --tick=50 --snap=5 --dump=5 \
        --corruption_probability=0.0 --noise_config=identity --s_max=4 \
        --cache=False --duration=$DURATION --seed=$TRAIN_SEED --workers=8 \
        --t_schedule="$SCHEDULE" $RESUME
    [ $? -ne 0 ] && { echo "ERROR: training failed for $NAME"; exit 1; }
fi
[ ! -f "$CKPT" ] && { echo "ERROR: no ${TOTAL_KIMG}-kimg checkpoint for $NAME"; exit 1; }

if [ ! -f "${GEN_OUT}/.complete" ]; then
    rm -rf "$GEN_OUT"; mkdir -p "$GEN_OUT"
    echo "--- Generating 5000 images ---"
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network="$CKPT" --outdir="$GEN_OUT" --seeds=0-4999 --batch=64
    [ $? -ne 0 ] && { echo "ERROR: generation failed for $NAME"; exit 1; }
    touch "${GEN_OUT}/.complete"
fi

if [ ! -f "$MIND_JSON" ]; then
    echo "--- MIND ---"
    $PYTHON eval_mind.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
        --ref_cache="$MIND_REF" --out_path="$MIND_JSON"
    [ $? -ne 0 ] && { echo "ERROR: MIND failed for $NAME"; exit 1; }
fi
if [ ! -f "$FID_JSON" ]; then
    echo "--- FID ---"
    $PYTHON eval_fid.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
        --out_path="$FID_JSON" || echo "WARN: FID failed for $NAME (non-fatal)"
fi

find "$RUNDIR" \( -name "network-snapshot-0*.pkl" ! -name "$SNAP_NAME" -o -name "training-state-*.pt" \) -delete 2>/dev/null
echo "=== DONE $NAME | MIND=$($PYTHON -c "import json;print(json.load(open('$MIND_JSON'))['mind'])") | $(date) ==="
