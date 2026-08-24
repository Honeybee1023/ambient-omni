#!/bin/bash
# One dynamic-T schedule experiment: train 2000 kimg -> generate 5k -> MIND + FID.
# Usage: bash run_dyn_job.sh <run_name> <gpu_uuid> [seed] [slot]
#
# Every schedule shares ONE dataset (celeba_dynamic_t_v2): the b5 bucket carries
# the sigma_min=999 sentinel that --t_schedule overwrites each iteration. So
# there is no per-schedule dataset to build, and the schedule JSON is the only
# thing that differs between runs.
#
# The schedule is looked up by name in the manifest rather than passed on the
# command line: a 5-point control-point list does not survive being handed
# through a queue file, a tmux command and an `eval` without something eating a
# bracket. The name is a single token; the JSON never leaves the file.
#
# Every stage is skip-if-done, so an interrupted run resumes instead of redoing.

set -u

for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break
    [ -d "$_c" ] && AMBIENT_BASE="$_c"
done
AMBIENT_BASE="${AMBIENT_BASE:-/data/scratch/honjar}"
export AMBIENT_BASE

RUN_NAME=${1:-}
# A GPU-<uuid>, not an index: CUDA and nvidia-smi enumerate cards differently
# when only some are reachable, so an index means different things to the
# scheduler's memory check and to this job. Both accept a UUID.
GPU_ID=${2:-}
TRAIN_SEED=${3:-0}
SLOT=${4:-0}

if [ -z "$RUN_NAME" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash run_dyn_job.sh <run_name> <gpu_uuid> [seed] [slot]"; exit 2
fi

# Under Slurm the allocation already pins CUDA_VISIBLE_DEVICES to the GPUs we
# were granted; overwriting it with our own id would point the job at a card
# this job does not own. Pass the literal "slurm" to leave it alone.
if [ "$GPU_ID" != "slurm" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((30500 + SLOT * 11 + RANDOM % 7))
# Cuts fragmentation, which is what turns a tight-but-sufficient card into an
# OOM when a neighbour on the same card is also growing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The key baked into the older run_scripts is committed to a PUBLIC repo and
# still needs rotating; nothing new carries it.
if [ -f "${AMBIENT_BASE}/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "${AMBIENT_BASE}/.wandb_key")
else
    export WANDB_MODE=offline
fi

# Checkpoint cadence. On CSAIL we run as scavengers on a shared QOS and get
# evicted mid-run, so dump every tick (50 kimg, ~9 min of work) instead of every
# 5 (250 kimg, ~46 min). Detected from SLURM_JOB_ID rather than passed in, so
# jobs already sitting in the queue pick it up when they start -- no resubmit.
#
# snap and dump must move TOGETHER: train.py derives the network-snapshot .pkl
# name from the training-state .pt it resumes from (train.py:226), so a state
# dump with no matching snapshot at the same kimg is not resumable.
#
# Cost at snap=dump=1: 223MB pkl + 669MB pt per checkpoint, 40 of each over a
# 2000 kimg run = ~36GB transient per run, reclaimed at the end. CSAIL scratch
# had 4.7TB free.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SNAP_TICKS=${SNAP_TICKS:-1}
    DUMP_TICKS=${DUMP_TICKS:-1}
else
    SNAP_TICKS=${SNAP_TICKS:-5}
    DUMP_TICKS=${DUMP_TICKS:-5}
fi

PYTHON=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
BASE=${AMBIENT_BASE}
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion || exit 1

MANIFEST="${BASE}/generated/dyn_search_manifest.json"
# Which dataset. TWO exist under the name celeba_dynamic_t_v2: lysine carries
# the historical 182,598-image build (b0+b5 plus five parked buckets, what every
# mind_v2_* reference was trained on), while proline and CSAIL carry a
# 26,514-image b0+b5-only variant. This search runs entirely on the 26,514 one,
# so lysine must be pointed at a matching copy rather than its own default.
# Phase 0 measured the two as indistinguishable (<0.00065 apart, noise 0.00099),
# but the batch stays on one file regardless -- mixing them would reintroduce
# exactly the cross-era comparison this project keeps getting burned by.
DATA="${BASE}/annotated_datasets/${DYN_DATASET:-celeba_dynamic_t_v2}"
HOLDOUT="${BASE}/celeba_processed_v2b/holdout_64"
MIND_REF="${BASE}/generated/mind_ref_cache.npz"
NAME="dyn_${RUN_NAME}_s${TRAIN_SEED}"
RUNDIR="${BASE}/train_outputs/dyn_search/${NAME}"
CKPT="${RUNDIR}/network-snapshot-002000.pkl"
GEN_OUT="${BASE}/generated/${NAME}_5k_gen"
MIND_JSON="${BASE}/generated/mind_${NAME}.json"
FID_JSON="${BASE}/generated/fid_${NAME}.json"

if [ -f "$MIND_JSON" ]; then
    echo "$NAME already has MIND, nothing to do."; exit 0
fi

SCHEDULE=$($PYTHON -c "
import json,sys
m=json.load(open('$MANIFEST'))
r=[e for e in m['runs'] if e['name']=='$RUN_NAME']
if not r: sys.exit('no such run: $RUN_NAME')
print(json.dumps(r[0]['schedule'],separators=(',',':')))
") || exit 1

echo "=== $NAME | GPU $GPU_ID (slot $SLOT) | seed $TRAIN_SEED | $(date) ==="
echo "    schedule: $SCHEDULE"
echo "    checkpoint every $((DUMP_TICKS * 50)) kimg (snap=$SNAP_TICKS dump=$DUMP_TICKS)"
echo "    dataset: $DATA ($(wc -l < "$DATA/annotations.jsonl" 2>/dev/null || echo '?') annotations)"
if [ "$GPU_ID" = "slurm" ]; then
    nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader 2>/dev/null
else
    nvidia-smi --id="$GPU_ID" --query-gpu=index,uuid,memory.free --format=csv,noheader 2>/dev/null
fi

# --- Train (resume from the newest state dump if a previous attempt died) ---
if [ ! -f "$CKPT" ]; then
    RESUME=""
    if [ -d "$RUNDIR" ]; then
        STATE=$(ls -1 "$RUNDIR"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
        [ -n "$STATE" ] && RESUME="--resume=$STATE" && echo "    resuming from $(basename "$STATE")"
    fi
    [ -z "$RESUME" ] && { rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; }

    echo "--- Training 2000 kimg ---"
    # --t_schedule is passed as a single argv element, no eval, no word
    # splitting: the JSON contains braces, brackets and commas and must reach
    # click byte-for-byte.
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
        --outdir="$RUNDIR" --nosubdir --data="$DATA" --expr_id="$NAME" \
        --cond=0 --arch=ddpmpp --batch=64 --tick=50 \
        --snap=$SNAP_TICKS --dump=$DUMP_TICKS \
        --corruption_probability=0.0 --noise_config=identity --s_max=4 \
        --cache=False --duration=2 --seed=$TRAIN_SEED --workers=8 \
        --t_schedule="$SCHEDULE" $RESUME
    if [ $? -ne 0 ]; then echo "ERROR: training failed for $NAME"; exit 1; fi
fi

if [ ! -f "$CKPT" ]; then echo "ERROR: no 2k checkpoint for $NAME"; exit 1; fi

# --- Generate 5K ---
if [ ! -f "${GEN_OUT}/.complete" ]; then
    rm -rf "$GEN_OUT"; mkdir -p "$GEN_OUT"
    echo "--- Generating 5000 images ---"
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network="$CKPT" --outdir="$GEN_OUT" --seeds=0-4999 --batch=64
    if [ $? -ne 0 ]; then echo "ERROR: generation failed for $NAME"; exit 1; fi
    touch "${GEN_OUT}/.complete"
fi

# --- Metrics. MIND is the objective; FID is recorded alongside it because the
#     hand-crafted runs have both and dropping it would break that comparison. ---
if [ ! -f "$MIND_JSON" ]; then
    echo "--- MIND ---"
    $PYTHON eval_mind.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
        --ref_cache="$MIND_REF" --out_path="$MIND_JSON"
    if [ $? -ne 0 ]; then echo "ERROR: MIND failed for $NAME"; exit 1; fi
fi
if [ ! -f "$FID_JSON" ]; then
    echo "--- FID ---"
    $PYTHON eval_fid.py --gen_path="$GEN_OUT" --ref_path="$HOLDOUT" \
        --out_path="$FID_JSON" || echo "WARN: FID failed for $NAME (non-fatal)"
fi

# --- Reclaim disk: intermediate snapshots are ~7G per run and never reused. ---
find "$RUNDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" \
    -o -name "training-state-*.pt" \) -delete 2>/dev/null

echo "=== DONE $NAME | MIND=$($PYTHON -c "import json;print(json.load(open('$MIND_JSON'))['mind'])") | $(date) ==="
