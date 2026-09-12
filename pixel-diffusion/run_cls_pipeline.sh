#!/bin/bash
# Does the threshold Ambient-o's own classifier assigns to our blur bucket match
# the MIND-optimal static threshold (0.50)?  Three stages, each skip-if-done:
#
#   1. build the balanced classifier training set     (seconds)
#   2. train the time-conditional classifier            (~1 h, 200 kimg)
#   3. annotate every blurred image, summarise in T     (a few h)
#
# The downstream training runs (per-image thresholds verbatim; static at the
# median) are launched separately once the median is known.
#
# Usage: bash run_cls_pipeline.sh <gpu_uuid>
set -u

# HARD RULE: never kill a job you did not start yourself. See MACHINES.md.

for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break
    [ -d "$_c" ] && AMBIENT_BASE="$_c"
done
export AMBIENT_BASE="${AMBIENT_BASE:-/data/scratch/honjar}"

GPU_ID=${1:-}
[ -z "$GPU_ID" ] && { echo "Usage: bash run_cls_pipeline.sh <gpu_uuid>"; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export MASTER_ADDR=localhost MASTER_PORT=$((31900 + RANDOM % 50))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "${AMBIENT_BASE}/.wandb_key" ] && export WANDB_API_KEY=$(cat "${AMBIENT_BASE}/.wandb_key") || export WANDB_MODE=offline
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
PY=python

# lysine's b0+b5 build is celeba_dynamic_t_v2_b0b5; proline's is celeba_dynamic_t_v2.
SRC="${CLS_SRC:-$( [ -d "${AMBIENT_BASE}/annotated_datasets/celeba_dynamic_t_v2_b0b5" ] && echo celeba_dynamic_t_v2_b0b5 || echo celeba_dynamic_t_v2 )}"
CLS_DATA="${AMBIENT_BASE}/annotated_datasets/${CLS_DATASET:-celeba_cls_paired}"
CLS_OUT="${CLS_OUT:-${AMBIENT_BASE}/train_outputs/cls_paired_v4}"
CLS_KIMG=${CLS_KIMG:-1500}
CKPT="${CLS_OUT}/network-snapshot-$(printf '%06d' "$CLS_KIMG").pkl"
ANN_OUT="${ANN_OUT:-${AMBIENT_BASE}/annotated_datasets/celeba_amb_perimage_v4}"

echo "=== classifier pipeline | GPU $GPU_ID | src $SRC | $(date) ==="

# --- 1. dataset ------------------------------------------------------------
if [ ! -f "${CLS_DATA}/cls_labels.jsonl" ]; then
    echo "--- building classifier dataset from $SRC"
    $PY dataset_creation/create_cls_dataset.py --src "$SRC" --name "${CLS_DATASET:-celeba_cls_paired}" --blur_sigma "${CLS_BLUR:-0.5}" || exit 1
fi

# --- 2. train --------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
    echo "--- training classifier for ${CLS_KIMG} kimg"
    mkdir -p "$CLS_OUT"
    # --precond=edmcls: EDMPrecondCLS + AmbientEDMCLSLoss (Ambient-o's classifier).
    # All sigma_min are 0 so both classes are drawn at every noise level; the
    # loss noises x0 itself. --snap/--dump in ticks of 10 kimg.
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
        --outdir="$CLS_OUT" --nosubdir --data="$CLS_DATA" --expr_id=cls_paired \
        --precond=edmcls --overwrite_cls_labels_path="${CLS_DATA}/cls_labels.jsonl" \
        --cond=0 --arch=ddpmpp --batch=64 --tick=10 --snap=5 --dump=5 \
        --corruption_probability=0.0 --noise_config=identity --s_max=4 \
        --lr=3e-4 --lr_rampup_kimg=20 ${CLS_EXTRA:-} \
        --cache=False --duration=$(awk "BEGIN{print $CLS_KIMG/1000}") --seed=0 --workers=8
    [ -f "$CKPT" ] || { echo "ERROR: no classifier checkpoint at $CKPT"; ls "$CLS_OUT"; exit 1; }
fi

# Refuse to annotate with a classifier that never learned. Chance for binary
# cross-entropy is ln 2 = 0.693. For a blur as mild as sigma=0.5 the cue is
# only visible at LOW noise, so the honest check is the lowest-noise quartile
# (Loss/bucket_0): that bucket must be clearly below chance. The higher
# buckets are expected to stay near chance -- that is what makes the
# annotation land at a low threshold, and is the method working, not failing.
# Attempt 1 (200 kimg, LR <= 2e-5): bucket_0 0.70 -> 0.65, rest unmoved,
# thresholds noise. Attempt 2 (400 kimg, LR 1e-3): nothing moved by 50 kimg.
B0_LOSS=$($PY -c "
import json
L=[json.loads(l) for l in open('${CLS_OUT}/stats.jsonl')]
print(sum(r['Loss/bucket_0']['mean'] for r in L[-5:])/5)")
echo "    classifier low-noise-bucket loss (mean of last 5 ticks): $B0_LOSS   (chance = 0.693)"
if $PY -c "import sys; sys.exit(0 if float('$B0_LOSS') < 0.55 else 1)"; then :; else
    echo "ERROR: low-noise bucket loss $B0_LOSS is too close to chance; not annotating."; exit 1
fi

# --- 3. annotate + summarise ----------------------------------------------
if [ ! -f "${ANN_OUT}/threshold_summary.json" ]; then
    echo "--- annotating blurred images with the classifier"
    $PY analysis/annotate_precorrupted.py \
        --checkpoint_path "$CKPT" \
        --dataset_path "${AMBIENT_BASE}/annotated_datasets/${SRC}" \
        --out "$ANN_OUT" --num_sigmas ${NUM_SIGMAS:-64} --num_trials_per_t 4 || exit 1
fi

echo "=== DONE classifier pipeline | $(date) ==="
cat "${ANN_OUT}/threshold_summary.json"
