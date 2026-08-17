#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cb_stab_eval
#SBATCH --output=/data/scratch/honjar/train_logs/%j_cb_stab_eval.out
#SBATCH --requeue

CKPT=$1
KIMG_LABEL=$2

if [ -z "$CKPT" ] || [ -z "$KIMG_LABEL" ]; then
    echo "Usage: sbatch script.sh <checkpoint_path> <kimg_label>"
    echo "Example: sbatch script.sh /path/to/network-snapshot-000500.pkl 500"
    exit 1
fi

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

GEN_BASE=${AMBIENT_BASE}/generated
REF_STATS=${AMBIENT_BASE}/celeba_processed/celeba_holdout_ref_stats.npz

OUTDIR="${GEN_BASE}/celeba_2d_b3_T050_stability_${KIMG_LABEL}kimg"
OUTJSON="${GEN_BASE}/metrics_celeba_2d_b3_T050_${KIMG_LABEL}kimg.json"

echo "=== Stability Eval: ${KIMG_LABEL} kimg ==="
echo "Checkpoint: $CKPT"

if [ -d "$OUTDIR" ] && [ $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l) -ge 1000 ]; then
    echo "Images already generated, skipping."
else
    echo "Generating 1000 images..."
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT \
        --outdir=$OUTDIR \
        --seeds=0-999 \
        --batch=64
    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed"
        exit 1
    fi
fi

if [ -f "$OUTJSON" ]; then
    echo "Metrics already computed, skipping."
else
    if [ ! -f "$REF_STATS" ]; then
        echo "ERROR: Reference stats not found at $REF_STATS"
        exit 1
    fi
    echo "Computing FID against holdout..."
    python eval_fid.py \
        --gen_path=$OUTDIR \
        --ref_stats=$REF_STATS \
        --out_path=$OUTJSON
fi

echo "=== Done: ${KIMG_LABEL} kimg ==="
echo "Results: $OUTJSON"
