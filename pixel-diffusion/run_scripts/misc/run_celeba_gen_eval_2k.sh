#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cb_eval2k
#SBATCH --output=/data/scratch/honjar/train_logs/%j_cb_eval2k.out
#SBATCH --requeue

DATASET_NAME=$1
if [ -z "$DATASET_NAME" ]; then
    echo "Error: No dataset name provided."
    exit 1
fi

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion
export HF_HOME=/data/scratch/honjar/.cache/huggingface
export TORCH_HOME=/data/scratch/honjar/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

GEN_BASE=/data/scratch/honjar/generated
REF_STATS=/data/scratch/honjar/celeba_processed/celeba_holdout_ref_stats.npz

# Find 2000 kimg checkpoint (may be 002000 or 002001 etc due to resume drift)
CKPT=$(ls /data/scratch/honjar/train_outputs/*-${DATASET_NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No 2k kimg checkpoint found for $DATASET_NAME"
    exit 1
fi

KIMG_LABEL=$(basename $CKPT | sed 's/network-snapshot-0*\([0-9]*\)\.pkl/\1/')
OUTDIR="${GEN_BASE}/${DATASET_NAME}_002000kimg"
OUTJSON="${GEN_BASE}/metrics_${DATASET_NAME}_2000kimg.json"

echo "=== Gen+Eval 2k: $DATASET_NAME ==="
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
    if [ $? -ne 0 ]; then echo "ERROR: Generation failed"; exit 1; fi
fi

if [ -f "$OUTJSON" ]; then
    echo "Metrics already computed, skipping."
else
    if [ ! -f "$REF_STATS" ]; then echo "ERROR: Reference stats not found"; exit 1; fi
    echo "Computing FID..."
    python eval_fid.py \
        --gen_path=$OUTDIR \
        --ref_stats=$REF_STATS \
        --out_path=$OUTJSON
fi

echo "=== Done: $DATASET_NAME ==="
