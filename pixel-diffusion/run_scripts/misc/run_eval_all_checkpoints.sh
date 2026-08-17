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
#SBATCH --job-name=eval_fid
#SBATCH --output=/data/scratch/honjar/train_logs/eval_%j.out

DATASET_NAME=$1

if [ -z "$DATASET_NAME" ]; then
    echo "Error: No dataset name provided."
    exit 1
fi

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

REF_STATS="${AMBIENT_BASE}/annotated_datasets/wolves_ref_stats.npz"
GEN_BASE="${AMBIENT_BASE}/generated"
mkdir -p "$GEN_BASE"

# Collect ALL network-snapshot files across all matching run directories
# Sort by filename (= sort by kimg since zero-padded)
# Skip 000000 (random init)
ALL_CHECKPOINTS=$(ls ${AMBIENT_BASE}/train_outputs/*-${DATASET_NAME}-*/network-snapshot-*.pkl 2>/dev/null | grep -v "network-snapshot-000000.pkl" | while read f; do echo "$(basename $f) $f"; done | sort | awk '{print $2}')

if [ -z "$ALL_CHECKPOINTS" ]; then
    echo "No checkpoints found for $DATASET_NAME"
    exit 1
fi

# For baseline_wolves_only, skip checkpoints we already evaluated (1001, 2001, 3002, 4003)
SKIP_PATTERN=""
if [ "$DATASET_NAME" = "baseline_wolves_only" ]; then
    SKIP_PATTERN="001001|002001|003002|004003"
    echo "Baseline wolves: skipping already-evaluated checkpoints ($SKIP_PATTERN)"
fi

echo "=== Evaluating all checkpoints for: $DATASET_NAME ==="

for CKPT in $ALL_CHECKPOINTS; do
    CKPT_BASE=$(basename $CKPT .pkl)
    KIMG=$(echo $CKPT_BASE | sed 's/network-snapshot-//')

    # Skip if already evaluated (baseline only)
    if [ -n "$SKIP_PATTERN" ] && echo "$KIMG" | grep -qE "$SKIP_PATTERN"; then
        echo "Skipping $KIMG (already evaluated)"
        continue
    fi

    OUTDIR="${GEN_BASE}/${DATASET_NAME}_${KIMG}kimg"

    # Skip if FID already computed for this checkpoint
    if [ -f "${OUTDIR}/fid_out.json" ]; then
        echo "Skipping $KIMG (FID already computed)"
        continue
    fi

    echo ""
    echo "--- Checkpoint: $KIMG kimg ---"
    echo "Network: $CKPT"
    echo "Output: $OUTDIR"

    # Generate 1000 images
    echo "Generating images..."
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network=$CKPT \
        --outdir=$OUTDIR \
        --seeds=0-999 \
        --batch=64

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed for $KIMG"
        continue
    fi

    # Compute FID
    echo "Computing FID..."
    python eval_fid.py \
        --gen_path=$OUTDIR \
        --ref_stats=$REF_STATS \
        --batch_size=64

    if [ $? -ne 0 ]; then
        echo "ERROR: FID computation failed for $KIMG"
        continue
    fi

    echo "Done with $KIMG"
done

echo ""
echo "=== All checkpoints evaluated for $DATASET_NAME ==="
echo "Exit code: $?"
