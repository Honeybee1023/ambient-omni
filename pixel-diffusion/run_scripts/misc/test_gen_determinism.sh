#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=det_test
#SBATCH --output=${AMBIENT_BASE}/train_logs/%j_det_test.out
#SBATCH --requeue

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))
cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

# Use one existing T=1 model (should be wolves-only effectively)
CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-pilot2d_dog_T100-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
echo "Checkpoint: $CKPT"

# Run 1: same seeds as original
echo "=== Run 1 (seeds 0-999) ==="
python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT \
    --outdir=${AMBIENT_BASE}/generated/det_test_run1 \
    --seeds=0-999 --batch=64

python eval_new_metrics.py \
    --image_dir ${AMBIENT_BASE}/generated/det_test_run1 \
    --prompt "a photograph of a wolf" --max_images 1000 \
    --output_json ${AMBIENT_BASE}/generated/det_test_run1.json

# Run 2: identical (tests pure determinism)
rm -rf ${AMBIENT_BASE}/generated/det_test_run2
echo "=== Run 2 (seeds 0-999 again) ==="
python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT \
    --outdir=${AMBIENT_BASE}/generated/det_test_run2 \
    --seeds=0-999 --batch=64

python eval_new_metrics.py \
    --image_dir ${AMBIENT_BASE}/generated/det_test_run2 \
    --prompt "a photograph of a wolf" --max_images 1000 \
    --output_json ${AMBIENT_BASE}/generated/det_test_run2.json

# Run 3: different seeds (tests generation variance with different noise)
echo "=== Run 3 (seeds 1000-1999) ==="
python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
    --network=$CKPT \
    --outdir=${AMBIENT_BASE}/generated/det_test_run3 \
    --seeds=1000-1999 --batch=64

python eval_new_metrics.py \
    --image_dir ${AMBIENT_BASE}/generated/det_test_run3 \
    --prompt "a photograph of a wolf" --max_images 1000 \
    --output_json ${AMBIENT_BASE}/generated/det_test_run3.json

echo ""
echo "=== RESULTS ==="
echo "Run 1 vs Run 2: should be IDENTICAL if generation is deterministic"
echo "Run 1 vs Run 3: shows variance from different generation seeds"
for f in det_test_run1.json det_test_run2.json det_test_run3.json; do
    echo "$f:"
    cat ${AMBIENT_BASE}/generated/$f
    echo ""
done
