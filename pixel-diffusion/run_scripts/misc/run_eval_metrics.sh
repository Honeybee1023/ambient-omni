#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_metrics
#SBATCH --output=/data/scratch/honjar/train_logs/%j_eval_metrics.out

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export HF_HOME=/data/scratch/honjar/.cache/huggingface
export TORCH_HOME=/data/scratch/honjar/.cache/torch

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

GENDIR=/data/scratch/honjar/generated

echo "================================================"
echo "Metric Evaluation: wolves-only vs naive_all"
echo "================================================"

echo ""
echo ">>> Evaluating wolves-only (1501 kimg)..."
python eval_new_metrics.py \
    --image_dir ${GENDIR}/baseline_wolves_only_001501kimg \
    --prompt "a photograph of a wolf" \
    --max_images 1000 \
    --output_json ${GENDIR}/metrics_wolves_only_1501kimg.json

echo ""
echo ">>> Evaluating naive_all (1001 kimg)..."
python eval_new_metrics.py \
    --image_dir ${GENDIR}/baseline_naive_all_001001kimg \
    --prompt "a photograph of a wolf" \
    --max_images 1000 \
    --output_json ${GENDIR}/metrics_naive_all_1001kimg.json

echo ""
echo "================================================"
echo "DONE! Check the JSON files for side-by-side results."
echo "================================================"
