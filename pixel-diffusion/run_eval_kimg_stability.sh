#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=kimg_stability
#SBATCH --output=/data/scratch/honjar/train_logs/%j_kimg_stability.out

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export HF_HOME=/data/scratch/honjar/.cache/huggingface
export TORCH_HOME=/data/scratch/honjar/.cache/torch

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

GENDIR=/data/scratch/honjar/generated

echo "================================================"
echo "kimg Stability Test: PickScore + Vendi over training"
echo "================================================"

for KIMG in 000500 005004 009007 012010; do
    IMGDIR="${GENDIR}/baseline_wolves_only_${KIMG}kimg"

    if [ ! -d "$IMGDIR" ]; then
        echo "WARNING: $IMGDIR does not exist, skipping"
        continue
    fi

    OUTJSON="${GENDIR}/metrics_wolves_only_${KIMG}kimg.json"

    if [ -f "$OUTJSON" ]; then
        echo "Skipping $KIMG (metrics already computed)"
        continue
    fi

    echo ""
    echo ">>> Evaluating wolves-only at ${KIMG} kimg..."
    python eval_new_metrics.py \
        --image_dir ${IMGDIR} \
        --prompt "a photograph of a wolf" \
        --max_images 1000 \
        --output_json ${OUTJSON}
done

echo ""
echo "================================================"
echo "DONE! Compare JSON files to see metric stability."
echo "================================================"
