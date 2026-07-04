#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mind_eval
#SBATCH --output=/data/scratch/honjar/train_logs/%j_mind_eval.out
#SBATCH --requeue

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion
cd /data/scratch/honjar/ambient-omni/pixel-diffusion

for NAME in celeba_cleanonly_500 celeba_cleanonly_750 celeba_cleanonly_1000 celeba_cleanonly_2000 celeba_cleanonly_5000 celeba_cleanonly_10000 celeba_cleanonly_22k; do
    GEN_DIR="/data/scratch/honjar/generated/${NAME}_002000kimg"
    OUT="/data/scratch/honjar/generated/mind_${NAME}.json"
    if [ -f "$OUT" ]; then
        echo "Skipping $NAME (already done)"
        continue
    fi
    echo "=== $NAME ==="
    python eval_mind.py --gen_path=$GEN_DIR --out_path=$OUT
done
echo "=== All done ==="
