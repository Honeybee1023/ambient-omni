#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=dino_fd
#SBATCH --output=/data/scratch/honjar/train_logs/%j_dino_fd.out
#SBATCH --requeue

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export TORCH_HOME=/data/scratch/honjar/.cache/torch
cd /data/scratch/honjar/ambient-omni/pixel-diffusion

for NAME in celeba_cleanonly_500 celeba_cleanonly_750 celeba_cleanonly_1000 celeba_cleanonly_2000 celeba_cleanonly_5000 celeba_cleanonly_10000 celeba_cleanonly_22k; do
    GEN_DIR="/data/scratch/honjar/generated/${NAME}_002000kimg"
    OUT="/data/scratch/honjar/generated/dino_fd_${NAME}.json"
    if [ -f "$OUT" ]; then
        echo "Skipping $NAME (already done)"
        continue
    fi
    echo "=== $NAME ==="
    python eval_dino_fd.py --gen_path=$GEN_DIR --out_path=$OUT
done
echo "=== All done ==="
