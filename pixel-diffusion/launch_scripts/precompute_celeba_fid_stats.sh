#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=celeba_fid_ref
#SBATCH --output=/data/scratch/honjar/train_logs/%j_celeba_fid_ref.out
#SBATCH --requeue

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion
export HF_HOME=/data/scratch/honjar/.cache/huggingface
export TORCH_HOME=/data/scratch/honjar/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

echo "=== Precomputing FID reference stats for CelebA holdout ==="
echo "Holdout dir: /data/scratch/honjar/celeba_processed/holdout_64"
echo "Start time: $(date)"

python -c "
import ambient_utils
import numpy as np

print('Computing inception stats for 20,000 holdout images...')
mu, sigma, inception_score = ambient_utils.eval_utils.calculate_inception_stats(
    '/data/scratch/honjar/celeba_processed/holdout_64', max_batch_size=64)

out_path = '/data/scratch/honjar/celeba_processed/celeba_holdout_ref_stats.npz'
np.savez(out_path, mu=mu, sigma=sigma)
print(f'Saved reference stats to {out_path}')
print(f'Inception score of holdout: {inception_score}')
print(f'mu shape: {mu.shape}, sigma shape: {sigma.shape}')
"

echo "End time: $(date)"
echo "Exit code: $?"
