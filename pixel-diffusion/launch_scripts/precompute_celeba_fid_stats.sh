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
#SBATCH --job-name=celeba_fid_ref
#SBATCH --output=${AMBIENT_BASE}/train_logs/%j_celeba_fid_ref.out
#SBATCH --requeue

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface
export TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 1000 + 10000))

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

echo "=== Precomputing FID reference stats for CelebA holdout ==="
echo "Holdout dir: ${AMBIENT_BASE}/celeba_processed/holdout_64"
echo "Start time: $(date)"

python -c "
import ambient_utils
import numpy as np

print('Computing inception stats for 20,000 holdout images...')
mu, sigma, inception_score = ambient_utils.eval_utils.calculate_inception_stats(
    '${AMBIENT_BASE}/celeba_processed/holdout_64', max_batch_size=64)

out_path = '${AMBIENT_BASE}/celeba_processed/celeba_holdout_ref_stats.npz'
np.savez(out_path, mu=mu, sigma=sigma)
print(f'Saved reference stats to {out_path}')
print(f'Inception score of holdout: {inception_score}')
print(f'mu shape: {mu.shape}, sigma shape: {sigma.shape}')
"

echo "End time: $(date)"
echo "Exit code: $?"
