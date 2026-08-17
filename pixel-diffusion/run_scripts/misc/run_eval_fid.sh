#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_fid
#SBATCH --output=/data/scratch/honjar/train_logs/fid_%j.out

export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion

cd ${AMBIENT_BASE}/ambient-omni/pixel-diffusion

python eval_fid.py \
  --gen_path "$1" \
  --ref_path "$2" \
  --batch_size 64
