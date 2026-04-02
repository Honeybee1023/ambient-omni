#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_fid
#SBATCH --output=/data/scratch/honjar/train_logs/fid_%j.out

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion

cd /data/scratch/honjar/ambient-omni/pixel-diffusion

python eval_fid.py \
  --gen_path "$1" \
  --ref_path "$2" \Si
  --batch_size 64
