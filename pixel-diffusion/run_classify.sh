#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=classify_wolves
#SBATCH --output=/data/scratch/honjar/ambient-omni/pixel-diffusion/classify_wolves_%j.out

echo "Job started on $(hostname)"

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=/data/scratch/honjar/ambient-omni/pixel-diffusion

cd /data/scratch/honjar/ambient-omni/pixel-diffusion
python classify_wolves.py
echo "Exit code: $?"