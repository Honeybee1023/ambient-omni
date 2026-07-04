#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=create_datasets
#SBATCH --output=/data/scratch/honjar/ambient-omni/pixel-diffusion/create_datasets_%j.out

echo "Job started on $(hostname)"

export PATH=/data/scratch/honjar/miniconda3/envs/ambient/bin:$PATH

cd /data/scratch/honjar/ambient-omni/pixel-diffusion
python create_annotated_datasets.py --num_random_vectors 10
echo "Exit code: $?"