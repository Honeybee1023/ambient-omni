#!/bin/bash
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=classify_wolves
#SBATCH --output=classify_wolves_%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ambient

cd ~/ambient-omni/pixel-diffusion
python classify_wolves.py