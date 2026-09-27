#!/bin/bash
# Ambient-o's clean-vs-corrupted noise classifier for the mixed-blur study, trained with
# the paper's own recipe (pixel-diffusion/README.md, scripts/train_noise_classifier):
#   --precond=edmcls --arch=ddpmpp --cond=0 --lr=1e-4, total batch 512, EDM defaults otherwise
#   (lr warm-up 10000 kimg, ema 0.5, dropout 0.13, augment 0.12), and the length of their
#   released checkpoint (…-iter15k): 15k iterations x 512 = 7,680 kimg.
# Data: celeba_cls_mix4, the paired set (each of the 500 clean faces clean once per level and
# blurred once at 0.3/0.5/1.0/2.0), so labels are balanced and identity carries no label.
#
# Resumes from the newest training-state dump, so a preempted Slurm job can be requeued.
# Usage (CSAIL):  sbatch ... --gres=gpu:h200:4 --wrap "bash run_cls_mix4.sh"
set -u
export AMBIENT_BASE=${AMBIENT_BASE:-/data/scratch/honjar}
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export MASTER_ADDR=localhost MASTER_PORT=$((32100 + RANDOM % 200))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
DATA="${AMBIENT_BASE}/annotated_datasets/celeba_cls_mix4"
OUT="${AMBIENT_BASE}/train_outputs/cls_mix4"
mkdir -p "$OUT"
RESUME=""
STATE=$(ls -1 "$OUT"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$STATE" ] && RESUME="--resume=$STATE" && echo "resuming from $(basename "$STATE")"
echo "=== cls_mix4 | ${NGPU} GPUs | $(date) ==="
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py \
    --outdir="$OUT" --nosubdir --data="$DATA" --expr_id=cls_mix4 \
    --precond=edmcls --overwrite_cls_labels_path="${DATA}/cls_labels.jsonl" \
    --cond=0 --arch=ddpmpp --batch=512 --lr=1e-4 \
    --tick=40 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=7.68 --seed=0 --workers=8 $RESUME
