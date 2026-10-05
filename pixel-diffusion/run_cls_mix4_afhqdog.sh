#!/bin/bash
# Ambient-o's clean-vs-corrupted noise classifier for the AFHQ-dog mixed-blur set: a parameterised
# copy of run_cls_mix4.sh with the IDENTICAL recipe (paper's own, scripts/train_noise_classifier):
#   --precond=edmcls --arch=ddpmpp --cond=0 --lr=1e-4, total batch 512, EDM defaults otherwise
#   (lr warm-up 10000 kimg, ema 0.5, dropout 0.13, augment 0.12), 15k iterations x 512 = 7,680 kimg.
# Only the dataset/output names change (env CLS_DATA / CLS_OUT / CLS_ID). Data: afhqdog_cls_mix4, the
# paired set (each of the 474 clean dogs clean once per level and blurred once at 0.3/0.5/1.0/2.0,
# rounded, PNG), so labels are balanced and identity carries no label.
#
# The total batch stays 512 whatever the GPU count (train.py splits it per rank; 256/GPU on 2 GPUs
# needs ~41 GB, so ask for H200/H100, not L40S). Resumes from the newest training-state dump, so the
# 6-h public-partition limit is handled by chaining a second job with --dependency=afterany; a job that
# finds the final snapshot already written exits at once.
#
# KEEP_LAST_DUMPS is forced to 0 (keep everything), as in the CelebA run: with --snap=--dump the
# pruning in training_loop.py would also delete every intermediate network-snapshot, and those are
# what tell us when the classifier starts to see mild blur. ~12 GB in total.
#
# Engaging (public partition only; never the Sloan partitions for this):
#   J1=$(sbatch --parsable -p mit_normal_gpu --gres=gpu:h200:2 -c 20 --mem=128G -t 06:00:00 \
#        -J cls_afhqdog -o $AMBIENT_BASE/train_logs/afhqdog/cls_mix4-%j.out --wrap "bash run_cls_mix4_afhqdog.sh")
#   sbatch -p mit_normal_gpu ... --dependency=afterany:$J1 --wrap "bash run_cls_mix4_afhqdog.sh"
set -u
export AMBIENT_BASE=${AMBIENT_BASE:?set AMBIENT_BASE}
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export MASTER_ADDR=localhost MASTER_PORT=$((32100 + RANDOM % 200))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export KEEP_LAST_DUMPS=0
# CLS_EXTRA: extra train.py flags, e.g. "--batch-gpu=128" to run the same total batch 512 on one GPU
# by gradient accumulation (identical recipe, fewer GPUs).
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
CLS_ID=${CLS_ID:-cls_mix4_afhqdog}
DATA="${AMBIENT_BASE}/annotated_datasets/${CLS_DATA:-afhqdog_cls_mix4}"
OUT="${AMBIENT_BASE}/train_outputs/${CLS_OUT:-cls_mix4_afhqdog}"
FINAL="$OUT/network-snapshot-007680.pkl"
if [ -f "$FINAL" ]; then echo "$FINAL exists; classifier finished, nothing to do."; exit 0; fi
[ -f "${DATA}/cls_labels.jsonl" ] || { echo "ERROR: no ${DATA}/cls_labels.jsonl"; exit 1; }
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
mkdir -p "$OUT"
RESUME=""
STATE=$(ls -1 "$OUT"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$STATE" ] && RESUME="--resume=$STATE" && echo "resuming from $(basename "$STATE")"
echo "=== ${CLS_ID} | ${NGPU} GPUs | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py \
    --outdir="$OUT" --nosubdir --data="$DATA" --expr_id="$CLS_ID" \
    --precond=edmcls --overwrite_cls_labels_path="${DATA}/cls_labels.jsonl" \
    --cond=0 --arch=ddpmpp --batch=512 --lr=1e-4 \
    --tick=40 --snap=5 --dump=5 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --duration=7.68 --seed=0 --workers=8 $RESUME ${CLS_EXTRA:-}
