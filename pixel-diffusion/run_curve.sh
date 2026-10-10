#!/bin/bash
# Training-curve run (deployment-length phase, 2026-10-10): ONE long constant-LR run whose snapshots are
# each a valid run of that length, used to find where training stops improving.
# Usage (inside a 4-GPU Slurm allocation): bash run_curve.sh <name>
#   env CURVE_DATA     dataset folder under annotated_datasets (required)
#       CURVE_SCHED    per_group schedule JSON with phases at 0 only (static), or "none"
#       CURVE_DROPOUT  0.05 for CelebA (EDM FFHQ-64), 0.25 for AFHQ (EDM AFHQv2-64)   (required)
#       CURVE_MIMG     total length in Mimg (default 20.48 = 40 snapshots of 512 kimg). Raising it later
#                      and resubmitting continues exactly where the run stopped.
#       CURVE_EXTRA    extra train.py flags (Ambient-o arm: --cls_ema_window=1, as every *_ambo run)
#
# Settings are EDM's 64x64 recipe (FFHQ/AFHQv2-64, Karras et al. 2022, Table 7 + README), every one passed
# explicitly because train.py's defaults are the old CIFAR/1e-3/10-Mimg-ramp values:
#   DDPM++ with cres 1,2,2,2 (61.8M params), batch 256 over 4 GPUs (64/GPU, no accumulation), lr 2e-4,
#   EMA half-life 0.5 Mimg with ramp ratio 0.05, augment 0.15, fp32 with TF32 off.
#   Deviations (disclosed): 100-kimg linear warm-up instead of EDM's 10 Mimg (EDM's would put half of
#   the curve inside the ramp; same as the *_wu runs), gradient clipping 1.0 (Ambient-o's code).
# Nothing in these settings depends on the planned total: warm-up and EMA use images seen so far, the
# schedule is static, the LR is constant. So the snapshot at k kimg IS a k-kimg run.
#
# Snapshot grid: tick = 64 kimg = exactly 250 steps of 256, snapshot and state dump every 8 ticks = 512 kimg.
# KEEP_SNAPSHOTS=1 keeps every network snapshot (~250 MB each); only state dumps are pruned (last 2 kept).
# Never deletes anything else. Resumes from the newest state dump, so chaining jobs past the 1-day limit
# (or extending CURVE_MIMG) needs no other step.
set -u
export AMBIENT_BASE=${AMBIENT_BASE:?set AMBIENT_BASE}
NAME=${1:?usage: run_curve.sh <name>}
: "${CURVE_DATA:?}" "${CURVE_DROPOUT:?}"
MIMG=${CURVE_MIMG:-20.48}
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost MASTER_PORT=$((31000 + RANDOM % 900))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export KEEP_LAST_DUMPS=2 KEEP_SNAPSHOTS=1
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1

DATA="${AMBIENT_BASE}/annotated_datasets/${CURVE_DATA}"
RUNDIR="${AMBIENT_BASE}/train_outputs/curve/${NAME}"
FINAL_KIMG=$(awk "BEGIN{printf \"%d\", ${MIMG}*1000}")
if ls "$RUNDIR"/network-snapshot-*.pkl >/dev/null 2>&1; then
    LAST=$(ls -1 "$RUNDIR"/network-snapshot-*.pkl | sort -V | tail -1 | sed 's/.*snapshot-0*\([0-9][0-9]*\)\.pkl/\1/')
    if [ "${LAST:-0}" -ge "$FINAL_KIMG" ]; then echo "$NAME already at ${LAST} kimg >= ${FINAL_KIMG}; nothing to do."; exit 0; fi
fi
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
[ "$NGPU" = 4 ] || { echo "ERROR: curve runs need exactly 4 GPUs (batch 256 = 4 x 64), got $NGPU"; exit 1; }
[ -f "$DATA/annotations.jsonl" ] || { echo "ERROR: no $DATA/annotations.jsonl"; exit 1; }
SCHED=${CURVE_SCHED:-none}
if [ "$SCHED" = none ]; then SCHED_ARG=(); else SCHED_ARG=(--t_schedule="$SCHED"); fi

mkdir -p "$RUNDIR"
RESUME=""
STATE=$(ls -1 "$RUNDIR"/training-state-*.pt 2>/dev/null | sort -V | tail -1)
[ -n "$STATE" ] && RESUME="--resume=$STATE"
echo "=== curve $NAME | $(hostname) | $(date) ==="
echo "    data $DATA ($(wc -l < "$DATA/annotations.jsonl") annotations) | dropout $CURVE_DROPOUT | to ${MIMG} Mimg"
echo "    schedule: $SCHED | extra: ${CURVE_EXTRA:-} | ${RESUME:-fresh start}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    --outdir="$RUNDIR" --nosubdir --data="$DATA" --expr_id="curve_${NAME}" \
    --cond=0 --arch=ddpmpp --precond=edm --cres=1,2,2,2 \
    --batch=256 --lr=2e-4 --lr_rampup_kimg=100 --ema=0.5 \
    --dropout="$CURVE_DROPOUT" --augment=0.15 --clip=1.0 --fp16=0 \
    --tick=64 --snap=8 --dump=8 \
    --corruption_probability=0.0 --noise_config=identity --s_max=4 \
    --cache=False --workers=8 --duration="$MIMG" --seed=0 \
    "${SCHED_ARG[@]}" $RESUME ${CURVE_EXTRA:-}
rc=$?
echo "=== curve $NAME exit $rc | $(date) ==="
exit $rc
