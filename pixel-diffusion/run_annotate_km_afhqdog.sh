#!/bin/bash
# After the AFHQ-dog classifier (run_cls_mix4_afhqdog.sh) has finished: the same two steps that made
# celeba_mix4_ambo and celeba_mix4_km, each skip-if-done.
#   1. Ambient-o per-image annotation of every blurred dog with the final classifier snapshot,
#      exactly CSAIL's train_logs/mix4/annotate_mix4.sh: 64-sigma grid, 4 trials per sigma, eps 0.05,
#      EMA window 1 (auto-scaled from 32 * 64/2048)                       -> afhqdog_mix4_ambo
#      Train on it like mix4_ambo: TRAIN_EXTRA=--cls_ema_window=1.
#   2. 1-D k-means (k=4) of those thresholds, k1 = mildest              -> afhqdog_mix4_km
# Prints the classifier's low-noise-bucket loss first (chance 0.693); the CelebA run did not gate on it,
# so neither does this -- read it before using the annotations.
# Engaging (public GPU partition, ~1 GPU-hour or less for 4,265 images):
#   sbatch -p mit_normal_gpu --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 -J annot_afhqdog \
#          -o $AMBIENT_BASE/train_logs/afhqdog/annotate-%j.out \
#          --wrap "bash $AMBIENT_BASE/ambient-omni/pixel-diffusion/run_annotate_km_afhqdog.sh"
set -u
export AMBIENT_BASE=${AMBIENT_BASE:?set AMBIENT_BASE}
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
D=${AMBIENT_BASE}/annotated_datasets
# CLS_CKPT: the classifier snapshot (default: the original 7.68-Mimg one). ANN_SRC / ANN_OUT / KM_OUT: dataset names.
CKPT=${CLS_CKPT:-${AMBIENT_BASE}/train_outputs/${CLS_OUT:-cls_mix4_afhqdog}/network-snapshot-007680.pkl}
SRC=${ANN_SRC:-afhqdog_mix4_v1}; AMB=${ANN_OUT:-afhqdog_mix4_ambo}; KM=${KM_OUT:-afhqdog_mix4_km}
[ -f "$CKPT" ] || { echo "ERROR: classifier not finished, no $CKPT"; exit 1; }
echo "=== annotate + k-means | $(hostname) | $(date) ==="
python -c "
import json
L=[json.loads(l) for l in open('$(dirname "$CKPT")/stats.jsonl')]
for b in range(4):
    k=f'Loss/bucket_{b}'
    if k in L[-1]: print(f'  classifier {k} (mean of last 5 ticks): {sum(r[k][\"mean\"] for r in L[-5:])/5:.4f}   (chance 0.693)')
"

if [ ! -f "$D/$AMB/threshold_summary.json" ]; then
    python analysis/annotate_precorrupted.py \
        --checkpoint_path "$CKPT" \
        --dataset_path "$D/$SRC" \
        --out "$D/$AMB" \
        --num_sigmas 64 --num_trials_per_t 4 \
        --clean_prefix b0_ --corrupt_prefix g03_,g05_,g10_,g20_ || exit 1
fi
cat "$D/$AMB/threshold_summary.json"

if [ ! -f "$D/$KM/km_assignment.json" ]; then
    python dataset_creation/create_km_dataset.py --src $SRC --ambo $AMB \
        --name $KM || exit 1
fi
echo "=== done | $(date) ==="
