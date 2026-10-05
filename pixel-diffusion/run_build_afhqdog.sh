#!/bin/bash
# AFHQ-dog mixed-blur data, CPU only, each stage skip-if-done:
#   1. afhqdog_mix4_v1 + clean references (dataset_creation/create_afhqdog_mix4.py)
#   2. afhqdog_cls_mix4, the paired classifier set (create_cls_dataset.py, rounded blur, PNG)
#   3. verification + example grid (dataset_creation/verify_afhqdog_mix4.py)
#   4. MIND reference cache from the 4,739 clean train dogs, plus a sanity MIND of the 500 held-out
#      val dogs against it (eval_mind.py caches the reference features on first use)
# Never on a login node. Engaging:
#   sbatch -p mit_normal -c 16 --mem=32G -t 01:00:00 -o $AMBIENT_BASE/train_logs/afhqdog/build-%j.out \
#          --wrap "bash $AMBIENT_BASE/ambient-omni/pixel-diffusion/run_build_afhqdog.sh"
set -u
export AMBIENT_BASE=${AMBIENT_BASE:?set AMBIENT_BASE}
PY=${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
D=${AMBIENT_BASE}/annotated_datasets
echo "=== afhqdog build | $(hostname) | $(date) ==="

[ -f "$D/afhqdog_mix4_v1/annotations.jsonl" ] || $PY dataset_creation/create_afhqdog_mix4.py || exit 1
[ -f "$D/afhqdog_cls_mix4/cls_labels.jsonl" ] || $PY dataset_creation/create_cls_dataset.py \
    --src afhqdog_mix4_v1 --name afhqdog_cls_mix4 --blur_sigmas 0.3,0.5,1.0,2.0 --ext .png --rint || exit 1
$PY dataset_creation/verify_afhqdog_mix4.py || exit 1

MIND_REF=${AMBIENT_BASE}/generated/mind_ref_cache_afhqdog.npz
VAL_JSON=${AMBIENT_BASE}/generated/mind_afhqdog_val500_vs_train4739.json
if [ ! -f "$VAL_JSON" ]; then
    $PY eval_mind.py --gen_path="${AMBIENT_BASE}/afhqdog_processed/val_64" \
        --ref_path="${AMBIENT_BASE}/afhqdog_processed/train_clean_64" \
        --ref_cache="$MIND_REF" --out_path="$VAL_JSON" || exit 1
fi
echo "=== done | $(date) ==="
