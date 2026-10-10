#!/bin/bash
# Score every snapshot of a training-curve run (run_curve.sh) that has no result yet, on ONE GPU:
#   generate 10k samples (EDM 64-px sampler: deterministic Heun, 40 steps = 79 NFE, seeds 0-9999)
#   -> FID-10k vs the clean training set (EDM/Ambient-o reference), plus CURVE_REF2 if set (CelebA test split)
#   -> MIND vs the same training set (own cache file per reference)
#   -> memorization: Gu et al. nearest-neighbour ratio < 1/3 in pixel l2 vs the clean training originals
# Samples go to node-local scratch and are deleted after scoring (4 PNGs kept), so scratch's file quota
# never sees them. Results: $AMBIENT_BASE/generated/curve/<name>/k<kimg>.json, one file per snapshot.
# When it has scored everything available and the training run is not finished, it resubmits itself to
# start again in 2 hours (CURVE_EVAL_RESUBMIT=0 turns that off).
# Usage: bash run_curve_eval.sh <name>   env CURVE_REF (clean train folder, required), CURVE_REF2 (optional),
#        CURVE_FINAL_KIMG (default 20480)
set -u
export AMBIENT_BASE=${AMBIENT_BASE:?set AMBIENT_BASE}
NAME=${1:?usage: run_curve_eval.sh <name>}
: "${CURVE_REF:?}"
export PATH=${AMBIENT_BASE}/miniconda3/envs/ambient/bin:$PATH
export PYTHONPATH=${AMBIENT_BASE}/ambient-omni/pixel-diffusion
export HF_HOME=${AMBIENT_BASE}/.cache/huggingface TORCH_HOME=${AMBIENT_BASE}/.cache/torch
export MASTER_ADDR=localhost MASTER_PORT=$((33000 + RANDOM % 900))
cd "${AMBIENT_BASE}/ambient-omni/pixel-diffusion" || exit 1
RUNDIR="${AMBIENT_BASE}/train_outputs/curve/${NAME}"
OUT="${AMBIENT_BASE}/generated/curve/${NAME}"
KEEP="${OUT}/samples"
mkdir -p "$OUT" "$KEEP" "${AMBIENT_BASE}/generated/curve/refstats"
TMP=${TMPDIR:-/tmp}/curve_${NAME}_$$
FINAL=${CURVE_FINAL_KIMG:-20480}

refstats() {   # FID reference statistics, computed once per reference folder
    local ref=$1 npz="${AMBIENT_BASE}/generated/curve/refstats/$(basename "$(dirname "$1")")_$(basename "$1").npz"
    if [ ! -f "$npz" ]; then
        python -c "
import numpy as np, ambient_utils
mu, sigma, _ = ambient_utils.eval_utils.calculate_inception_stats('$ref', max_batch_size=250)
np.savez('$npz', mu=mu, sigma=sigma)" >&2 || return 1
    fi
    echo "$npz"
}
mindcache() { echo "${AMBIENT_BASE}/generated/curve/refstats/mind_$(basename "$(dirname "$1")")_$(basename "$1").npz"; }

R1=$(refstats "$CURVE_REF") || { echo "ERROR: reference stats for $CURVE_REF"; exit 1; }
R2=""; [ -n "${CURVE_REF2:-}" ] && { R2=$(refstats "$CURVE_REF2") || exit 1; }

for pkl in $(ls -1 "$RUNDIR"/network-snapshot-*.pkl 2>/dev/null | sort -V); do
    k=$(basename "$pkl" .pkl | sed 's/network-snapshot-0*//'); k=${k:-0}
    [ "$k" -eq 0 ] && continue
    RES="$OUT/k$(printf '%06d' "$k").json"
    [ -f "$RES" ] && continue
    # a snapshot is complete once a newer one exists or the training job has finished writing it (age > 2 min)
    [ $(( $(date +%s) - $(stat -c %Y "$pkl") )) -lt 120 ] && continue
    echo "--- $NAME @ ${k} kimg | $(date +%T) ---"
    rm -rf "$TMP"; mkdir -p "$TMP/gen"
    python -m torch.distributed.run --standalone --nproc_per_node=1 generate.py \
        --network="$pkl" --outdir="$TMP/gen" --seeds=0-9999 --batch=250 --steps=40 >/dev/null || { echo "ERROR: generate $k"; continue; }
    python eval_fid.py --gen_path="$TMP/gen" --ref_stats="$R1" --batch_size=250 --out_path="$TMP/fid1.json" >/dev/null || { echo "ERROR: fid $k"; continue; }
    [ -n "$R2" ] && python eval_fid.py --gen_path="$TMP/gen" --ref_stats="$R2" --batch_size=250 --out_path="$TMP/fid2.json" >/dev/null
    python eval_mind.py --gen_path="$TMP/gen" --ref_path="$CURVE_REF" --ref_cache="$(mindcache "$CURVE_REF")" \
        --out_path="$TMP/mind.json" >/dev/null || { echo "ERROR: mind $k"; continue; }
    python analysis/eval_memo_l2.py --gen_path="$TMP/gen" --train_path="$CURVE_REF" --out_path="$TMP/memo.json" >/dev/null \
        || { echo "ERROR: memo $k"; continue; }
    mkdir -p "$KEEP/k$(printf '%06d' "$k")"
    ls "$TMP/gen" | grep -E '\.png$' | sort | head -4 | while read f; do cp "$TMP/gen/$f" "$KEEP/k$(printf '%06d' "$k")/"; done
    python - "$k" "$TMP" "$RES" <<'P'
import json, os, sys
k, tmp, res = int(sys.argv[1]), sys.argv[2], sys.argv[3]
r = {"kimg": k, "n_samples": 10000, "sampler": "heun 40 steps deterministic"}
r["fid_train"] = json.load(open(f"{tmp}/fid1.json"))["fid_score"]
if os.path.exists(f"{tmp}/fid2.json"):
    r["fid_ref2"] = json.load(open(f"{tmp}/fid2.json"))["fid_score"]
r["mind"] = json.load(open(f"{tmp}/mind.json"))["mind"]
r["memo"] = json.load(open(f"{tmp}/memo.json"))
json.dump(r, open(res, "w"), indent=1)
print(json.dumps({x: r[x] for x in r if x != "memo"}), "memo", r["memo"]["memorized_frac"])
P
done
rm -rf "$TMP"

LAST=$(ls -1 "$RUNDIR"/network-snapshot-*.pkl 2>/dev/null | sort -V | tail -1 | sed 's/.*snapshot-0*\([0-9][0-9]*\)\.pkl/\1/')
DONE_ALL=$(ls "$OUT"/k*.json 2>/dev/null | wc -l)
N_SNAP=$(ls "$RUNDIR"/network-snapshot-*.pkl 2>/dev/null | grep -v 'snapshot-000000' | wc -l)
echo "=== $NAME scored $DONE_ALL of $N_SNAP snapshots; training at ${LAST:-0} of $FINAL kimg | $(date) ==="
if [ "${CURVE_EVAL_RESUBMIT:-1}" = 1 ] && { [ "${LAST:-0}" -lt "$FINAL" ] || [ "$DONE_ALL" -lt "$N_SNAP" ]; } && [ -n "${SLURM_JOB_ID:-}" ]; then
    # same Blackwell exclusion as bo_mix4.sh: torch 2.6+cu124 has no sm_120 kernels
    sbatch --begin=now+2hours --exclude='node[4004,4007-4008,5003-5005,5101,5103-5104,5106,5202-5204]'  $(scontrol show job "$SLURM_JOB_ID" -o | grep -o 'Partition=[^ ]*' | sed 's/Partition=/-p /') \
        --gres=gpu:1 -c 8 --mem=96G -t 06:00:00 -J "ceval_${NAME}" \
        -o "${AMBIENT_BASE}/train_logs/curve/eval_${NAME}-%j.out" \
        --export=ALL --wrap "bash ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_curve_eval.sh ${NAME}"
fi
