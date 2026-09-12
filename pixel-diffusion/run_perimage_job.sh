#!/bin/bash
# Ambient-o verbatim: train on a per-image annotated dataset (classifier
# probabilities -> first-confusion sigma_min per image, computed by the training
# loop itself), no schedule, then generate 5k and MIND. Usage:
#   GPU=<uuid> bash run_perimage_job.sh <annotated_dataset_dir_name> <run_name> [seed]
# --cls_ema_window is 1 because annotate_precorrupted.py used a 64-point sigma
# grid (train.py's default 32 is for annotate.py's 2048-point grid).
set -u
for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break; [ -d "$_c" ] && AMBIENT_BASE="$_c"; done
export AMBIENT_BASE; BASE=$AMBIENT_BASE; PY=$BASE/miniconda3/envs/ambient/bin/python
cd $BASE/ambient-omni/pixel-diffusion || exit 1
[ -f "$BASE/.wandb_key" ] && export WANDB_API_KEY=$(cat $BASE/.wandb_key) || export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<uuid>}
DATA="$BASE/annotated_datasets/$1"; NAME="dyn_$2_s${3:-0}"; SEED=${3:-0}
RUNDIR="$BASE/train_outputs/dyn_search/$NAME"; CK="$RUNDIR/network-snapshot-002000.pkl"
HOLDOUT="$BASE/celeba_processed_v2b/holdout_64"; MIND_REF="$BASE/generated/mind_ref_cache.npz"
GEN="$BASE/generated/${NAME}_5k_gen"; JS="$BASE/generated/mind_${NAME}.json"
[ -f "$JS" ] && { echo "$NAME done"; exit 0; }
if [ ! -f "$CK" ]; then rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) train.py \
    --outdir="$RUNDIR" --nosubdir --data="$DATA" --expr_id="$NAME" --cond=0 --arch=ddpmpp --batch=64 --tick=50 \
    --snap=40 --dump=40 --corruption_probability=0.0 --noise_config=identity --s_max=4 --cache=False \
    --duration=2 --seed=$SEED --workers=8 --cls_ema_window=1 --cls_epsilon=0.05 || { echo "TRAIN FAIL"; exit 1; }
fi
if [ ! -f "$GEN/.complete" ]; then rm -rf "$GEN"; mkdir -p "$GEN"
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) generate.py \
    --network="$CK" --outdir="$GEN" --seeds=0-4999 --batch=64 || exit 1; touch "$GEN/.complete"; fi
$PY eval_mind.py --gen_path="$GEN" --ref_path="$HOLDOUT" --ref_cache="$MIND_REF" --out_path="$JS" || exit 1
find "$RUNDIR" \( -name "network-snapshot-000*.pkl" -o -name "network-snapshot-001*.pkl" -o -name "training-state-*.pt" \) -delete
echo "=== DONE $NAME MIND=$($PY -c "import json;print(json.load(open('$JS'))['mind'])") $(date)"
