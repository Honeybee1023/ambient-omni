#!/bin/bash
# Beam-lite controller (lookahead in the loop). Usage:
#   GPU=<uuid> [DYN_DATASET=...] [SEED=0] [SEG=250] [BR=50] [DELTA=0.2] [NGEN=2000] bash run_beam.sh <run_name>
#
# Training proceeds in segments of SEG kimg. At each segment boundary the run
# forks from its saved training state into candidate arms that each train BR
# kimg under a different T for the coming segment (T stays / T ramps up by
# DELTA / T ramps down by DELTA, clipped to [0, 0.95]); each arm generates NGEN
# samples with the same seeds and is scored by MIND against the held-out set;
# the best arm's state becomes the main line and finishes the segment at that
# arm's end T. The choice is therefore made by the objective itself, with a BR
# kimg horizon -- the smallest lookahead controller that is not greedy on a
# proxy. Cost: (arms-1)*BR/SEG extra training + arms*NGEN samples per segment.
set -u
for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break; [ -d "$_c" ] && AMBIENT_BASE="$_c"; done
export AMBIENT_BASE; BASE=$AMBIENT_BASE; PY=$BASE/miniconda3/envs/ambient/bin/python
cd $BASE/ambient-omni/pixel-diffusion || exit 1
[ -f "$BASE/.wandb_key" ] && export WANDB_API_KEY=$(cat $BASE/.wandb_key) || export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<uuid>}
RUN=${1:?run name}; SEED=${SEED:-0}; SEG=${SEG:-250}; BR=${BR:-50}; DELTA=${DELTA:-0.2}; NGEN=${NGEN:-2000}
TOTAL=2000; TEND=0.95
DATA="$BASE/annotated_datasets/${DYN_DATASET:-celeba_dynamic_t_v2}"
HOLDOUT="$BASE/celeba_processed_v2b/holdout_64"; MIND_REF="$BASE/generated/mind_ref_cache.npz"
NAME="dyn_${RUN}_s${SEED}"; ROOT="$BASE/train_outputs/beam/$NAME"; mkdir -p "$ROOT"
LOG="$ROOT/beam_log.jsonl"
FINAL_JSON="$BASE/generated/mind_${NAME}.json"
[ -f "$FINAL_JSON" ] && { echo "$NAME done"; exit 0; }

train() {  # outdir duration_kimg sched [resume] -> trains to duration (absolute kimg)
    local out=$1 dur=$2 sched=$3 resume=${4:-}
    local extra=""; [ -n "$resume" ] && extra="--resume=$resume"
    $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) train.py \
        --outdir="$out" --nosubdir --data="$DATA" --cond=0 --arch=ddpmpp --batch=64 --tick=50 \
        --corruption_probability=0.0 --noise_config=identity --s_max=4 --cache=False --workers=8 \
        --duration=$(python3 -c "print($dur/1000)") --seed=$SEED --snap=1000 --dump=1000 --expr_id="${NAME}_$(basename $out)" \
        --t_schedule="$sched" $extra
}
mind_of() {  # ckpt tag -> prints MIND
    local ck=$1 tag=$2 gen="$ROOT/gen_$tag" js="$ROOT/mind_$tag.json"
    if [ ! -f "$js" ]; then rm -rf "$gen"; mkdir -p "$gen"
        $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) generate.py \
            --network="$ck" --outdir="$gen" --seeds=0-$((NGEN-1)) --batch=64 >/dev/null 2>&1 || return 1
        $PY eval_mind.py --gen_path="$gen" --ref_path="$HOLDOUT" --ref_cache="$MIND_REF" --out_path="$js" >/dev/null 2>&1 || return 1
        rm -rf "$gen"; fi
    python3 -c "import json;print(json.load(open('$js'))['mind'])"
}
pw() { python3 -c "import json;print(json.dumps({'type':'piecewise','control_points':$1},separators=(',',':')))"; }

# state: T at the start of the current segment, and the main-line training state
T=0.0; k=0; MAIN=""
if [ -f "$LOG" ]; then  # resume from the last completed segment
    last=$(tail -1 "$LOG"); T=$(python3 -c "import json;print(json.loads('$last')['T_end'])"); k=$(python3 -c "import json;print(json.loads('$last')['k_end'])")
    MAIN=$(ls $ROOT/main_$(printf %06d $k)/training-state-*.pt 2>/dev/null | tail -1)
fi
while [ $k -lt $TOTAL ]; do
    kb=$((k+BR)); [ $kb -gt $TOTAL ] && kb=$TOTAL
    kend=$((k+SEG)); [ $kend -gt $TOTAL ] && kend=$TOTAL
    p0=$(python3 -c "print($k/$TOTAL)"); pb=$(python3 -c "print($kb/$TOTAL)")
    best=""; bestm=""; declare -A M=()
    for d in 0 $DELTA -$DELTA; do
        Tc=$(python3 -c "print(round(min(max($T+($d),0.0),$TEND),4))")
        [ "$d" != "0" ] && [ "$Tc" = "$T" ] && continue      # clipped into the same arm
        tag="k$(printf %06d $k)_T${Tc}"; out="$ROOT/arm_$tag"
        if ! ls $out/training-state-*.pt >/dev/null 2>&1; then rm -rf "$out"; mkdir -p "$out"
            train "$out" $kb "$(pw "[[0,$T],[$p0,$T],[$pb,$Tc],[1,$Tc]]")" "$MAIN" || { echo "ARM FAIL $tag"; exit 1; }; fi
        ck=$(ls $out/network-snapshot-*.pkl | tail -1); m=$(mind_of "$ck" "$tag") || { echo "MIND FAIL $tag"; exit 1; }
        M[$Tc]=$m; echo "  arm $tag  MIND $m"
        if [ -z "$best" ] || python3 -c "import sys;sys.exit(0 if $m < $bestm else 1)"; then best=$Tc; bestm=$m; bestout=$out; fi
    done
    # continue the main line from the winning arm to the end of the segment
    main_out="$ROOT/main_$(printf %06d $kend)"; st=$(ls $bestout/training-state-*.pt | tail -1)
    if [ $kend -gt $kb ]; then rm -rf "$main_out"; mkdir -p "$main_out"
        train "$main_out" $kend "{\"type\":\"static\",\"t_start\":$best}" "$st" || { echo "MAIN FAIL"; exit 1; }
        MAIN=$(ls $main_out/training-state-*.pt | tail -1)
    else MAIN=$st; mkdir -p "$main_out"; cp $bestout/network-snapshot-*.pkl "$main_out/"; fi
    echo "{\"k_start\":$k,\"k_end\":$kend,\"T_start\":$T,\"T_end\":$best,\"arms\":$(python3 -c "import json;print(json.dumps({$(for key in "${!M[@]}"; do printf '"%s":%s,' "$key" "${M[$key]}"; done | sed 's/,$//')}))")}" >> "$LOG"
    echo "=== segment $k-$kend: T $T -> $best  $(date)"
    # reclaim the losing arms
    for d in $ROOT/arm_k$(printf %06d $k)_*; do [ "$d" != "$bestout" ] && rm -rf "$d"; done
    T=$best; k=$kend
done
CK=$(ls $ROOT/main_$(printf %06d $TOTAL)/network-snapshot-*.pkl | tail -1)
GEN="$BASE/generated/${NAME}_5k_gen"; rm -rf "$GEN"; mkdir -p "$GEN"
$PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) generate.py --network="$CK" --outdir="$GEN" --seeds=0-4999 --batch=64 || exit 1
$PY eval_mind.py --gen_path="$GEN" --ref_path="$HOLDOUT" --ref_cache="$MIND_REF" --out_path="$FINAL_JSON" || exit 1
rm -rf "$GEN"; find "$ROOT" -name "*.pt" -delete; find "$ROOT" -name "network-snapshot-00[01]*.pkl" -delete
echo "=== DONE $NAME MIND=$(python3 -c "import json;print(json.load(open('$FINAL_JSON'))['mind'])") $(date)"
