#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Submit confound tests + context sweeps + BO round 4

echo "=== Disk check ==="
du -sh ${AMBIENT_BASE}/train_outputs/

echo ""
echo "=== Confound Tests (3 models) ==="
for ds in celeba_confound_b2only celeba_confound_b1only celeba_confound_b1b2; do
    TRAIN=$(sbatch --exclude=aia-h200-7 --parsable ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k.sh "$ds")
    EVAL=$(sbatch --exclude=aia-h200-7 --parsable --dependency=afterok:${TRAIN} ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_celeba_gen_eval_2k.sh "$ds")
    echo "  $ds: train=$TRAIN eval=$EVAL"
done

echo ""
echo "=== B2 Context Sweep (10 models) ==="
for t in 080 085 090 092 094 095 096 097 098 099; do
    ds="celeba_ctx_b2_T${t}"
    TRAIN=$(sbatch --exclude=aia-h200-7 --parsable ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k.sh "$ds")
    EVAL=$(sbatch --exclude=aia-h200-7 --parsable --dependency=afterok:${TRAIN} ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_celeba_gen_eval_2k.sh "$ds")
    echo "  $ds: train=$TRAIN eval=$EVAL"
done

echo ""
echo "=== B3 Context Sweep (10 models) ==="
for t in 085 088 090 092 094 095 096 097 098 099; do
    ds="celeba_ctx_b3_T${t}"
    TRAIN=$(sbatch --exclude=aia-h200-7 --parsable ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k.sh "$ds")
    EVAL=$(sbatch --exclude=aia-h200-7 --parsable --dependency=afterok:${TRAIN} ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_celeba_gen_eval_2k.sh "$ds")
    echo "  $ds: train=$TRAIN eval=$EVAL"
done

echo ""
echo "=== BO Round 4 (15 models) ==="
MANIFEST="${AMBIENT_BASE}/generated/bo_round4_manifest.json"
DATASETS=$(${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python -c \
    "import json; d=json.load(open('$MANIFEST')); print(' '.join(d['datasets']))")
for ds in $DATASETS; do
    TRAIN=$(sbatch --exclude=aia-h200-7 --parsable ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_2k.sh "$ds")
    EVAL=$(sbatch --exclude=aia-h200-7 --parsable --dependency=afterok:${TRAIN} ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_celeba_gen_eval_2k.sh "$ds")
    echo "  $ds: train=$TRAIN eval=$EVAL"
done

echo ""
echo "=== Total: 38 train+eval pairs ==="
echo "Monitor: squeue -u honjar"
