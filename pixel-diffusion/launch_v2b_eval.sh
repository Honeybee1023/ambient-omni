#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

EVAL_SCRIPT="${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
GEN_BASE="${AMBIENT_BASE}/generated"
EXCLUDE="aia-h200-7"
LOG_DIR="${AMBIENT_BASE}/train_logs"
submitted=0; done_count=0; not_ready=0

echo "============================================"
echo "CelebA v2b — Eval Launcher"
echo "============================================"

for NAME in celeba_v2b_baseline $(for b in 1 2 3 4 5 6 7; do for t in 000 020 040 060 080 090 095; do echo "celeba_v2b_b${b}_T${t}"; done; done); do
    if [ -f "${GEN_BASE}/mind_${NAME}_2000kimg.json" ]; then
        done_count=$((done_count+1)); continue; fi
    CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)
    if [ -z "$CKPT" ]; then
        not_ready=$((not_ready+1)); continue; fi
    sbatch --exclude=$EXCLUDE --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" "$EVAL_SCRIPT" "$NAME"
    submitted=$((submitted+1))
done

echo "Submitted: $submitted | Done: $done_count | Not ready: $not_ready"
