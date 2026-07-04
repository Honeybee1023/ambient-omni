#!/bin/bash
# Launch gen+eval for all completed v2 training jobs.
# Only submits eval for models that have a 2k checkpoint but no MIND result yet.

EVAL_SCRIPT="/data/scratch/honjar/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
GEN_BASE="/data/scratch/honjar/generated"
EXCLUDE="aia-h200-7"
LOG_DIR="/data/scratch/honjar/train_logs"

submitted=0
already_done=0
not_ready=0

echo "============================================"
echo "CelebA v2 — Eval Launcher"
echo "============================================"

for NAME in celeba_v2_baseline $(for b in 1 2 3 4 5 6 7; do for t in 000 020 040 060 080 090 095; do echo "celeba_v2_b${b}_T${t}"; done; done); do

    MIND_JSON="${GEN_BASE}/mind_${NAME}_2000kimg.json"
    CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)

    if [ -f "$MIND_JSON" ]; then
        already_done=$((already_done + 1))
        continue
    fi

    if [ -z "$CKPT" ]; then
        not_ready=$((not_ready + 1))
        continue
    fi

    sbatch --exclude=$EXCLUDE \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME"

    submitted=$((submitted + 1))
done

echo ""
echo "Submitted: $submitted eval jobs"
echo "Already done: $already_done"
echo "Not ready (no checkpoint): $not_ready"
echo "============================================"
