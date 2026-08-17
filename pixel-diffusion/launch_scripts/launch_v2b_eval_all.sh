#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Idempotent eval launcher for ALL v2b experiments.
# Checks: checkpoint exists? MIND JSON exists? Only submits if training done but eval not done.

EVAL_SCRIPT="${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_v2_gen_eval.sh"
GEN_BASE="${AMBIENT_BASE}/generated"
EXCLUDE="aia-h200-7"
LOG_DIR="${AMBIENT_BASE}/train_logs"

submitted=0
already_done=0
not_ready=0

check_and_submit() {
    local NAME=$1
    local MIND_JSON="${GEN_BASE}/mind_${NAME}_2000kimg.json"
    local CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-002*.pkl 2>/dev/null | sort | tail -1)

    if [ -f "$MIND_JSON" ]; then
        already_done=$((already_done + 1))
        return
    fi

    if [ -z "$CKPT" ]; then
        echo "  NOT READY: $NAME"
        not_ready=$((not_ready + 1))
        return
    fi

    sbatch --exclude=$EXCLUDE \
        --job-name="eval_${NAME}" \
        --output="${LOG_DIR}/%j_eval_${NAME}.out" \
        "$EVAL_SCRIPT" "$NAME"
    submitted=$((submitted + 1))
}

echo "============================================"
echo "CelebA v2b — Unified Eval Launcher"
echo "============================================"
echo ""

# --- Coarse sweep (try both prefixes) ---
echo "--- Coarse sweep ---"
for PREFIX in celeba_v2b celeba_v2; do
    BNAME="${PREFIX}_baseline"
    if [ -d "${AMBIENT_BASE}/annotated_datasets/${BNAME}" ]; then
        check_and_submit "$BNAME"
        for b in 1 2 3 4 5 6 7; do
            for t in 000 020 040 060 080 090 095; do
                check_and_submit "${PREFIX}_b${b}_T${t}"
            done
        done
        echo "  (using prefix: ${PREFIX}_)"
        break
    fi
done

# --- Fine sweep ---
echo ""
echo "--- Fine sweep ---"
for b in 1 2; do
    for t in 030 035 045 050; do
        check_and_submit "celeba_v2b_b${b}_T${t}"
    done
done
for t in 030 035 045 050 055 065 070; do
    check_and_submit "celeba_v2b_b3_T${t}"
done
for b in 4 5; do
    for t in 050 055 065 070; do
        check_and_submit "celeba_v2b_b${b}_T${t}"
    done
done

# --- T->1 convergence ---
echo ""
echo "--- T->1 convergence ---"
for b in 1 5; do
    for t in 099 100; do
        check_and_submit "celeba_v2b_b${b}_T${t}"
    done
done

# --- Baseline re-seed ---
echo ""
echo "--- Baseline re-seed ---"
check_and_submit "celeba_v2b_baseline_s1"

echo ""
echo "============================================"
echo "Submitted: $submitted eval jobs"
echo "Already done: $already_done"
echo "Not ready: $not_ready"
echo "============================================"
