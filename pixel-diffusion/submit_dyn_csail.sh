#!/bin/bash
# Submit dynamic-T schedule runs to CSAIL Slurm, one GPU per run.
#
# CSAIL is the only one of our machines with a scheduler, so the polling
# scheduler used on proline/lysine is not the right tool here -- Slurm already
# does the queueing. One sbatch per run, 1 GPU each.
#
#   bash submit_dyn_csail.sh run_a run_b ...      # submit these runs
#   DRY=1 bash submit_dyn_csail.sh ...            # print without submitting
#
# Each run must appear in exactly ONE machine's queue. This does not check
# proline's queue (it cannot see it) -- partition the work before calling this.
#
# --requeue plus the job script's resume-from-dump makes preemption survivable:
# the shared QOS can evict us, and a re-queued job restarts from the newest
# training-state (dumped every 5 ticks = 250 kimg, ~1.6h of work at risk).

set -u
BASE=/data/scratch/honjar
REPO=${BASE}/ambient-omni/pixel-diffusion
LOGDIR=${BASE}/train_logs/dyn_search
PART=${PART:-csail-shared-h200}
QOS=${QOS:-shared-if-available}
GRES=${GRES:-gpu:h200:1}
TIME=${TIME:-24:00:00}
SEED=${SEED:-0}
mkdir -p "$LOGDIR"

[ $# -gt 0 ] || { echo "usage: $0 <run_name> [run_name ...]" >&2; exit 2; }

for run in "$@"; do
    if [ -f "${BASE}/generated/mind_dyn_${run}_s${SEED}.json" ]; then
        echo "SKIP $run (MIND already present)"; continue
    fi
    # Guard against submitting the same run twice: Slurm job names are ours to
    # choose, so an existing pending/running job with this name means it is
    # already covered.
    if squeue -h -u "$(whoami)" -n "dyn_${run}" -o "%i" 2>/dev/null | grep -q .; then
        echo "SKIP $run (already queued/running)"; continue
    fi
    cmd=(sbatch --parsable -D "$BASE"
         -o "${LOGDIR}/${run}-%j.out" -J "dyn_${run}"
         -p "$PART" --qos="$QOS" --gres="$GRES"
         --cpus-per-task=8 --mem=96G -t "$TIME" --requeue
         --wrap "bash ${REPO}/run_dyn_job.sh ${run} slurm ${SEED} 0")
    if [ -n "${DRY:-}" ]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
    id=$("${cmd[@]}" 2>&1 | tail -1)
    echo "submitted $run -> job $id"
done
