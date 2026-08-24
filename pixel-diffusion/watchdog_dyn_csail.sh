#!/bin/bash
# Self-healing watchdog for the CSAIL half of the dynamic-T search.
#
# Slurm retries a PREEMPTED job (we submit with --requeue) but not one that
# failed for its own reasons, and not one lost with a node. This loop closes
# that gap: every 15 min it resubmits any assigned run that has no result and
# no job in the queue, up to MAX_ATTEMPTS times.
#
# Runs in tmux ON THE LOGIN NODE so it survives ssh dropping -- CSAIL access is
# unreliable and nothing here may depend on an operator being reachable.
#
#   tmux new-session -d -s dynwd 'bash watchdog_dyn_csail.sh'
#   tmux attach -t dynwd
#
# It only ever touches runs in RUNS below. proline owns the rest, and a run must
# appear in exactly one machine's queue -- never widen this list without
# narrowing proline's.

set -u
BASE=/data/scratch/honjar
REPO=${BASE}/ambient-omni/pixel-diffusion
STATE=${BASE}/dyn_csail_state
LOG=${STATE}/watchdog.log
SEED=${SEED:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
POLL=${POLL:-300}
mkdir -p "$STATE"

RUNS="p0_cosine_pw5 p0_cosine_pw10 p1_a_linear_0to095 p1_a_twophase_050 \
p1_s_early_steep p1_s_early_mid p1_s_late_hard p1_s_late_extreme \
p1_q_sobol00 p1_q_sobol01 p1_q_sobol02 p1_q_sobol03 p1_q_sobol04 \
p1_q_sobol05 p1_q_sobol06 p1_q_sobol07 p1_q_sobol08 p1_q_sobol09"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watchdog up (pid $$), $(echo $RUNS | wc -w) runs assigned, max ${MAX_ATTEMPTS} attempts each"

while true; do
    done_n=0 pend_n=0 resub=0 exhausted=0
    # One squeue call, not one per run: cheap, and avoids hammering the
    # controller with 18 queries every cycle.
    QNAMES=$(squeue -h -u "$(whoami)" -o "%j" 2>/dev/null)
    for run in $RUNS; do
        if [ -f "${BASE}/generated/mind_dyn_${run}_s${SEED}.json" ]; then
            done_n=$((done_n+1)); continue
        fi
        if printf '%s\n' "$QNAMES" | grep -qx "dyn_${run}"; then
            pend_n=$((pend_n+1)); continue
        fi
        # `|| true`, never `|| echo 0`: grep -c PRINTS "0" and EXITS 1 when there
        # is no match, so `|| echo 0` yields the two-line value "0\n0" and every
        # later arithmetic on it dies. That fires on the first real resubmit --
        # exactly when the watchdog is the only thing keeping the queue alive.
        tries=$(grep -cx "$run" "${STATE}/attempts.txt" 2>/dev/null || true)
        tries=${tries:-0}
        if [ "$tries" -ge "$MAX_ATTEMPTS" ]; then
            exhausted=$((exhausted+1)); continue
        fi
        echo "$run" >> "${STATE}/attempts.txt"
        out=$(bash "${REPO}/submit_dyn_csail.sh" "$run" 2>&1 | grep -v '^sbatch:' | tail -1)
        log "resubmit ${run} (attempt $((tries+1))/${MAX_ATTEMPTS}): ${out}"
        resub=$((resub+1))
        sleep 2
    done
    log "cycle: done=${done_n} queued=${pend_n} resubmitted=${resub} exhausted=${exhausted}"
    if [ "$done_n" -eq "$(echo $RUNS | wc -w)" ]; then
        log "all assigned runs complete -- watchdog exiting."
        break
    fi
    sleep "$POLL"
done
