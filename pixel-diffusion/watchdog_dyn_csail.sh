#!/bin/bash
# Self-healing watchdog for the CSAIL half of the dynamic-T search.
#
# Slurm retries a PREEMPTED job (we submit with --requeue) but not one that
# failed for its own reasons, and not one lost with a node. This loop closes
# that gap: every 15 min it resubmits any assigned run that has no result and
# no job in the queue, up to MAX_ATTEMPTS times.
#
# Runs as a CPU Slurm job on tig-cpu (4-day walltime), NOT in login-node tmux:
# login nodes reboot, and nothing here may depend on an operator being
# reachable. Submit with --dependency=singleton on a fixed job name so a second
# copy can never race the first into double-submitting.
#
#   sbatch -p tig-cpu --qos=tig-main -t 4-00:00:00 --requeue \
#          --dependency=singleton -J dyn_watchdog \
#          --wrap 'bash watchdog_dyn_csail.sh'
#
# Verified 2026-08-24 that sbatch works from inside a job on this cluster.
# The attempts ledger lives on shared scratch, so a requeued watchdog resumes
# its counts rather than granting every run a fresh 5 attempts.
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

# The assignment lives in a file on shared scratch, not baked in here, and is
# re-read every cycle: rebalancing between proline and CSAIL then needs no
# restart of this job, and there is exactly one place that says who owns what.
ASSIGNED="${STATE}/assigned_runs.txt"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
if [ ! -s "$ASSIGNED" ]; then
    log "FATAL: no assignment file at $ASSIGNED -- refusing to guess what we own"
    exit 1
fi
log "watchdog up (pid $$), $(wc -l < "$ASSIGNED") runs assigned, max ${MAX_ATTEMPTS} attempts each"

while true; do
    done_n=0 pend_n=0 resub=0 exhausted=0
    # Re-read every cycle so a rebalance takes effect without a restart.
    RUNS=$(grep -v '^\s*$' "$ASSIGNED" 2>/dev/null | tr '\n' ' ')
    if [ -z "$RUNS" ]; then log "assignment file empty -- skipping cycle"; sleep "$POLL"; continue; fi
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
    if [ "$done_n" -eq "$(echo $RUNS | wc -w)" ] && [ "$done_n" -gt 0 ]; then
        log "all assigned runs complete -- watchdog exiting."
        break
    fi
    sleep "$POLL"
done
