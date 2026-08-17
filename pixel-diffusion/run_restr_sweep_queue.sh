#!/bin/bash
# Opportunistic GPU-slot scheduler for the restricted-bucket sweep on lysine.
#
# lysine is a shared box: at the time of writing all four A100s were held by
# other users. This drains a queue of experiments onto whichever GPUs are free,
# one job per GPU, claiming each slot as soon as it opens and never holding one
# idle while work remains.
#
#   bash run_restr_sweep_queue.sh init     # build the queue from the manifest
#   bash run_restr_sweep_queue.sh start    # start the scheduler (detached, in tmux)
#   bash run_restr_sweep_queue.sh status   # progress + what is on each GPU
#   bash run_restr_sweep_queue.sh stop     # stop scheduling (running jobs finish)
#
# Designed to be abandoned: it runs under tmux + setsid, so it outlives the ssh
# session that launched it. Every stage of every job is skip-if-done, so killing
# and restarting the scheduler is always safe -- it picks up where it left off.

set -u

AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"
REPO="${AMBIENT_BASE}/ambient-omni/pixel-diffusion"
STATE="${AMBIENT_BASE}/restr_sweep_state"
QUEUE="${STATE}/queue.txt"
LOCK="${STATE}/queue.lock"
LOGDIR="${STATE}/logs"
SCHED_LOG="${STATE}/scheduler.log"
SCHED_PID="${STATE}/scheduler.pid"
STOP_FLAG="${STATE}/STOP"
MANIFEST="${AMBIENT_BASE}/generated/restr_sweep_manifest.json"

GPUS="0 1 2 3"
# A run settles at ~33GB but spikes to ~46.5GB reserved during the first tick,
# so this is the honest floor. Starting below it risks OOMing our own job and,
# worse, a neighbour's. On an 80GB A100 this means we take a card only when it
# is genuinely idle or lightly used.
MIN_FREE_MB=${MIN_FREE_MB:-50000}
POLL=${POLL:-60}
# Small gap between two launches so four jobs do not hit the disk at once.
STAGGER=${STAGGER:-45}
SEED=${SEED:-0}

mkdir -p "$STATE" "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- queue ------

cmd_init() {
    if [ ! -f "$MANIFEST" ]; then
        echo "No manifest at $MANIFEST -- run create_v2b_restricted_sweeps.py first." >&2
        exit 1
    fi
    if [ -s "$QUEUE" ]; then
        echo "Queue already exists with $(wc -l < "$QUEUE") entries. Refusing to clobber." >&2
        echo "Delete $QUEUE first if you really want to rebuild it." >&2
        exit 1
    fi
    "${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python" - "$MANIFEST" > "$QUEUE" <<'PY'
import json, sys
for name in json.load(open(sys.argv[1])):
    print(name)
PY
    : > "${STATE}/done.txt"; : > "${STATE}/failed.txt"; : > "${STATE}/started.txt"
    echo "Queued $(wc -l < "$QUEUE") experiments."
}

# Pop the head of the queue under an exclusive lock, so the scheduler can never
# hand the same dataset to two GPUs.
pop_next() {
    local popped="${STATE}/.popped"
    (
        flock -x 200
        if [ -s "$QUEUE" ]; then
            head -1 "$QUEUE" > "$popped"
            tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
        else
            : > "$popped"
        fi
    ) 200>"$LOCK"
    cat "$popped"
}

requeue() {   # put a job back on the front, for transient failures
    (
        flock -x 200
        { echo "$1"; cat "$QUEUE" 2>/dev/null; } > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
    ) 200>"$LOCK"
}

already_done() {
    [ -f "${AMBIENT_BASE}/generated/mind_${1}_2000kimg.json" ] && \
    [ -f "${AMBIENT_BASE}/generated/valloss_${1}_2000kimg.json" ]
}

free_mb() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0; }

slot_busy() {   # $1 = gpu id
    local pf="${STATE}/gpu${1}.pid"
    [ -f "$pf" ] || return 1
    local pid; pid=$(cat "$pf" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 0
    rm -f "$pf"
    return 1
}

# ------------------------------------------------------------ scheduler ------

cmd_run() {
    log "scheduler up (pid $$) | min_free=${MIN_FREE_MB}MB poll=${POLL}s seed=${SEED}"
    log "queue: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) pending"
    local idle_notice=0

    while true; do
        if [ -f "$STOP_FLAG" ]; then
            log "STOP flag seen; not scheduling anything further."
            break
        fi

        local running=0 launched=0
        for gpu in $GPUS; do
            if slot_busy "$gpu"; then running=$((running+1)); continue; fi

            [ -s "$QUEUE" ] || continue

            local fm; fm=$(free_mb "$gpu")
            if [ "${fm:-0}" -lt "$MIN_FREE_MB" ]; then continue; fi

            local job; job=$(pop_next)
            [ -n "$job" ] || continue

            if already_done "$job"; then
                log "skip $job (metrics already on disk)"
                echo "$job" >> "${STATE}/done.txt"
                continue
            fi

            log "launch $job on GPU $gpu (${fm}MB free)"
            echo "$job" >> "${STATE}/started.txt"
            setsid nohup bash -c "
                bash '${REPO}/run_restr_job.sh' '$job' '$gpu' '$SEED'
                ec=\$?
                if [ \$ec -eq 0 ]; then echo '$job' >> '${STATE}/done.txt'
                else echo '\$(date +%F_%T) $job exit=\$ec' >> '${STATE}/failed.txt'; fi
            " > "${LOGDIR}/${job}.log" 2>&1 &
            echo $! > "${STATE}/gpu${gpu}.pid"
            running=$((running+1)); launched=$((launched+1))
            sleep "$STAGGER"
        done

        local pending; pending=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
        if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
            log "queue drained and no jobs running -- scheduler exiting."
            break
        fi

        # Report roughly every 30 min while blocked, so the log shows *why* we are waiting.
        if [ "$launched" -eq 0 ] && [ "$running" -eq 0 ] && [ "$pending" -gt 0 ]; then
            idle_notice=$((idle_notice+1))
            if [ $((idle_notice % 30)) -eq 1 ]; then
                log "waiting for capacity: $pending pending, GPU free MB = [$(for g in $GPUS; do printf '%s ' "$(free_mb $g)"; done)] need ${MIN_FREE_MB}"
            fi
        else
            idle_notice=0
        fi

        sleep "$POLL"
    done
    rm -f "$SCHED_PID"
    log "scheduler done. done=$(wc -l < "${STATE}/done.txt" 2>/dev/null || echo 0) failed=$(wc -l < "${STATE}/failed.txt" 2>/dev/null || echo 0)"
}

cmd_start() {
    if [ -f "$SCHED_PID" ] && kill -0 "$(cat "$SCHED_PID")" 2>/dev/null; then
        echo "Scheduler already running (pid $(cat "$SCHED_PID"))."; exit 0
    fi
    [ -s "$QUEUE" ] || { echo "Queue is empty -- run '$0 init' first." >&2; exit 1; }
    rm -f "$STOP_FLAG"
    # The scheduler lives in tmux, which outlives the ssh session that started it.
    # The jobs it spawns are setsid'd separately, so they survive even if the
    # scheduler or the whole tmux server goes away mid-run.
    if tmux has-session -t restr_sweep 2>/dev/null; then
        echo "tmux session 'restr_sweep' already exists -- attach with: tmux attach -t restr_sweep" >&2
        exit 1
    fi
    tmux new-session -d -s restr_sweep \
        "bash '${REPO}/run_restr_sweep_queue.sh' run 2>&1 | tee -a '${SCHED_LOG}'"
    sleep 3
    pgrep -f "run_restr_sweep_queue.sh run" | head -1 > "$SCHED_PID"
    if [ -s "$SCHED_PID" ]; then
        echo "Scheduler started in tmux session 'restr_sweep' (pid $(cat "$SCHED_PID"))."
        echo "  log:    $SCHED_LOG"
        echo "  attach: tmux attach -t restr_sweep"
    else
        echo "Scheduler failed to start -- check $SCHED_LOG" >&2; exit 1
    fi
}

cmd_stop() {
    touch "$STOP_FLAG"
    if [ -f "$SCHED_PID" ]; then kill "$(cat "$SCHED_PID")" 2>/dev/null; rm -f "$SCHED_PID"; fi
    echo "Scheduler stopped. Jobs already running were left alone."
}

cmd_status() {
    echo "=== restricted-bucket sweep ==="
    echo "pending : $(wc -l < "$QUEUE" 2>/dev/null || echo 0)"
    echo "done    : $(sort -u "${STATE}/done.txt" 2>/dev/null | wc -l)"
    echo "failed  : $(wc -l < "${STATE}/failed.txt" 2>/dev/null || echo 0)"
    if [ -f "$SCHED_PID" ] && kill -0 "$(cat "$SCHED_PID")" 2>/dev/null; then
        echo "scheduler: RUNNING (pid $(cat "$SCHED_PID"))"
    else
        echo "scheduler: not running"
    fi
    echo
    printf "%-6s %-10s %s\n" GPU FREE_MB JOB
    for gpu in $GPUS; do
        local job="-"
        if slot_busy "$gpu"; then
            job=$(tail -1 "${STATE}/started.txt" 2>/dev/null)
            job=$(ls -t "${LOGDIR}"/*.log 2>/dev/null | head -4 | xargs -r grep -l "GPU ${gpu} " 2>/dev/null | head -1 | xargs -r basename 2>/dev/null | sed 's/\.log$//')
            [ -n "$job" ] || job="(running)"
        fi
        printf "%-6s %-10s %s\n" "$gpu" "$(free_mb "$gpu")" "$job"
    done
    echo
    echo "--- results so far ---"
    for f in "${AMBIENT_BASE}"/generated/mind_celeba_v2b_restr_*_2000kimg.json; do
        [ -e "$f" ] || continue
        printf "  %-34s %s\n" "$(basename "$f" | sed 's/^mind_//; s/_2000kimg\.json$//')" \
            "$(grep -o '"mind"[^,]*' "$f" | cut -d: -f2 | tr -d ' ')"
    done
}

case "${1:-}" in
    init)   cmd_init ;;
    start)  cmd_start ;;
    run)    cmd_run ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *) echo "usage: $0 {init|start|status|stop}" >&2; exit 2 ;;
esac
