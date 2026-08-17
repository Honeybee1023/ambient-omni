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

for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break
    [ -d "$_c" ] && AMBIENT_BASE="$_c"
done
AMBIENT_BASE="${AMBIENT_BASE:-/data/scratch/honjar}"
export AMBIENT_BASE
REPO="${AMBIENT_BASE}/ambient-omni/pixel-diffusion"
STATE="${AMBIENT_BASE}/restr_sweep_state"
QUEUE="${STATE}/queue.txt"
LOCK="${STATE}/queue.lock"
LOGDIR="${STATE}/logs"
SCHED_LOG="${STATE}/scheduler.log"
SCHED_PID="${STATE}/scheduler.pid"
STOP_FLAG="${STATE}/STOP"
MANIFEST="${AMBIENT_BASE}/generated/restr_sweep_manifest.json"

# GPUs are addressed by UUID, never by index. CUDA and nvidia-smi enumerate
# differently when a session can only reach some of the cards: on proline a job
# launched with CUDA_VISIBLE_DEVICES=0 lands on nvidia-smi index 1. Checking free
# memory on index 0 and then launching onto a different physical card is how the
# first attempt OOM'd against a neighbour. Both nvidia-smi --id= and
# CUDA_VISIBLE_DEVICES accept a GPU-<uuid> string, so the UUID is the one handle
# that means the same thing to the guard and to the job.
GPUS=${GPUS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
# How many times a dataset may be attempted before it is left alone. Failures
# here are usually a neighbour growing into the card rather than anything wrong
# with the job, so it is worth retrying a few times over a multi-day run.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
# Off by default. Demanding an idle card sounds safer but deadlocks where the
# only reachable GPUs are shared ones: on proline the two cards CUDA can address
# both permanently host another user, while the two idle cards are unreachable.
# Now that GPUs are addressed by UUID the free-memory check refers to the card
# the job will actually land on, so a margin there is the real protection. Set
# this to e.g. 10000 to insist on idle cards where that is affordable.
FOREIGN_MAX_MB=${FOREIGN_MAX_MB:-100000000}
# A run settles at ~33GB but spikes to ~46.5GB reserved during the first tick.
# The margin above that is deliberate: on a shared card the neighbour keeps
# allocating after we check, and an OOM costs hours of queue time.
# Empty when unset, meaning "derive per card" (see min_free_for below). An
# explicit value overrides for every GPU.
MIN_FREE_MB=${MIN_FREE_MB:-}
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
    if [ -f "${AMBIENT_BASE}/generated/mind_${1}_2000kimg.json" ] && \
       [ -f "${AMBIENT_BASE}/generated/valloss_${1}_2000kimg.json" ]; then
        return 0
    fi
    # Work finished on another machine. The two schedulers share no filesystem,
    # so neither can see the other's results; without this they would both train
    # whatever is left in the middle of the queue. Refresh from the machine that
    # is ahead, e.g. from the Mac which can reach both:
    #   ssh <other> 'ls $BASE/generated/mind_celeba_v2b_restr_*_2000kimg.json' \
    #     | xargs -n1 basename | sed 's/^mind_//; s/_2000kimg\.json$//' \
    #     | ssh <this> 'cat >> '"$STATE"'/done_elsewhere.txt'
    if [ -f "${STATE}/done_elsewhere.txt" ] && grep -qx "$1" "${STATE}/done_elsewhere.txt"; then
        return 0
    fi
    return 1
}

free_mb() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0; }

# How much free memory this particular card must have before we take it. A run
# settles near 33GB but spikes to ~46.5GB reserved on the first tick, so 55GB is
# the floor anywhere. On a larger card ask for half of it instead: there the
# neighbours are larger too, and an OOM costs hours of queue time. A fixed number
# cannot serve both -- 70GB is right for a 143GB H200 but would mean an 80GB A100
# never qualifies at all.
min_free_for() {
    if [ -n "$MIN_FREE_MB" ]; then echo "$MIN_FREE_MB"; return; fi
    local total; total=$(nvidia-smi --id="$1" --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null || echo 0)
    local half=$(( ${total:-0} / 2 ))
    if [ "$half" -gt 55000 ]; then echo "$half"; else echo 55000; fi
}

# Memory held on a card by processes that are not ours. Returns 0 when the card
# is idle or only we are on it.
foreign_mb() {
    local uuid=$1
    local me total=0
    me=$(id -u)
    # gpu_uuid ties each compute process to a physical card unambiguously.
    while IFS=, read -r u pid mem; do
        u=$(echo "$u" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ' | tr -dc '0-9')
        [ "$u" = "$uuid" ] || continue
        [ -n "$pid" ] && [ "$pid" != "[N/A]" ] || continue
        local owner; owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$owner" ] || continue
        [ "$owner" = "$me" ] && continue
        total=$((total + ${mem:-0}))
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
             --format=csv,noheader,nounits 2>/dev/null)
    echo "$total"
}

# nvidia-smi listing a GPU does not mean this session may use it: on proline
# CUDA reports "No CUDA GPUs are available" for two of the four cards that
# nvidia-smi happily reports as idle. Without this check the scheduler hands job
# after job to a card that kills each one in seconds, draining the whole queue.
# Cached, because it costs a couple of seconds per probe.
gpu_usable() {
    # Two statements on purpose: bash expands the whole `local` line before it
    # assigns, so `local g=$1 marker=...${g}...` would expand ${g} from the outer
    # scope and trip `set -u` when nothing there is called g.
    local g=$1
    local marker="${STATE}/.gpu$(echo "$g" | tr -c 'A-Za-z0-9' '_').usable"
    # Cache the verdict, but let a negative expire: over a multi-day run a card
    # this session cannot currently touch may become available, and a permanently
    # cached "no" would leave it idle for the rest of the sweep.
    if [ -f "$marker" ]; then
        if [ "$(cat "$marker")" = "yes" ]; then return 0; fi
        if [ -z "$(find "$marker" -mmin +60 2>/dev/null)" ]; then return 1; fi
    fi
    # Must actually allocate. torch.cuda.device_count() reports 1 even for cards
    # this session cannot touch -- on proline it returns 1 for all four H200s
    # while allocating on two of them raises "No CUDA GPUs are available".
    local out
    out=$(CUDA_VISIBLE_DEVICES="$g" "${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python" -c \
        "import torch
try:
    torch.zeros(256, 256, device='cuda'); print(1)
except Exception:
    print(0)" 2>/dev/null | tail -1)
    if [ "$out" = "1" ]; then
        echo yes > "$marker"; log "GPU $g is usable by CUDA"; return 0
    fi
    echo no > "$marker"
    log "GPU $g is visible to nvidia-smi but NOT to CUDA -- excluding it"
    return 1
}

# Runs one job detached, then records the outcome and puts the dataset back on
# the queue if it failed and has attempts left. Kept as a subcommand rather than
# an inline `bash -c` string so the quoting cannot silently break (an earlier
# version wrote the literal text "$(date +%F_%T)" into failed.txt).
cmd_wrap() {
    local job=$1 gpu=$2 seed=$3 slot=${4:-0}
    echo $$ > "${STATE}/gpu${slot}.pid"
    bash "${REPO}/run_restr_job.sh" "$job" "$gpu" "$seed" "$slot"
    local ec=$?
    if [ "$ec" -eq 0 ]; then
        echo "$job" >> "${STATE}/done.txt"
    else
        echo "$(date '+%F %T') $job gpu=$gpu exit=$ec" >> "${STATE}/failed.txt"
        local tries; tries=$(grep -c "^${job}$" "${STATE}/started.txt" 2>/dev/null || echo 0)
        if [ "$tries" -lt "${MAX_ATTEMPTS}" ]; then
            # Back of the queue, not the front: a job that fails for its own
            # reasons must not spin ahead of work that can still succeed.
            (
                flock -x 200
                { cat "$QUEUE" 2>/dev/null; echo "$job"; } > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
            ) 200>"$LOCK"
            echo "$(date '+%F %T') requeued $job (attempt $tries/${MAX_ATTEMPTS})" >> "${STATE}/failed.txt"
        fi
    fi
}

slot_busy() {   # $1 = slot index
    local pf="${STATE}/gpu${1}.pid"
    [ -f "$pf" ] || return 1
    local pid; pid=$(cat "$pf" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 0
    rm -f "$pf"
    return 1
}

# ------------------------------------------------------------ scheduler ------

cmd_run() {
    log "scheduler up (pid $$) | min_free=${MIN_FREE_MB:-per-card}MB poll=${POLL}s seed=${SEED}"
    log "queue: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) pending"
    local idle_notice=0

    while true; do
        if [ -f "$STOP_FLAG" ]; then
            log "STOP flag seen; not scheduling anything further."
            break
        fi

        local running=0 launched=0 slot=-1
        for gpu in $GPUS; do
            slot=$((slot+1))
            if slot_busy "$slot"; then running=$((running+1)); continue; fi

            [ -s "$QUEUE" ] || continue

            local fm need; fm=$(free_mb "$gpu"); need=$(min_free_for "$gpu")
            if [ "${fm:-0}" -lt "$need" ]; then continue; fi

            # Probe before popping, so an unusable card never consumes a job.
            gpu_usable "$gpu" || continue

            local fgn; fgn=$(foreign_mb "$gpu")
            if [ "${fgn:-0}" -gt "$FOREIGN_MAX_MB" ]; then continue; fi

            local job; job=$(pop_next)
            [ -n "$job" ] || continue

            if already_done "$job"; then
                log "skip $job (metrics already on disk)"
                echo "$job" >> "${STATE}/done.txt"
                continue
            fi

            log "launch $job on $gpu (${fm}MB free, ${fgn}MB foreign)"
            echo "$job" >> "${STATE}/started.txt"
            setsid nohup bash "${REPO}/run_restr_sweep_queue.sh" wrap "$job" "$gpu" "$SEED" "$slot" \
                > "${LOGDIR}/${job}.log" 2>&1 &
            echo $! > "${STATE}/gpu${slot}.pid"
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
                log "waiting for capacity: $pending pending, GPU free MB = [$(for g in $GPUS; do printf '%s/%s ' "$(free_mb $g)" "$(min_free_for $g)"; done)]"
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
    wrap)   cmd_wrap "$2" "$3" "${4:-0}" ;;
    probe)  # why each GPU is or is not eligible right now
            printf "%-6s %-10s %-10s %-6s %s\n" SMI_IDX FREE_MB FOREIGN_MB CUDA ELIGIBLE
            for g in $GPUS; do
                idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
                      | awk -F', ' -v u="$g" '$2==u{print $1}')
                fm=$(free_mb "$g"); fg=$(foreign_mb "$g")
                if gpu_usable "$g" >/dev/null 2>&1; then cu=yes; else cu=NO; fi
                el=yes
                [ "$cu" = "NO" ] && el="no (CUDA cannot see it)"
                nd=$(min_free_for "$g")
                [ "$el" = "yes" ] && [ "${fm:-0}" -lt "$nd" ] && el="no (needs ${nd}MB free)"
                [ "$el" = "yes" ] && [ "${fg:-0}" -gt "$FOREIGN_MAX_MB" ] && el="no (another user holds ${fg}MB)"
                printf "%-6s %-10s %-10s %-6s %s\n" "${idx:-?}" "$fm" "$fg" "$cu" "$el"
            done ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *) echo "usage: $0 {init|start|status|stop}" >&2; exit 2 ;;
esac
