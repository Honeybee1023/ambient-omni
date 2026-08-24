#!/bin/bash
# Opportunistic GPU-slot scheduler for the discrete dynamic-T schedule search.
#
#   bash run_dyn_queue.sh init [phase]   # build the queue (phase: 0, 1, or all)
#   bash run_dyn_queue.sh start          # start the scheduler (detached, in tmux)
#   bash run_dyn_queue.sh status         # progress + what is on each GPU
#   bash run_dyn_queue.sh probe          # why each GPU is or is not eligible
#   bash run_dyn_queue.sh stop           # stop scheduling (running jobs finish)
#
# Derived from run_restr_sweep_queue.sh, which drained 37 runs across two shared
# machines without losing one. Same hard-won guards, one generalisation:
# SLOTS_PER_GPU lets a single instance put more than one job on a card, so the
# two-instances-with-disjoint-queues dance is no longer needed.
#
# Designed to be abandoned: the scheduler lives in tmux and every job is
# setsid'd, so both outlive the ssh session. Every stage of every job is
# skip-if-done, so killing and restarting is always safe.

set -u

for _c in /data-local/honjar /var/local/honjar /data/scratch/honjar; do
    [ -n "${AMBIENT_BASE:-}" ] && break
    [ -d "$_c" ] && AMBIENT_BASE="$_c"
done
AMBIENT_BASE="${AMBIENT_BASE:-/data/scratch/honjar}"
export AMBIENT_BASE
REPO="${AMBIENT_BASE}/ambient-omni/pixel-diffusion"

STATE="${DYN_STATE:-${AMBIENT_BASE}/dyn_search_state}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
TMUX_SESSION="${DYN_SESSION:-dyn_search}"
QUEUE="${STATE}/queue.txt"
LOCK="${STATE}/queue.lock"
LOGDIR="${STATE}/logs"
SCHED_LOG="${STATE}/scheduler.log"
SCHED_PID="${STATE}/scheduler.pid"
STOP_FLAG="${STATE}/STOP"
MANIFEST="${AMBIENT_BASE}/generated/dyn_search_manifest.json"

# GPUs are addressed by UUID, never by index. CUDA and nvidia-smi enumerate
# differently when a session can only reach some cards: on proline a job
# launched with CUDA_VISIBLE_DEVICES=0 lands on nvidia-smi index 1. Checking
# free memory on one card and launching onto another is how the first attempt
# OOM'd. Both nvidia-smi --id= and CUDA_VISIBLE_DEVICES accept GPU-<uuid>.
GPUS=${GPUS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
# Upper bound on jobs per card. The memory floor below is what actually decides:
# on a 143GB H200 two jobs fit and a third does not; on an 80GB A100 the floor
# admits only one, which is right -- two jobs peaking at 46.5GB each would
# exceed the card during the first tick.
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
# Off by default. Demanding an idle card deadlocks where the only reachable
# GPUs are shared ones. UUID addressing means the free-memory check refers to
# the card the job actually lands on, which is the real protection.
FOREIGN_MAX_MB=${FOREIGN_MAX_MB:-100000000}
MIN_FREE_MB=${MIN_FREE_MB:-}
# Which dataset the jobs train on. Must be restated in the tmux command below
# and exported here: tmux does not inherit the launching shell's environment,
# and a scheduler that silently fell back to the default would put lysine on its
# own 182,598-image build instead of the 26,514 one the rest of the batch uses.
export DYN_DATASET="${DYN_DATASET:-}"
POLL=${POLL:-60}
STAGGER=${STAGGER:-45}
SEED=${SEED:-0}

mkdir -p "$STATE" "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- queue ------

cmd_init() {
    local phase=${1:-all}
    if [ ! -f "$MANIFEST" ]; then
        echo "No manifest at $MANIFEST -- run dynamic_t_search.py first." >&2; exit 1
    fi
    if [ -s "$QUEUE" ]; then
        echo "Queue already has $(wc -l < "$QUEUE") entries. Refusing to clobber." >&2
        echo "Delete $QUEUE first if you really want to rebuild it." >&2; exit 1
    fi
    # Phase 0 first and in listed order: if the piecewise code path is wrong,
    # that shows up in the validation runs before the search burns GPU-days.
    "${AMBIENT_BASE}/miniconda3/envs/ambient/bin/python" - "$MANIFEST" "$phase" > "$QUEUE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); want = sys.argv[2]
for e in m["runs"]:
    if want == "all" or str(e["phase"]) == want:
        print(e["name"])
PY
    : > "${STATE}/done.txt"; : > "${STATE}/failed.txt"; : > "${STATE}/started.txt"
    echo "Queued $(wc -l < "$QUEUE") runs (phase=$phase)."
}

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

already_done() {
    [ -f "${AMBIENT_BASE}/generated/mind_dyn_${1}_s${SEED}.json" ] && return 0
    # Work finished on another machine. The schedulers share no filesystem, so
    # neither sees the other's results. Prefer splitting the queue so each run
    # appears in exactly one queue; this is the backstop, not the plan.
    if [ -f "${STATE}/done_elsewhere.txt" ] && grep -qx "$1" "${STATE}/done_elsewhere.txt"; then
        return 0
    fi
    return 1
}

free_mb() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0; }

# An occupancy cap, not just an OOM guard. A job needs ~33GB (46.5GB at its
# first tick), so memory alone would fit four on a 143GB H200 -- but measured on
# proline, per-job cost went 11.4 -> 36 sec/kimg from two jobs on a card to
# three. That is worse than linear, so three per card has roughly HALF the
# throughput of two. Two thirds of the card free admits a first job (143>95) and
# a second (110>95) but not a third (77<95). The 55GB floor keeps an 80GB A100
# to one job, where a second would collide with the first tick's peak.
min_free_for() {
    if [ -n "$MIN_FREE_MB" ]; then echo "$MIN_FREE_MB"; return; fi
    local total; total=$(nvidia-smi --id="$1" --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null || echo 0)
    local cap=$(( ${total:-0} * 2 / 3 ))
    if [ "$cap" -gt 55000 ]; then echo "$cap"; else echo 55000; fi
}

foreign_mb() {
    local uuid=$1 me total=0
    me=$(id -u)
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
# CUDA reports "No CUDA GPUs are available" for two of four cards that
# nvidia-smi reports as idle. Without this the scheduler hands job after job to
# a card that kills each in seconds, draining the queue in minutes.
gpu_usable() {
    # Two statements on purpose: bash expands the whole `local` line before
    # assigning, so `local g=$1 marker=...${g}...` would expand ${g} from the
    # outer scope and trip `set -u`.
    local g=$1
    local marker="${STATE}/.gpu$(echo "$g" | tr -c 'A-Za-z0-9' '_').usable"
    # Cache the verdict, but let a negative expire: over a multi-day run a card
    # this session cannot currently touch may become available.
    if [ -f "$marker" ]; then
        if [ "$(cat "$marker")" = "yes" ]; then return 0; fi
        if [ -z "$(find "$marker" -mmin +60 2>/dev/null)" ]; then return 1; fi
    fi
    # Must actually allocate. torch.cuda.device_count() reports 1 even for cards
    # this session cannot touch.
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

# Kept as a subcommand rather than an inline `bash -c` string so the quoting
# cannot silently break (an earlier version wrote the literal text
# "$(date +%F_%T)" into failed.txt).
cmd_wrap() {
    local job=$1 gpu=$2 seed=$3 slot=${4:-0}
    echo $$ > "${STATE}/slot${slot}.pid"
    bash "${REPO}/run_dyn_job.sh" "$job" "$gpu" "$seed" "$slot"
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

slot_busy() {
    local pf="${STATE}/slot${1}.pid"
    [ -f "$pf" ] || return 1
    local pid; pid=$(cat "$pf" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 0
    rm -f "$pf"
    return 1
}

slot_job_file() { echo "${STATE}/slot${1}.job"; }

# ------------------------------------------------------------ scheduler ------

cmd_run() {
    log "scheduler up (pid $$) | slots/gpu=${SLOTS_PER_GPU} min_free=${MIN_FREE_MB:-per-card}MB poll=${POLL}s seed=${SEED}"
    log "queue: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) pending"
    local idle_notice=0

    while true; do
        if [ -f "$STOP_FLAG" ]; then log "STOP flag seen; not scheduling further."; break; fi

        local running=0 launched=0 gpu_idx=-1
        for gpu in $GPUS; do
            gpu_idx=$((gpu_idx+1))
            local s=0
            while [ "$s" -lt "$SLOTS_PER_GPU" ]; do
                local slot=$((gpu_idx * SLOTS_PER_GPU + s))
                s=$((s+1))
                if slot_busy "$slot"; then running=$((running+1)); continue; fi
                [ -s "$QUEUE" ] || continue

                local fm need; fm=$(free_mb "$gpu"); need=$(min_free_for "$gpu")
                if [ "${fm:-0}" -lt "$need" ]; then continue; fi
                # Probe before popping, so an unusable card never eats a job.
                gpu_usable "$gpu" || continue
                local fgn; fgn=$(foreign_mb "$gpu")
                if [ "${fgn:-0}" -gt "$FOREIGN_MAX_MB" ]; then continue; fi

                local job; job=$(pop_next)
                [ -n "$job" ] || continue
                if already_done "$job"; then
                    log "skip $job (MIND already on disk)"
                    echo "$job" >> "${STATE}/done.txt"; continue
                fi

                log "launch $job on $gpu slot $slot (${fm}MB free, ${fgn}MB foreign)"
                echo "$job" >> "${STATE}/started.txt"
                echo "$job" > "$(slot_job_file "$slot")"
                setsid nohup bash "$SELF" wrap "$job" "$gpu" "$SEED" "$slot" \
                    > "${LOGDIR}/${job}.log" 2>&1 &
                echo $! > "${STATE}/slot${slot}.pid"
                running=$((running+1)); launched=$((launched+1))
                # Let the new job claim its memory before the next free_mb read,
                # otherwise two launches in a row both see the card as empty.
                sleep "$STAGGER"
            done
        done

        local pending; pending=$(wc -l < "$QUEUE" 2>/dev/null || echo 0)
        if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
            log "queue drained and no jobs running -- scheduler exiting."; break
        fi

        if [ "$launched" -eq 0 ] && [ "$running" -eq 0 ] && [ "$pending" -gt 0 ]; then
            idle_notice=$((idle_notice+1))
            if [ $((idle_notice % 30)) -eq 1 ]; then
                log "waiting for capacity: $pending pending, free/need MB = [$(for g in $GPUS; do printf '%s/%s ' "$(free_mb $g)" "$(min_free_for $g)"; done)]"
            fi
        else
            idle_notice=0
        fi
        sleep "$POLL"
    done
    rm -f "$SCHED_PID"
    log "scheduler done. done=$(sort -u "${STATE}/done.txt" 2>/dev/null | wc -l) failed=$(wc -l < "${STATE}/failed.txt" 2>/dev/null || echo 0)"
}

cmd_start() {
    if [ -f "$SCHED_PID" ] && kill -0 "$(cat "$SCHED_PID")" 2>/dev/null; then
        echo "Scheduler already running (pid $(cat "$SCHED_PID"))."; exit 0
    fi
    [ -s "$QUEUE" ] || { echo "Queue is empty -- run '$0 init' first." >&2; exit 1; }
    rm -f "$STOP_FLAG"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists -- attach: tmux attach -t $TMUX_SESSION" >&2
        exit 1
    fi
    # The env must be restated inside the tmux command: a variable exported in
    # the shell that runs `start` does not reach the process tmux spawns, and a
    # second instance falling back to the default state dir would silently share
    # the first instance's queue.
    tmux new-session -d -s "$TMUX_SESSION" \
        "DYN_STATE='$STATE' DYN_SESSION='$TMUX_SESSION' SLOTS_PER_GPU='$SLOTS_PER_GPU' SEED='$SEED' GPUS='$GPUS' DYN_DATASET='${DYN_DATASET:-}' bash '$SELF' run 2>&1 | tee -a '${SCHED_LOG}'"
    sleep 3
    # tmux session presence, not pgrep: `pgrep -f "<script> run"` matches the
    # pgrep process itself, so it can never report zero.
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux list-panes -t "$TMUX_SESSION" -F '#{pane_pid}' 2>/dev/null | head -1 > "$SCHED_PID"
        echo "Scheduler started in tmux session '$TMUX_SESSION'."
        echo "  log:    $SCHED_LOG"
        echo "  attach: tmux attach -t $TMUX_SESSION"
    else
        echo "Scheduler failed to start -- check $SCHED_LOG" >&2; exit 1
    fi
}

cmd_stop() {
    touch "$STOP_FLAG"
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
    rm -f "$SCHED_PID"
    echo "Scheduler stopped. Jobs already running were left alone."
}

cmd_status() {
    echo "=== dynamic-T schedule search ==="
    echo "pending : $(wc -l < "$QUEUE" 2>/dev/null || echo 0)"
    echo "done    : $(sort -u "${STATE}/done.txt" 2>/dev/null | wc -l)"
    echo "failed  : $(wc -l < "${STATE}/failed.txt" 2>/dev/null || echo 0)"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "scheduler: RUNNING (tmux '$TMUX_SESSION')"
    else
        echo "scheduler: not running"
    fi
    echo
    printf "%-6s %-38s %-10s %s\n" SLOT GPU FREE_MB JOB
    local gpu_idx=-1
    for gpu in $GPUS; do
        gpu_idx=$((gpu_idx+1))
        local s=0
        while [ "$s" -lt "$SLOTS_PER_GPU" ]; do
            local slot=$((gpu_idx * SLOTS_PER_GPU + s)); s=$((s+1))
            local job="-"
            # Read the slot's own job file rather than grepping ps for the
            # dataset name: `ps | grep <name>` matches the grep itself and has
            # falsely reported a job as running before.
            if slot_busy "$slot"; then job=$(cat "$(slot_job_file "$slot")" 2>/dev/null || echo "(running)"); fi
            printf "%-6s %-38s %-10s %s\n" "$slot" "$gpu" "$(free_mb "$gpu")" "$job"
        done
    done
    echo
    echo "--- results so far ---"
    for f in "${AMBIENT_BASE}"/generated/mind_dyn_*.json; do
        [ -e "$f" ] || continue
        printf "  %-34s %s\n" "$(basename "$f" | sed 's/^mind_dyn_//; s/\.json$//')" \
            "$(grep -o '"mind"[^,}]*' "$f" | cut -d: -f2 | tr -d ' ')"
    done
}

case "${1:-}" in
    init)   cmd_init "${2:-all}" ;;
    start)  cmd_start ;;
    run)    cmd_run ;;
    wrap)   cmd_wrap "$2" "$3" "${4:-0}" "${5:-0}" ;;
    probe)  printf "%-6s %-10s %-10s %-6s %s\n" SMI_IDX FREE_MB FOREIGN_MB CUDA ELIGIBLE
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
    *) echo "usage: $0 {init [phase]|start|status|probe|stop}" >&2; exit 2 ;;
esac
