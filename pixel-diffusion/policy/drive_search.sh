#!/bin/bash
# Mac-side driver for the policy search: runs policy/search_step.py on CSAIL every 10 minutes and
# EXITS (waking the session that launched it) when a generation finishes, the search ends, a
# candidate fails, or anything errors. A 6-hour heartbeat exit makes a stall visible too.
# Re-launch it after handling each exit; all search state is on CSAIL, so nothing is lost.
start=$(date +%s)
while true; do
    out=$(ssh -o ConnectTimeout=30 csail-slurm \
        'cd /data/scratch/honjar/ambient-omni/pixel-diffusion && /data/scratch/honjar/miniconda3/envs/ambient/bin/python policy/search_step.py 2>&1' \
        2>/dev/null | grep -v -E "bashrc|autocast|warnings.warn")
    [ -n "$out" ] && echo "[$(date '+%m-%d %H:%M')] $out" | tail -3
    if echo "$out" | grep -q -E "GEN_DONE|SEARCH_DONE|FAILED|SUBMITTED|Traceback|Error"; then
        echo "EVENT"; exit 0
    fi
    if [ $(( $(date +%s) - start )) -ge 21600 ]; then echo "HEARTBEAT: 6h without an event"; exit 0; fi
    sleep 600
done
