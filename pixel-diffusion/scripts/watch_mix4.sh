#!/bin/bash
# Mac-side watcher for the mixed-blur go/no-go study on CSAIL. Polls every 10 min and EXITS
# (which wakes the session that launched it) as soon as something needs attention:
#   - a new MIND result (mind_dyn_mix4_*.json) appears
#   - a job log shows a Traceback / ERROR
#   - the classifier reaches its final snapshot; the annotation finishes
#   - heartbeat: 6 hours with no event, so a stall is noticed too
# State (what was already reported) lives in $STATE so a re-launch only reports new events.
STATE=${STATE:-$HOME/ambient-omni-private-notes/.watch_mix4_state}
touch "$STATE"
start=$(date +%s)
while true; do
    snap=$(ssh -o ConnectTimeout=30 csail-slurm '
        B=/data/scratch/honjar
        for f in $B/generated/mind_dyn_mix4_*.json; do [ -f "$f" ] && echo "RESULT $(basename $f)"; done
        for f in $B/train_logs/dyn_search/mix4_*.out $B/train_logs/mix4/*.out; do
            [ -f "$f" ] && grep -q -E "Traceback|^ERROR" "$f" && echo "ERROR $(basename $f)"; done
        [ -f $B/train_outputs/cls_mix4/network-snapshot-007680.pkl ] && echo "CLS_DONE"
        [ -f $B/annotated_datasets/celeba_mix4_ambo/threshold_summary.json ] && echo "ANN_DONE"
        grep -q "^restored set: 26514" $B/train_logs/mix4/dataloops1-*.out 2>/dev/null && echo "RESTORED"
        squeue -h -u honjar -o "QUEUED %j"
    ' 2>/dev/null) || { sleep 600; continue; }
    new=$(echo "$snap" | grep -E "^(RESULT|ERROR|CLS_DONE|ANN_DONE|RESTORED)" | grep -v -x -F -f "$STATE")
    if [ -n "$new" ]; then
        echo "$new" >> "$STATE"
        echo "EVENT at $(date):"; echo "$new"
        echo "--- queue:"; echo "$snap" | grep ^QUEUED
        exit 0
    fi
    if [ $(( $(date +%s) - start )) -ge 21600 ]; then
        echo "HEARTBEAT at $(date): no new event in 6h"; echo "$snap" | grep ^QUEUED; exit 0
    fi
    sleep 600
done
