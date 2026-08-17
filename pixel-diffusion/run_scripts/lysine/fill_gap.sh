#!/bin/bash
# Fills the idle tail at the end of the replication chain.
#
# 10 runs across 4 GPUs does not divide evenly, so two GPUs finish their last
# seed-2 run ~7.7h before the other two finish theirs and sit idle. Rather than
# waste that, we spend it on a 4th seed of the two arms that decide the headline
# number (warmup_0to095 and linear_0to095, the top two at seed 0). More seeds on
# the arms under dispute is worth more than a first seed on an arm that already
# lost by a wide margin.
#
#     tmux new-session -d -s fill1 'GPU=1 ./run_scripts/lysine/fill_gap.sh'
#
# Waits for the tmux session named chain$GPU to disappear, which is the signal
# that this GPU's chain has run out of seed-1/seed-2 work and exited. Polling
# the session rather than the locks avoids firing during the gap between one
# experiment releasing a lock and the next claiming it.
#
# Entirely optional work: killing the tmux session, now or mid-run, costs
# nothing that the 3-seed result depends on.

set -u

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"


GPU=${GPU:-1}
SEED=${SEED:-3}
ARMS=${ARMS:-"warmup_0to095 linear_0to095"}
QUEUE=${QUEUE:-${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_scripts/lysine/run_dynamic_t_v2_queue.sh}

echo "[fill] GPU $GPU waiting for chain$GPU to finish ($(date '+%F %T'))"

while tmux has-session -t "chain$GPU" 2>/dev/null; do
  sleep 300
done

echo "[fill] chain$GPU done $(date '+%F %T'); starting seed $SEED on: $ARMS"

GPU=$GPU SEED=$SEED ONLY="$ARMS" bash "$QUEUE"

echo "[fill] done $(date '+%F %T')"
