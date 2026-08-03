#!/bin/bash
# Keeps the GPUs busy after the seed-0 sweep drains, without needing a human.
#
# The seed-0 queue answers "which schedule looks best". It cannot answer "is
# that gap real", because a single seed of this model has a MIND noise floor
# around 0.001 and the interesting gaps are only a few times that. So as soon
# as seed 0 finishes, this replicates the arms that actually placed, at seed 1
# and then seed 2, and leaves the rest alone.
#
# Which arms get replicated is decided when the sweep ends, not now: we take
# the top-N by MIND plus clean_only as the anchor. That way the choice reflects
# the finished sweep rather than my guess today.
#
#     tmux new-session -d -s chain0 'GPU=0 ./run_scripts/lysine/chain_replication.sh'
#     tmux new-session -d -s chain3 'GPU=3 ./run_scripts/lysine/chain_replication.sh'
#
# Safe to start while the seed-0 queue is still running: it waits. Safe to run
# on two GPUs at once: the queue's mkdir locks stop them duplicating work.

set -u

GPU=${GPU:-0}
TOPN=${TOPN:-4}
QUEUE=${QUEUE:-/data/honjar/ambient-omni/pixel-diffusion/run_scripts/lysine/run_dynamic_t_v2_queue.sh}
GENDIR=/data/honjar/generated
LOCKDIR=/data/honjar/train_logs/dynamic_t_v2/locks
PY=/data/honjar/miniconda3/envs/ambient/bin/python

echo "[chain] GPU $GPU waiting for the seed-0 sweep to drain ($(date '+%F %T'))"

# The sweep is done when nothing holds a seed-0 lock. Poll rather than watch a
# pid: the queue may be restarted by hand, and we want to survive that.
while true; do
  if ! ls -d "$LOCKDIR"/*_s0 >/dev/null 2>&1; then
    # Two consecutive clear reads, 5 min apart, so we do not fire during the
    # brief gap between one experiment releasing its lock and the next claiming.
    sleep 300
    ls -d "$LOCKDIR"/*_s0 >/dev/null 2>&1 || break
  fi
  sleep 300
done

echo "[chain] seed-0 sweep finished $(date '+%F %T'); picking arms to replicate"

ONLY=$("$PY" - "$GENDIR" "$TOPN" <<'EOF'
import glob, json, os, re, sys
gendir, topn = sys.argv[1], int(sys.argv[2])
rows = []
for p in glob.glob(os.path.join(gendir, "mind_v2_*_s0.json")):
    tag = re.match(r"mind_v2_(.+)_s0\.json$", os.path.basename(p)).group(1)
    if tag == "validate_T0475":      # the pipeline gate, not a research arm
        continue
    try:
        with open(p) as f:
            rows.append((json.load(f)["mind"], tag))
    except Exception:
        pass
rows.sort()
tags = [t for _, t in rows[:topn]]
if "clean_only" not in tags:          # anchor: every plot needs the baseline
    tags.append("clean_only")
print(" ".join(tags))
EOF
)

if [ -z "$ONLY" ]; then
  echo "[chain] no seed-0 results found; refusing to guess. Stopping."
  exit 1
fi

echo "[chain] replicating: $ONLY"

for SEED in 1 2; do
  echo ""
  echo "[chain] ===== seed $SEED on GPU $GPU ($(date '+%F %T')) ====="
  GPU=$GPU SEED=$SEED ONLY="$ONLY" bash "$QUEUE"
done

echo "[chain] all replication done $(date '+%F %T')"
