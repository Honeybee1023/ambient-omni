#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

echo "=== Training Progress ==="
DONE=$(ls ${AMBIENT_BASE}/train_outputs/*-celeba_2d_*/network-snapshot-001000.pkl 2>/dev/null | wc -l)
TOTAL=98
echo "  Completed: $DONE / $TOTAL"
echo ""
echo "=== Stability Test ==="
ls ${AMBIENT_BASE}/train_outputs/*-celeba_2d_b3_T050-*/network-snapshot-*.pkl 2>/dev/null | sort
echo ""
echo "=== Gen+Eval Progress ==="
METRICS=$(ls ${AMBIENT_BASE}/generated/metrics_celeba_2d_*_1000kimg.json 2>/dev/null | wc -l)
echo "  Metrics computed: $METRICS / $DONE"
echo ""
echo "=== FID Reference Stats ==="
ls -la ${AMBIENT_BASE}/celeba_processed/celeba_holdout_ref_stats.npz 2>/dev/null || echo "  NOT READY"
echo ""
echo "=== Queue ==="
squeue -u honjar | head -5
squeue -u honjar | wc -l
