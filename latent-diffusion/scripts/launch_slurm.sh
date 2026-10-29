#!/bin/bash
# Multi-node training under Slurm: one scripts/run_train.sh per node, joined by torchrun rendezvous on the first node.
#
# Usage (inside an sbatch script with -N <nodes>, --ntasks-per-node=1 and --signal=B:USR1@<seconds>):
#     bash scripts/launch_slurm.sh <gpus_per_node> <train_edm2.py args...>
#     rc=$?; [ $rc -eq 3 ] && scontrol requeue $SLURM_JOB_ID
#
# Slurm's warning signal reaches this batch shell only (B:). It is NOT passed through srun (that signals every process
# on the node, torchrun included, which dies); this script writes <outdir>/STOP and each node's run_train.sh sends
# USR1 to its own training processes. Exit codes as run_train.sh: 0 finished, 3 stopped cleanly.
set -u
GPN=$1; shift
OUTDIR=""
for a in "$@"; do case $a in --outdir=*) OUTDIR=${a#--outdir=} ;; esac; done
[ -n "$OUTDIR" ] || { echo "launch_slurm.sh: pass --outdir=DIR"; exit 2; }
# Several runs in one job (gb allows only 2 running jobs per user): RUN_NODES=<n> NODE_OFFSET=<k> puts this run on
# nodes k..k+n-1 of the allocation. Defaults: all nodes.
OFF=${NODE_OFFSET:-0}
export NNODES=${RUN_NODES:-$SLURM_JOB_NUM_NODES}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "$((OFF + 1))p")
export MASTER_PORT=${MASTER_PORT:-29500}
mkdir -p "$OUTDIR"; rm -f "$OUTDIR/REQUEUE" "$OUTDIR/STOP"
HERE=$(cd "$(dirname "$0")" && pwd)
srun --export=ALL --nodes="$NNODES" --relative="$OFF" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
     bash -c 'NODE_RANK=$SLURM_PROCID exec bash "$0" "$@"' "$HERE/run_train.sh" "$GPN" "$@" &
SR=$!
forward() { echo "launch_slurm.sh: stop signal, asking every node to stop"; touch "$OUTDIR/STOP"; }
trap forward USR1 TERM
while kill -0 "$SR" 2>/dev/null; do sleep 5; done
wait "$SR"; rc=$?
rm -f "$OUTDIR/STOP"
if [ -f "$OUTDIR/REQUEUE" ]; then echo "launch_slurm.sh: stopped cleanly at $(cat "$OUTDIR/REQUEUE") images"; exit 3; fi
exit $rc
