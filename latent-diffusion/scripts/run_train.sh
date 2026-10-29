#!/bin/bash
# Run train_edm2.py under torchrun on one node so that a Slurm warning signal stops it cleanly.
#
# Usage (inside an sbatch script started with --signal=B:USR1@<seconds>; several nodes: scripts/launch_slurm.sh):
#     bash scripts/run_train.sh <gpus_per_node> <train_edm2.py args...>
#     rc=$?; [ $rc -eq 3 ] && scontrol requeue $SLURM_JOB_ID
#
# Slurm sends USR1 to this batch shell only (B:). The trap forwards it to the training processes (torchrun's children),
# not to torchrun: torchrun dies on USR1 and would take a half-written checkpoint with it. Each training process
# finishes its step, all ranks save one training-state checkpoint, rank 0 writes <outdir>/REQUEUE, and they exit.
# Exit codes: 0 = finished, 3 = stopped cleanly (resume by rerunning the same command), other = crash.
set -u
NPROC=$1; shift
OUTDIR=""
for a in "$@"; do case $a in --outdir=*) OUTDIR=${a#--outdir=} ;; esac; done
[ -n "$OUTDIR" ] || { echo "run_train.sh: pass --outdir=DIR"; exit 2; }
[ "${NNODES:-1}" -gt 1 ] || rm -f "$OUTDIR/REQUEUE" "$OUTDIR/STOP"   # multi-node: launch_slurm.sh clears them
cd "$(dirname "$0")/.."
if [ "${NNODES:-1}" -gt 1 ]; then
  # One copy of this script per node (scripts/launch_slurm.sh starts them); torchrun joins them by rendezvous.
  torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC" --rdzv_backend=c10d \
           --rdzv_endpoint="$MASTER_ADDR:${MASTER_PORT:-29500}" --rdzv_id="${SLURM_JOB_ID:-0}" train_edm2.py "$@" &
else
  torchrun --standalone --nproc_per_node="$NPROC" train_edm2.py "$@" &
fi
TR=$!
forward() { echo "run_train.sh: stop signal, forwarding USR1 to the training processes"; pkill -USR1 -P "$TR"; }
trap forward USR1 TERM
# Multi-node: srun would deliver a signal to every process on the node (torchrun included, which dies on it), so
# launch_slurm.sh drops <outdir>/STOP instead and each node forwards USR1 to its own training processes only.
sent=0
while kill -0 "$TR" 2>/dev/null; do
  if [ $sent -eq 0 ] && [ -f "$OUTDIR/STOP" ]; then forward; sent=1; fi
  sleep 5
done
wait "$TR"; rc=$?
if [ -f "$OUTDIR/REQUEUE" ]; then echo "run_train.sh: stopped cleanly at $(cat "$OUTDIR/REQUEUE") images"; exit 3; fi
exit $rc
