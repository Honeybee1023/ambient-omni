#!/bin/bash
# Run train_edm2.py under torchrun on one node so that a Slurm warning signal stops it cleanly.
#
# Usage (inside an sbatch script started with --signal=B:USR1@<seconds>):
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
rm -f "$OUTDIR/REQUEUE"
cd "$(dirname "$0")/.."
torchrun --standalone --nproc_per_node="$NPROC" train_edm2.py "$@" &
TR=$!
forward() { echo "run_train.sh: stop signal, forwarding USR1 to the training processes"; pkill -USR1 -P "$TR"; }
trap forward USR1 TERM
rc=0
while kill -0 "$TR" 2>/dev/null; do wait "$TR"; rc=$?; done
if [ -f "$OUTDIR/REQUEUE" ]; then echo "run_train.sh: stopped cleanly at $(cat "$OUTDIR/REQUEUE") images"; exit 3; fi
exit $rc
