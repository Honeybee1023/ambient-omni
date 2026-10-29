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
# Optional, from the environment:
#   STAGE_DIR=<node-local dir>  copy the --data file there first and train from the copy. Random reads from a large
#                               zip on a shared filesystem can starve the GPUs (Lustre: ~5 ms per image; local: ~0.8 ms).
#   DATA_WORKERS=<n>            data loader processes per GPU (train_edm2.py --workers; default 2).
if [ -n "${STAGE_DIR:-}" ]; then
  args=()
  for a in "$@"; do
    case $a in
      --data=*)
        src=${a#--data=}; dst=$STAGE_DIR/$(basename "$src"); mkdir -p "$STAGE_DIR"
        need=$(stat -c %s "$src"); free=$(( $(df -B1 --output=avail "$STAGE_DIR" | tail -1) ))
        if [ "$(stat -c %s "$dst" 2>/dev/null)" = "$need" ]; then
          a=--data=$dst
        elif [ "$free" -gt $(( need + (20 << 30) )) ]; then
          echo "run_train.sh: $(hostname -s) copying $src to $dst ($(date +%T))"
          cp "$src" "$dst.part.$$" && mv "$dst.part.$$" "$dst" || { echo "run_train.sh: copy failed"; rm -f "$dst.part.$$"; exit 2; }
          echo "run_train.sh: $(hostname -s) copy done ($(date +%T))"
          a=--data=$dst
        else
          echo "run_train.sh: not enough space in $STAGE_DIR; reading $src in place"
        fi ;;
    esac
    args+=("$a")
  done
  set -- "${args[@]}"
fi
[ -n "${DATA_WORKERS:-}" ] && set -- "$@" --workers="$DATA_WORKERS"
if [ "${NNODES:-1}" -gt 1 ]; then
  # One copy of this script per node (scripts/launch_slurm.sh starts them), joined by a fixed node list (static
  # rendezvous). Not c10d: its agents keep checking in with node 0 and give up after 60 s without a reply, which
  # killed a healthy run after 3 h (one late reply, then the other node hung in NCCL for 10 min).
  torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC" \
           --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" train_edm2.py "$@" &
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
