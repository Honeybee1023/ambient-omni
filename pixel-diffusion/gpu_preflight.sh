#!/bin/bash
# Source from a Slurm job script before training: if any GPU we were given already has another process's memory
# on it (seen on node4400: a 129 GB process on "our" GPU -> OOM within a minute), put the job back in the queue
# with this node excluded instead of crashing. Requires the job to be submitted with --requeue.
gpu_preflight() {
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$2 > 2000 {print $1":"$2"MiB"}' | tr '\n' ' ')
    if [ -n "$busy" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        local node; node=$(hostname -s)
        local exc; exc=$(scontrol show job "$SLURM_JOB_ID" -o | grep -o 'ExcNodeList=[^ ]*' | sed 's/ExcNodeList=//;s/(null)//')
        echo "PREFLIGHT: GPUs already in use on $node ($busy); requeueing with $node excluded"
        # A running job's ExcNodeList cannot be changed and requeueing kills this script, so a tiny CPU job
        # outside this allocation adds the exclusion once the job is back in the queue (held) and releases it.
        sbatch -p mit_normal -c 1 --mem=1G -t 00:10:00 -J "preflight_fix_$SLURM_JOB_ID" -o /dev/null --wrap \
            "sleep 30; scontrol update jobid=$SLURM_JOB_ID ExcNodeList=${exc:+$exc,}$node; scontrol release $SLURM_JOB_ID"
        scontrol requeuehold "$SLURM_JOB_ID"
        sleep 60
        exit 1
    fi
}
