#!/bin/bash
SCRIPT=/data/scratch/honjar/ambient-omni/pixel-diffusion/run_percat_gen_eval.sh
READY=0

for cat in dog cat fox tiger lion leopard cheetah; do
    for suffix in 000 0125 025 0375 050 0625 075 080 085 090 091 092 093 094 095 097 099 100; do
        NAME="exp7d_${cat}_T${suffix}"
        JSON="/data/scratch/honjar/generated/metrics_${NAME}_1000kimg.json"
        CKPT=$(ls /data/scratch/honjar/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -n "$CKPT" ] && [ ! -f "$JSON" ]; then
            echo "Submitting: $NAME"
            sbatch $SCRIPT $NAME
            READY=$((READY+1))
        fi
    done
done
echo "Submitted $READY gen+eval jobs"
