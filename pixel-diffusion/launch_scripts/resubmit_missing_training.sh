#!/bin/bash

# Per-machine paths: see env.sh / SYNC.md at the repo root.
AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] && echo /data-local/honjar || echo /data/scratch/honjar)}"

# Resubmit all missing training jobs (7-cat main + extra + 2-domain v3)
COUNT=0

# 7-cat main (18 T values)
for cat in dog cat tiger lion fox leopard cheetah; do
    for suffix in 000 0125 025 0375 050 0625 075 080 085 090 091 092 093 094 095 097 099 100; do
        NAME="exp7d_${cat}_T${suffix}"
        CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            sbatch ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh $NAME
            COUNT=$((COUNT+1))
        fi
    done
done

# 7-cat extra (6 T values)
for cat in dog cat tiger lion fox leopard cheetah; do
    for suffix in 086 087 088 089 096 098; do
        NAME="exp7d_${cat}_T${suffix}"
        CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            sbatch ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh $NAME
            COUNT=$((COUNT+1))
        fi
    done
done

# 2-domain v3 extra (10 T values)
for cat in dog cat tiger lion fox leopard cheetah; do
    for suffix in 086 087 088 089 091 092 093 094 096 098; do
        NAME="pilot2d_${cat}_T${suffix}"
        CKPT=$(ls ${AMBIENT_BASE}/train_outputs/*-${NAME}-*/network-snapshot-001000.pkl 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            sbatch ${AMBIENT_BASE}/ambient-omni/pixel-diffusion/run_train_percat.sh $NAME
            COUNT=$((COUNT+1))
        fi
    done
done

echo "Submitted $COUNT training jobs"
