#!/bin/bash
# Refresh the data listings in inventory/ for whichever machine this runs on.
# See MACHINES.md.  Run it on each cluster, then commit the result.

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # repo root
. ./env.sh || exit 1

case "$AMBIENT_BASE" in
    /data-local/honjar)    TAG=lysine ;;
    /data/scratch/honjar)  TAG=csail ;;
    *)                     TAG=$(hostname | tr -cd '[:alnum:]') ;;
esac

mkdir -p inventory
for d in annotated_datasets train_outputs generated; do
    out="inventory/${TAG}_${d}.txt"
    if [ -d "$AMBIENT_BASE/$d" ]; then
        ls "$AMBIENT_BASE/$d" | sort > "$out"
        printf '%-8s %-20s %s entries\n' "$TAG" "$d" "$(wc -l < "$out" | tr -d ' ')"
    else
        printf '%-8s %-20s missing, skipped\n' "$TAG" "$d"
    fi
done

echo
echo "Listings written for $TAG. Commit them so the other machines can see them:"
echo "    ./sync.sh push \"Refresh $TAG data inventory\""
