# Per-machine paths for ambient-omni.  Source this before running anything:
#
#     source /path/to/ambient-omni/env.sh
#
# Every machine lays the tree out the same way, so one variable is enough:
#
#     $AMBIENT_BASE/
#         ambient-omni/         <- this repo
#         annotated_datasets/
#         train_outputs/
#         train_logs/
#         generated/
#         miniconda3/
#
# The only thing that differs per machine is where that base lives, so it is
# detected below.  Nothing here is machine-specific on disk: to override, drop
# an env.local.sh next to this file (it is gitignored) and set AMBIENT_BASE.

if [ -z "${AMBIENT_BASE:-}" ]; then
    for _candidate in /data-local/honjar /data/scratch/honjar; do
        if [ -d "$_candidate" ]; then
            AMBIENT_BASE="$_candidate"
            break
        fi
    done
    unset _candidate
fi

# Local override wins over detection, so a laptop or a new machine needs no code
# change -- only an untracked env.local.sh.
if [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local.sh" ]; then
    . "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local.sh"
fi

if [ -z "${AMBIENT_BASE:-}" ]; then
    echo "env.sh: could not detect AMBIENT_BASE on $(hostname)." >&2
    echo "        Create env.local.sh next to env.sh with: AMBIENT_BASE=/your/path" >&2
    return 1 2>/dev/null || exit 1
fi

export AMBIENT_BASE
export AMBIENT_REPO="${AMBIENT_REPO:-$AMBIENT_BASE/ambient-omni}"
export AMBIENT_DATASETS="${AMBIENT_DATASETS:-$AMBIENT_BASE/annotated_datasets}"
export AMBIENT_OUTPUTS="${AMBIENT_OUTPUTS:-$AMBIENT_BASE/train_outputs}"
export AMBIENT_LOGS="${AMBIENT_LOGS:-$AMBIENT_BASE/train_logs}"
export AMBIENT_GENERATED="${AMBIENT_GENERATED:-$AMBIENT_BASE/generated}"
export AMBIENT_PY="${AMBIENT_PY:-$AMBIENT_BASE/miniconda3/envs/ambient/bin/python}"
