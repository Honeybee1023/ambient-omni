"""Per-machine paths for ambient-omni, resolved the same way env.sh resolves them.

Every machine lays the tree out identically under a single base directory, so
one variable is all that varies::

    $AMBIENT_BASE/
        ambient-omni/         <- this repo
        annotated_datasets/
        train_outputs/
        train_logs/
        generated/
        miniconda3/

Use this module in new code::

    from ambient_paths import GENERATED, DATASETS
    path = f"{GENERATED}/metrics_foo.json"

Older scripts inline the two-line resolver instead of importing this, because
they live at varying depths and importing would need sys.path surgery.  Both
read the same AMBIENT_BASE environment variable, so `source env.sh` controls
everything either way.
"""

import os

#: Candidate bases, tried in order when $AMBIENT_BASE is unset.
_CANDIDATES = ("/data-local/honjar", "/data/scratch/honjar")


def resolve_base():
    """Return the machine's ambient base directory.

    $AMBIENT_BASE wins if set; otherwise the first candidate that exists on
    disk is used.  Raises if neither applies, since silently guessing a path
    produces confusing downstream failures.
    """
    base = os.environ.get("AMBIENT_BASE")
    if base:
        return base
    for candidate in _CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "Could not detect AMBIENT_BASE on this machine. "
        "Set the AMBIENT_BASE environment variable, or source env.sh at the repo root."
    )


AMBIENT_BASE = resolve_base()

REPO = os.environ.get("AMBIENT_REPO") or os.path.join(AMBIENT_BASE, "ambient-omni")
DATASETS = os.environ.get("AMBIENT_DATASETS") or os.path.join(AMBIENT_BASE, "annotated_datasets")
OUTPUTS = os.environ.get("AMBIENT_OUTPUTS") or os.path.join(AMBIENT_BASE, "train_outputs")
LOGS = os.environ.get("AMBIENT_LOGS") or os.path.join(AMBIENT_BASE, "train_logs")
GENERATED = os.environ.get("AMBIENT_GENERATED") or os.path.join(AMBIENT_BASE, "generated")
PYTHON = os.environ.get("AMBIENT_PY") or os.path.join(
    AMBIENT_BASE, "miniconda3", "envs", "ambient", "bin", "python"
)
