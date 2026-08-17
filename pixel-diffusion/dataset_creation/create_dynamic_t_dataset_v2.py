"""Create the v2 dynamic-T dataset (lysine paths).

Design mirrors the existing *static* b5 sweep datasets exactly, so that dynamic
runs are directly comparable to `celeba_v2b_b5_T*`:

    b0  (500 clean)      -> sigma_min = 0.0        (usable at every noise level)
    b5  (26,014 blurred) -> sigma_min = 999.0      (sentinel: "use the schedule")
    all other buckets    -> sigma_min = 12.2838    (T~=1.0, effectively off)

The 999.0 sentinel is replaced every iteration by `compute_scheduled_sigma_min`
in training/training_loop.py. Note the training loop now refuses to run if the
sentinel is present without a --t_schedule, so this dataset can only be used for
dynamic runs.

Also emits a clean-only variant (500 b0 images) for the floor baseline.

Usage:
    python dataset_creation/create_dynamic_t_dataset_v2.py
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)


import argparse
import json
import os
import shutil

import numpy as np
from scipy.stats import norm

SRC = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
OUT_ROOT = f"{AMBIENT_BASE}/annotated_datasets"

TARGET_BUCKET = "b5"      # sigma_blur = 0.5
CLEAN_BUCKET = "b0"
SENTINEL = 999.0

# EDM noise schedule: sigma = exp(P_STD * Phi^-1(T) + P_MEAN)
P_MEAN, P_STD = -1.2, 1.2


def t_to_sigma(t):
    return float(np.exp(P_STD * norm.ppf(np.clip(t, 0.001, 0.999)) + P_MEAN))


# The value the existing static sweeps use to park an unused bucket (T=0.999).
OFF_SIGMA = t_to_sigma(0.999)


def bucket_of(fname):
    return fname.split("_")[0]


def build(out_dir, include_corrupt=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    files = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    annotations = []
    counts = {}

    for fname in files:
        b = bucket_of(fname)

        if b == CLEAN_BUCKET:
            sigma_min = 0.0
        elif b == TARGET_BUCKET:
            if not include_corrupt:
                continue
            sigma_min = SENTINEL
        else:
            if not include_corrupt:
                continue
            sigma_min = OFF_SIGMA

        os.symlink(os.path.join(SRC, fname), os.path.join(out_dir, fname))
        annotations.append(
            {"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0}
        )
        counts[b] = counts.get(b, 0) + 1

    with open(os.path.join(out_dir, "annotations.jsonl"), "w") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")

    print(f"\n{out_dir}")
    for b in sorted(counts):
        tag = {
            CLEAN_BUCKET: "clean, sigma_min=0",
            TARGET_BUCKET: f"TARGET, sigma_min={SENTINEL} (schedule)",
        }.get(b, f"parked, sigma_min={OFF_SIGMA:.4f}")
        print(f"  {b}: {counts[b]:>6} images  [{tag}]")
    print(f"  total: {len(annotations)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="celeba_dynamic_t_v2")
    args = ap.parse_args()

    print(f"T=0.999 -> sigma_min={OFF_SIGMA:.4f} (used to park unused buckets)")
    build(os.path.join(OUT_ROOT, args.name), include_corrupt=True)
    build(os.path.join(OUT_ROOT, args.name + "_cleanonly"), include_corrupt=False)


if __name__ == "__main__":
    main()
