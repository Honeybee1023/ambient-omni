#!/usr/bin/env python3
"""Create v2b conditional sweep datasets on lysine: B1 fixed at T=0.5, sweep B2.
Extends the CSAIL sweep (0.4-0.7) to the tails: 0, 0.2, 0.3, 0.8, 0.9, 0.95, 0.99, 1.0
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, sys, json, glob
import numpy as np
from scipy.stats import norm

PROCESSED_DIR = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
DATASET_DIR = f"{AMBIENT_BASE}/annotated_datasets"
TVEC_DIR = f"{AMBIENT_BASE}/generated"
ALL_BLUR_BUCKETS = [1, 2, 3, 4, 5, 6, 7]
INACTIVE_T = 0.999

def t_to_sigma_min(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        t = INACTIVE_T
    return float(np.exp(-1.2 + 1.2 * norm.ppf(t)))

def create_dataset(name, bucket_t_map):
    ds_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(ds_dir):
        print(f"  SKIP {name} (already exists)")
        return False
    os.makedirs(ds_dir, exist_ok=True)
    annotations = []

    # Clean images (bucket 0)
    clean_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "b0_*.jpg")))
    for src in clean_files:
        fname = os.path.basename(src)
        os.symlink(src, os.path.join(ds_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    # All blur buckets
    for bucket in ALL_BLUR_BUCKETS:
        t_val = bucket_t_map.get(bucket, INACTIVE_T)
        smin = t_to_sigma_min(t_val)
        bucket_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, f"b{bucket}_*.jpg")))
        for src in bucket_files:
            fname = os.path.basename(src)
            os.symlink(src, os.path.join(ds_dir, fname))
            annotations.append({"filename": fname, "sigma_min": smin, "sigma_max": 0.0})

    annotations.sort(key=lambda x: x["filename"])
    with open(os.path.join(ds_dir, "annotations.jsonl"), "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    # T-vector sidecar
    tvec = {f"B{b}": bucket_t_map.get(b, INACTIVE_T) for b in ALL_BLUR_BUCKETS}
    with open(os.path.join(TVEC_DIR, f"tvec_{name}.json"), "w") as f:
        json.dump(tvec, f, indent=2)

    active_str = ", ".join(f"B{b}={bucket_t_map[b]}" for b in sorted(bucket_t_map))
    print(f"  Created {name}: {len(annotations)} images, active: {active_str}")
    return True

def main():
    print("=== Creating cond B2 sweep datasets (lysine) ===")
    created = 0
    sweep_ts = [0.0, 0.2, 0.3, 0.8, 0.9, 0.95, 0.99, 1.0]
    for t2 in sweep_ts:
        t_int = round(t2 * 1000)
        if t_int % 10 == 0:
            suffix = f"T{t_int // 10:03d}"
        else:
            suffix = f"T{t_int:04d}"
        name = f"celeba_v2b_cond_b2_{suffix}"
        if create_dataset(name, {1: 0.5, 2: t2}):
            created += 1
    print(f"\n=== Done! Created {created} new datasets (8 expected) ===")

if __name__ == "__main__":
    main()
