#!/usr/bin/env python3
"""Create v2b independence test datasets.

Each dataset has: clean (b0) + B1 at specified T + B2 at specified T.
Annotations sorted by filename for reproducibility (avoids symlink-ordering noise).

Test 1 (pairwise best): B1=0.5, B2=0.55 — absorbed into Test 3 as cond_b1_T050
Test 2 (shifts ±0.1):   4 configs varying both B1 and B2
Test 3 (conditional):   B2 fixed at 0.55, sweep B1 across 9 points
"""

import os
import json
import glob
import numpy as np
from scipy.stats import norm

PROCESSED_DIR = "/data/scratch/honjar/celeba_processed_v2b/shared_buckets_64"
DATASET_DIR = "/data/scratch/honjar/annotated_datasets"
TVEC_DIR = "/data/scratch/honjar/generated"


def t_to_sigma_min(t):
    """Convert noise threshold T to sigma_min for annotations."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return float("inf")
    return float(np.exp(-1.2 + 1.2 * norm.ppf(t)))


def create_dataset(name, bucket_t_map):
    """Create a multi-bucket dataset.

    Args:
        name: dataset directory name
        bucket_t_map: {bucket_number: T_value} for active buckets (1-7).
                      Clean (bucket 0) always included with sigma_min=0.
    """
    ds_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(ds_dir):
        print(f"  SKIP {name} (already exists)")
        return False

    os.makedirs(ds_dir, exist_ok=True)

    annotations = []

    # Clean images (bucket 0) — always sigma_min=0
    clean_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "b0_*.png")))
    for src in clean_files:
        fname = os.path.basename(src)
        os.symlink(src, os.path.join(ds_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    # Active blur buckets
    for bucket, t_val in sorted(bucket_t_map.items()):
        smin = t_to_sigma_min(t_val)
        bucket_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, f"b{bucket}_*.png")))
        for src in bucket_files:
            fname = os.path.basename(src)
            os.symlink(src, os.path.join(ds_dir, fname))
            annotations.append({"filename": fname, "sigma_min": smin, "sigma_max": 0.0})

    # Sort by filename for reproducibility
    annotations.sort(key=lambda x: x["filename"])

    with open(os.path.join(ds_dir, "annotations.jsonl"), "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    # Save T-vector sidecar for later analysis
    tvec_path = os.path.join(TVEC_DIR, f"tvec_{name}.json")
    tvec = {f"B{b}": t for b, t in sorted(bucket_t_map.items())}
    with open(tvec_path, "w") as f:
        json.dump(tvec, f, indent=2)

    print(f"  Created {name}: {len(annotations)} images, config={tvec}")
    return True


def main():
    print("=== Creating v2b independence test datasets ===")
    print(f"Processed images: {PROCESSED_DIR}")
    print(f"Output dir: {DATASET_DIR}")
    print()

    created = 0

    # --- Test 3: Conditional 1D sweep (B2=0.55 fixed, sweep B1) ---
    # Test 1 (pairwise best) = the T050 point (B1=0.5, B2=0.55)
    print("Test 3 (conditional sweep, B2=0.55 fixed, sweep B1):")
    print("  [Test 1 = cond_b1_T050 = B1@0.5 + B2@0.55]")
    cond_t_values = [0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 0.95]
    for t1 in cond_t_values:
        t_int = round(t1 * 1000)
        if t_int % 10 == 0:
            suffix = f"T{t_int // 10:03d}"       # 3-digit: T000, T050, etc.
        else:
            suffix = f"T{t_int:04d}"              # 4-digit: T0525, etc.
        name = f"celeba_v2b_cond_b1_{suffix}"
        if create_dataset(name, {1: t1, 2: 0.55}):
            created += 1

    # --- Test 2: Combinatorial shifts (±0.1 from best) ---
    # B1 best=0.5, B2 best=0.55
    print()
    print("Test 2 (shifts):")
    shift_configs = [
        ("bothup", {1: 0.6,   2: 0.65}),    # both shifted right +0.1
        ("bothdn", {1: 0.4,   2: 0.45}),     # both shifted left  -0.1
        ("apart",  {1: 0.4,   2: 0.65}),     # pushed further apart
        ("close",  {1: 0.525, 2: 0.525}),    # both at midpoint
    ]
    for label, config in shift_configs:
        name = f"celeba_v2b_shift_{label}"
        if create_dataset(name, config):
            created += 1

    print()
    print(f"=== Done! Created {created} new datasets (13 total expected) ===")


if __name__ == "__main__":
    main()
