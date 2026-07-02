#!/usr/bin/env python3
"""Create refinement datasets for B3/B4/B5 conditional sweeps.
Proline-compatible paths.
"""
import os, json, glob
import numpy as np
from scipy.stats import norm

PROCESSED_DIR = "/var/local/honjar/celeba_processed_v2b/shared_buckets_64"
DATASET_DIR = "/var/local/honjar/annotated_datasets"
TVEC_DIR = "/var/local/honjar/generated"
ALL_BLUR_BUCKETS = [1, 2, 3, 4, 5, 6, 7]
INACTIVE_T = 0.999
B2_FIXED_T = 0.55

def t_to_sigma_min(t):
    if t <= 0.0: return 0.0
    if t >= 1.0: t = INACTIVE_T
    return float(np.exp(-1.2 + 1.2 * norm.ppf(t)))

def create_dataset(name, bucket_t_map):
    ds_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(ds_dir):
        print(f"  SKIP {name} (already exists)")
        return False
    os.makedirs(ds_dir, exist_ok=True)
    annotations = []
    clean_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "b0_*.jpg")))
    for src in clean_files:
        fname = os.path.basename(src)
        os.symlink(src, os.path.join(ds_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})
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
    tvec = {f"B{b}": bucket_t_map.get(b, INACTIVE_T) for b in ALL_BLUR_BUCKETS}
    with open(os.path.join(TVEC_DIR, f"tvec_{name}.json"), "w") as f:
        json.dump(tvec, f, indent=2)
    active_str = ", ".join(f"B{b}={bucket_t_map[b]}" for b in sorted(bucket_t_map))
    print(f"  Created {name}: {len(annotations)} images, active: {active_str}")
    return True

def main():
    print("=== Refinement sweep: B3/B4/B5 fill-in + reruns (proline) ===")
    created = 0

    # B3 new points: T=0.525, 0.575, 0.7
    print("\n--- B3 new points (B2=0.55 fixed) ---")
    for t3 in [0.525, 0.575, 0.7]:
        t_int = round(t3 * 1000)
        if t_int % 10 == 0:
            suffix = f"T{t_int // 10:03d}"
        else:
            suffix = f"T{t_int:04d}"
        name = f"celeba_v2b_cond_b3_{suffix}"
        if create_dataset(name, {2: B2_FIXED_T, 3: t3}):
            created += 1

    # B3 rerun: T=0.55
    print("\n--- B3 T=0.55 rerun ---")
    if create_dataset("celeba_v2b_cond_b3_T055_r2", {2: B2_FIXED_T, 3: 0.55}):
        created += 1

    # B4 new points: T=0.75, 0.85
    print("\n--- B4 new points (B2=0.55 fixed) ---")
    for t4 in [0.75, 0.85]:
        t_int = round(t4 * 1000)
        suffix = f"T{t_int // 10:03d}"
        name = f"celeba_v2b_cond_b4_{suffix}"
        if create_dataset(name, {2: B2_FIXED_T, 4: t4}):
            created += 1

    # B4 rerun: T=0.8
    print("\n--- B4 T=0.8 rerun ---")
    if create_dataset("celeba_v2b_cond_b4_T080_r2", {2: B2_FIXED_T, 4: 0.8}):
        created += 1

    # B5 new point: T=0.99
    print("\n--- B5 T=0.99 (B2=0.55 fixed) ---")
    if create_dataset("celeba_v2b_cond_b5_T099", {2: B2_FIXED_T, 5: 0.99}):
        created += 1

    print(f"\n=== Done! Created {created} new datasets (8 expected) ===")

if __name__ == "__main__":
    main()
