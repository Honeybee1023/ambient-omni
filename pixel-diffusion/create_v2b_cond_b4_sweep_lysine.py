#!/usr/bin/env python3
"""Create v2b conditional sweep: B2 fixed at T=0.55, sweep B4.
Lysine-compatible paths. Only T values assigned to lysine.
"""
import os, json, glob
import numpy as np
from scipy.stats import norm

PROCESSED_DIR = "/data/honjar/celeba_processed_v2b/shared_buckets_64"
DATASET_DIR = "/data/honjar/annotated_datasets"
TVEC_DIR = "/data/honjar/generated"
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
    print("=== Cond B4 sweep (lysine portion) + missing cond B2 point ===")
    created = 0
    # Cond B4: T=0.7, 0.8, 0.9
    for t4 in [0.7, 0.8, 0.9]:
        t_int = round(t4 * 1000)
        suffix = f"T{t_int // 10:03d}"
        name = f"celeba_v2b_cond_b4_{suffix}"
        if create_dataset(name, {2: B2_FIXED_T, 4: t4}):
            created += 1
    # Missing cond B2 point: B1=0.5, B2=0.55
    if create_dataset("celeba_v2b_cond_b2_T055", {1: 0.5, 2: 0.55}):
        created += 1
    print(f"\n=== Done! Created {created} new datasets (4 expected) ===")

if __name__ == "__main__":
    main()
