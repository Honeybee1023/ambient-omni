"""
Create annotated datasets for CelebA v2 (data-poor regime) 2D sweep.

Each dataset: ALL 8 buckets present in full-vector format.
  - Bucket 0 (clean, 500 images): always sigma_min=0 (always used)
  - One active blur bucket at specified T value
  - All other blur buckets at T=1 (sigma_min~12.28, effectively off)

This ensures 2D and multi-bucket experiments use identical dataset format,
eliminating the format confound from the previous round.

Uses symlinks to celeba_processed_v2/shared_buckets_64/.

Coarse T: {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95} x 7 blur buckets = 49 datasets
Plus 1 baseline (all blur T=1) = 50 datasets total

Naming: celeba_v2_b{bucket}_T{suffix}  (sweep datasets)
        celeba_v2_baseline              (all blur at T=1)
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, shutil, sys
import numpy as np
from scipy.stats import norm

P_MEAN = -1.2
P_STD = 1.2

CELEBA_V2_ROOT = f"{AMBIENT_BASE}/celeba_processed_v2"
SHARED_DIR = os.path.join(CELEBA_V2_ROOT, "shared_buckets_64")
ANNOTATED_DIR = f"{AMBIENT_BASE}/annotated_datasets"

N_BUCKETS = 8

BLUR_SIGMAS = {
    0: 0.0,
    1: 0.1,
    2: 0.3,
    3: 0.5,
    4: 0.75,
    5: 1.0,
    6: 1.5,
    7: 2.0,
}

COARSE_T = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

T_OFF_SIGMA_MIN = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))


def t_to_sigma_min(t_value):
    if t_value == 0:
        return 0.0
    t_clipped = np.clip(t_value, 0.001, 0.999)
    return float(np.exp(P_STD * norm.ppf(t_clipped) + P_MEAN))


def t_to_suffix(t_val):
    t_milli = round(t_val * 1000)
    if t_milli % 10 == 0:
        return "%03d" % (t_milli // 10)
    else:
        return "%04d" % t_milli


def get_bucket_files(bucket_num):
    prefix = "b%d_" % bucket_num
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg')
                   and not f.startswith('._')])


def create_dataset(name, t_vector, all_bucket_files):
    dataset_dir = os.path.join(ANNOTATED_DIR, name)
    ann_path = os.path.join(dataset_dir, "annotations.jsonl")

    if os.path.exists(ann_path):
        print("  SKIP (exists): %s" % name)
        return True

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    annotations = []
    n_linked = 0

    for bucket in range(N_BUCKETS):
        t_val = t_vector[bucket]
        if t_val == 0:
            sigma_min = 0.0
        elif t_val >= 1.0:
            sigma_min = T_OFF_SIGMA_MIN
        else:
            sigma_min = t_to_sigma_min(t_val)

        for fname in all_bucket_files[bucket]:
            src = os.path.join(SHARED_DIR, fname)
            dst = os.path.join(dataset_dir, fname)
            os.symlink(src, dst)
            annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})
            n_linked += 1

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    active = []
    for b in range(1, N_BUCKETS):
        if t_vector[b] < 1.0:
            active.append("B%d=%.2f" % (b, t_vector[b]))
    active_str = ", ".join(active) if active else "none (baseline)"

    print("  CREATED: %s (%d images, active: %s)" % (name, n_linked, active_str))
    return False


def main():
    print("=" * 60)
    print("CelebA v2 — Full-Vector 2D Sweep Dataset Creation")
    print("=" * 60)

    if not os.path.exists(SHARED_DIR):
        print("ERROR: %s not found. Run prepare_celeba_v2.py first." % SHARED_DIR)
        sys.exit(1)

    all_bucket_files = {}
    total_images = 0
    for b in range(N_BUCKETS):
        all_bucket_files[b] = get_bucket_files(b)
        count = len(all_bucket_files[b])
        total_images += count
        sigma = BLUR_SIGMAS[b]
        label = "clean" if b == 0 else "sigma=%.2f" % sigma
        print("  Bucket %d (%s): %d images" % (b, label, count))
    print("  Total per dataset: %d images" % total_images)

    created = 0
    skipped = 0

    # Baseline: all blur buckets at T=1
    print("\n--- Baseline (all T=1) ---")
    baseline_tvec = {b: (0.0 if b == 0 else 1.0) for b in range(N_BUCKETS)}
    was_skipped = create_dataset("celeba_v2_baseline", baseline_tvec, all_bucket_files)
    if was_skipped:
        skipped += 1
    else:
        created += 1

    # Coarse 2D sweep
    print("\n--- Coarse 2D sweep: %d T values x %d buckets ---" % (
        len(COARSE_T), N_BUCKETS - 1))

    for b in range(1, N_BUCKETS):
        sigma = BLUR_SIGMAS[b]
        print("\n  Bucket %d (sigma=%.2f):" % (b, sigma))
        for t_val in COARSE_T:
            suffix = t_to_suffix(t_val)
            name = "celeba_v2_b%d_T%s" % (b, suffix)

            t_vector = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
            t_vector[b] = t_val

            was_skipped = create_dataset(name, t_vector, all_bucket_files)
            if was_skipped:
                skipped += 1
            else:
                created += 1

    # Save manifest
    manifest = {
        "version": "v2_coarse",
        "format": "full_8bucket",
        "n_clean": len(all_bucket_files[0]),
        "blur_sigmas": {str(b): float(s) for b, s in BLUR_SIGMAS.items()},
        "coarse_t_values": COARSE_T,
        "t_off_sigma_min": T_OFF_SIGMA_MIN,
        "datasets": {},
    }

    manifest["datasets"]["celeba_v2_baseline"] = {
        "t_vector": [0.0] + [1.0] * 7,
        "active_buckets": [],
    }

    for b in range(1, N_BUCKETS):
        for t_val in COARSE_T:
            suffix = t_to_suffix(t_val)
            name = "celeba_v2_b%d_T%s" % (b, suffix)
            tvec = [0.0] + [1.0] * 7
            tvec[b] = t_val
            manifest["datasets"][name] = {
                "t_vector": tvec,
                "active_buckets": [b],
                "active_t": t_val,
                "blur_sigma": BLUR_SIGMAS[b],
            }

    manifest_path = os.path.join(ANNOTATED_DIR, "celeba_v2_coarse_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Summary ===")
    print("  Created: %d new datasets" % created)
    print("  Skipped: %d existing datasets" % skipped)
    print("  Images per dataset: %d" % total_images)
    print("  Manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
