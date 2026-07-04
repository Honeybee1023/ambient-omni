"""
Create 2-domain annotated datasets for CelebA blur experiments.

Each dataset: bucket 0 (clean target, T=0) + one blur bucket at specified T.
Uses symlinks to celeba_processed/shared_buckets_64/.

Coarse: T in {0, 0.25, 0.5, 0.75, 1.0} x 7 blur levels = 35 datasets
Fine:   T in {0.125, 0.375, 0.625, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99} x 7 = 63 datasets
Total: 98 datasets

Naming: celeba_2d_b{bucket}_T{suffix}
"""
import os, json, shutil
import numpy as np
from scipy.stats import norm

P_MEAN = -1.2
P_STD = 1.2

CELEBA_ROOT = "/data/scratch/honjar/celeba_processed"
SHARED_DIR = os.path.join(CELEBA_ROOT, "shared_buckets_64")
ANNOTATED_DIR = "/data/scratch/honjar/annotated_datasets"

BLUR_BUCKETS = {
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 3.0,
    5: 4.0,
    6: 5.0,
    7: 8.0,
}

COARSE_T = [0.0, 0.25, 0.5, 0.75, 1.0]
FINE_T = [0.125, 0.375, 0.625, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


def t_to_sigma_min(t_value):
    t_clipped = np.clip(t_value, 0.001, 0.999)
    return float(np.exp(P_STD * norm.ppf(t_clipped) + P_MEAN))


def t_to_suffix(t_val):
    t_milli = round(t_val * 1000)
    if t_milli % 10 == 0:
        return '%03d' % (t_milli // 10)
    else:
        return '%04d' % t_milli


def get_bucket_files(bucket_num):
    prefix = "b%d_" % bucket_num
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg')
                   and not f.startswith('._')])


def create_dataset(bucket_num, t_val, target_files, bucket_files):
    suffix = t_to_suffix(t_val)
    name = "celeba_2d_b%d_T%s" % (bucket_num, suffix)
    dataset_dir = os.path.join(ANNOTATED_DIR, name)

    ann_path = os.path.join(dataset_dir, "annotations.jsonl")
    if os.path.exists(ann_path):
        print("  SKIP (exists): %s" % name)
        return name, True

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    sigma_min = 0.0 if t_val == 0 else t_to_sigma_min(t_val)
    annotations = []

    for fname in target_files:
        src = os.path.join(SHARED_DIR, fname)
        os.symlink(src, os.path.join(dataset_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    for fname in bucket_files:
        src = os.path.join(SHARED_DIR, fname)
        os.symlink(src, os.path.join(dataset_dir, fname))
        annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    print("  CREATED: %s (%d images, T=%.3f, sigma_min=%.4f)" % (
        name, len(annotations), t_val, sigma_min))
    return name, False


def main():
    print("=" * 60)
    print("CelebA 2-Domain Dataset Creation")
    print("=" * 60)

    target_files = get_bucket_files(0)
    print("Target (bucket 0, clean): %d images" % len(target_files))

    bucket_files = {}
    for b in sorted(BLUR_BUCKETS.keys()):
        bucket_files[b] = get_bucket_files(b)
        print("Bucket %d (sigma_blur=%.1f): %d images" % (b, BLUR_BUCKETS[b], len(bucket_files[b])))

    created = 0
    skipped = 0

    print("\n--- Coarse sweep: %d T values x %d buckets ---" % (len(COARSE_T), len(BLUR_BUCKETS)))
    for b in sorted(BLUR_BUCKETS.keys()):
        for t_val in COARSE_T:
            _, was_skipped = create_dataset(b, t_val, target_files, bucket_files[b])
            if was_skipped:
                skipped += 1
            else:
                created += 1

    print("\n--- Fine sweep: %d T values x %d buckets ---" % (len(FINE_T), len(BLUR_BUCKETS)))
    for b in sorted(BLUR_BUCKETS.keys()):
        for t_val in FINE_T:
            _, was_skipped = create_dataset(b, t_val, target_files, bucket_files[b])
            if was_skipped:
                skipped += 1
            else:
                created += 1

    print("\n=== Summary ===")
    print("  Created: %d new datasets" % created)
    print("  Skipped: %d existing datasets" % skipped)
    print("  Images per dataset: ~%d (target + one bucket)" % (len(target_files) + len(bucket_files[1])))


if __name__ == "__main__":
    main()
