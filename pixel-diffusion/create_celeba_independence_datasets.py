"""
Create multi-bucket datasets for CelebA independence test.

Three configs testing additivity of low-blur vs high-blur groups:
  all_tmid:     all 7 buckets at T_mid
  lowblur_tmid: buckets 1-3 at T_mid, 4-7 at T=1
  highblur_tmid: buckets 4-7 at T_mid, 1-3 at T=1

Independence prediction: FID(all) ≈ FID_baseline + Δ_low + Δ_high
where Δ_low = FID(lowblur) - FID_baseline
      Δ_high = FID(highblur) - FID_baseline
"""
import os, json, shutil
import numpy as np
from scipy.stats import norm

P_MEAN = -1.2
P_STD = 1.2

CELEBA_ROOT = "/data/scratch/honjar/celeba_processed"
SHARED_DIR = os.path.join(CELEBA_ROOT, "shared_buckets_64")
ANNOTATED_DIR = "/data/scratch/honjar/annotated_datasets"

BLUR_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 8.0}
TMID = {1: 0.416, 2: 0.561, 3: 0.803, 4: 0.878, 5: 0.902, 6: 0.913, 7: 0.928}


def t_to_sigma_min(t_value):
    if t_value <= 0.001:
        return 0.0
    if t_value >= 0.999:
        return float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))
    return float(np.exp(P_STD * norm.ppf(t_value) + P_MEAN))


def get_bucket_files(bucket_num):
    prefix = "b%d_" % bucket_num
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg')
                   and not f.startswith('._')])


CONFIGS = {
    "celeba_indep_all_tmid": {
        1: TMID[1], 2: TMID[2], 3: TMID[3],
        4: TMID[4], 5: TMID[5], 6: TMID[6], 7: TMID[7],
    },
    "celeba_indep_lowblur_tmid": {
        # Low blur (buckets 1-3) active at T_mid, high blur excluded
        1: TMID[1], 2: TMID[2], 3: TMID[3],
        4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
    },
    "celeba_indep_highblur_tmid": {
        # High blur (buckets 4-7) active at T_mid, low blur excluded
        1: 1.0, 2: 1.0, 3: 1.0,
        4: TMID[4], 5: TMID[5], 6: TMID[6], 7: TMID[7],
    },
}


def create_dataset(name, bucket_t_map, target_files, all_bucket_files):
    dataset_dir = os.path.join(ANNOTATED_DIR, name)

    ann_path = os.path.join(dataset_dir, "annotations.jsonl")
    if os.path.exists(ann_path):
        print("  SKIP (exists): %s" % name)
        return

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    annotations = []

    for fname in target_files:
        src = os.path.join(SHARED_DIR, fname)
        os.symlink(src, os.path.join(dataset_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    for b in sorted(bucket_t_map.keys()):
        t_val = bucket_t_map[b]
        sigma_min = 0.0 if t_val <= 0.001 else t_to_sigma_min(t_val)
        for fname in all_bucket_files[b]:
            src = os.path.join(SHARED_DIR, fname)
            os.symlink(src, os.path.join(dataset_dir, fname))
            annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    print("  CREATED: %s (%d images)" % (name, len(annotations)))
    for b in sorted(bucket_t_map.keys()):
        t_val = bucket_t_map[b]
        sigma_min = 0.0 if t_val <= 0.001 else t_to_sigma_min(t_val)
        active = "ACTIVE" if t_val < 0.999 else "excluded"
        print("    Bucket %d (blur=%.1f): T=%.3f sigma_min=%.4f [%s]" % (
            b, BLUR_SIGMAS[b], t_val, sigma_min, active))


def main():
    print("=" * 60)
    print("CelebA Independence Test Dataset Creation")
    print("=" * 60)

    target_files = get_bucket_files(0)
    print("Target (bucket 0, clean): %d images" % len(target_files))

    all_bucket_files = {}
    for b in sorted(BLUR_SIGMAS.keys()):
        all_bucket_files[b] = get_bucket_files(b)
        print("Bucket %d (sigma_blur=%.1f): %d images" % (b, BLUR_SIGMAS[b], len(all_bucket_files[b])))

    print("\n--- Creating independence test datasets ---")
    for name, config in CONFIGS.items():
        create_dataset(name, config, target_files, all_bucket_files)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
