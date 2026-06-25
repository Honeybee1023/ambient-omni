import os, json, shutil, sys
import numpy as np
from scipy.stats import norm

P_MEAN = -1.2
P_STD = 1.2

CELEBA_V2_ROOT = "/data/honjar/celeba_processed_v2b"
SHARED_DIR = os.path.join(CELEBA_V2_ROOT, "shared_buckets_64")
ANNOTATED_DIR = "/data/honjar/annotated_datasets"
N_BUCKETS = 8
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
                   if f.startswith(prefix) and f.endswith(".jpg")
                   and not f.startswith("._")])

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
            active.append("B%d=%.3f" % (b, t_vector[b]))
    active_str = ", ".join(active) if active else "none (=baseline)"
    print("  CREATED: %s (%d images, active: %s)" % (name, n_linked, active_str))
    return False

def main():
    print("CelebA v2b Fine Sweep + T-Convergence Datasets")
    if not os.path.exists(SHARED_DIR):
        print("ERROR: %s not found." % SHARED_DIR)
        sys.exit(1)
    all_bucket_files = {}
    total_images = 0
    for b in range(N_BUCKETS):
        all_bucket_files[b] = get_bucket_files(b)
        total_images += len(all_bucket_files[b])
        print("  Bucket %d: %d images" % (b, len(all_bucket_files[b])))
    print("  Total per dataset: %d images" % total_images)
    fine_sweep = {1: [0.3, 0.35, 0.45, 0.5], 2: [0.3, 0.35, 0.45, 0.5], 3: [0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7], 4: [0.5, 0.55, 0.65, 0.7], 5: [0.5, 0.55, 0.65, 0.7]}
    t_convergence = {1: [0.99, 1.0], 5: [0.99, 1.0]}
    created = 0
    skipped = 0
    for b, t_values in sorted(fine_sweep.items()):
        for t_val in t_values:
            suffix = t_to_suffix(t_val)
            name = "celeba_v2b_b%d_T%s" % (b, suffix)
            t_vector = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
            t_vector[b] = t_val
            was_skipped = create_dataset(name, t_vector, all_bucket_files)
            if was_skipped: skipped += 1
            else: created += 1
    for b, t_values in sorted(t_convergence.items()):
        for t_val in t_values:
            suffix = t_to_suffix(t_val)
            name = "celeba_v2b_b%d_T%s" % (b, suffix)
            t_vector = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
            t_vector[b] = t_val
            was_skipped = create_dataset(name, t_vector, all_bucket_files)
            if was_skipped: skipped += 1
            else: created += 1
    print("Created: %d new, Skipped: %d existing" % (created, skipped))

if __name__ == "__main__":
    main()