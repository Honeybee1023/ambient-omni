"""Create annotated datasets for v2b (mild blur 0.1-0.7) coarse 2D sweep."""
import os, json, shutil, sys
import numpy as np
from scipy.stats import norm

P_MEAN = -1.2
P_STD = 1.2

SHARED_DIR = "/data/honjar/celeba_processed_v2b/shared_buckets_64"
ANNOTATED_DIR = "/data/honjar/annotated_datasets"
N_BUCKETS = 8
BLUR_SIGMAS = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7}
COARSE_T = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
T_OFF_SIGMA_MIN = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))

def t_to_sigma_min(t_value):
    if t_value == 0: return 0.0
    return float(np.exp(P_STD * norm.ppf(np.clip(t_value, 0.001, 0.999)) + P_MEAN))

def t_to_suffix(t_val):
    t_milli = round(t_val * 1000)
    return "%03d" % (t_milli // 10) if t_milli % 10 == 0 else "%04d" % t_milli

def get_bucket_files(bucket_num):
    prefix = "b%d_" % bucket_num
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg') and not f.startswith('._')])

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
    for bucket in range(N_BUCKETS):
        t_val = t_vector[bucket]
        sigma_min = 0.0 if t_val == 0 else (T_OFF_SIGMA_MIN if t_val >= 1.0 else t_to_sigma_min(t_val))
        for fname in all_bucket_files[bucket]:
            os.symlink(os.path.join(SHARED_DIR, fname), os.path.join(dataset_dir, fname))
            annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    active = [f"B{b}=T{t_vector[b]:.2f}" for b in range(1, N_BUCKETS) if t_vector[b] < 1.0]
    print("  CREATED: %s (%d imgs, active: %s)" % (name, len(annotations), ", ".join(active) or "none"))
    return False

def main():
    print("=" * 60)
    print("CelebA v2b — Coarse 2D Sweep Dataset Creation")
    print("=" * 60)
    if not os.path.exists(SHARED_DIR):
        print("ERROR: %s not found. Run prepare_celeba_v2b.py first." % SHARED_DIR)
        sys.exit(1)

    all_bucket_files = {}
    total = 0
    for b in range(N_BUCKETS):
        all_bucket_files[b] = get_bucket_files(b)
        total += len(all_bucket_files[b])
        print("  Bucket %d (sigma=%.2f): %d images" % (b, BLUR_SIGMAS[b], len(all_bucket_files[b])))
    print("  Total per dataset: %d" % total)

    created = skipped = 0

    print("\n--- Baseline ---")
    tvec = {b: (0.0 if b == 0 else 1.0) for b in range(N_BUCKETS)}
    s = create_dataset("celeba_v2b_baseline", tvec, all_bucket_files)
    created, skipped = (created, skipped+1) if s else (created+1, skipped)

    print("\n--- Coarse 2D sweep ---")
    for b in range(1, N_BUCKETS):
        print(f"\n  Bucket {b} (sigma={BLUR_SIGMAS[b]}):")
        for t_val in COARSE_T:
            name = "celeba_v2b_b%d_T%s" % (b, t_to_suffix(t_val))
            tvec = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
            tvec[b] = t_val
            s = create_dataset(name, tvec, all_bucket_files)
            created, skipped = (created, skipped+1) if s else (created+1, skipped)

    manifest = {
        "version": "v2b_coarse", "format": "full_8bucket",
        "blur_sigmas": {str(b): float(s) for b, s in BLUR_SIGMAS.items()},
        "coarse_t_values": COARSE_T, "t_off_sigma_min": T_OFF_SIGMA_MIN, "datasets": {},
    }
    manifest["datasets"]["celeba_v2b_baseline"] = {"t_vector": [0.0]+[1.0]*7, "active_buckets": []}
    for b in range(1, N_BUCKETS):
        for t_val in COARSE_T:
            name = "celeba_v2b_b%d_T%s" % (b, t_to_suffix(t_val))
            tvec = [0.0]+[1.0]*7; tvec[b] = t_val
            manifest["datasets"][name] = {"t_vector": tvec, "active_buckets": [b],
                                          "active_t": t_val, "blur_sigma": BLUR_SIGMAS[b]}
    with open(os.path.join(ANNOTATED_DIR, "celeba_v2b_coarse_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCreated: {created}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
