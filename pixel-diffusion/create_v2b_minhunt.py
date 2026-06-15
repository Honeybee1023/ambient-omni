"""
Fine-grained minimum hunt for all buckets.
Fills gaps around the current empirical minimums.

B1: {0.475}                                    (1)
B2: {0.425, 0.475, 0.525}                      (3)
B3: {0.475, 0.525}                              (2)
B4: {0.35, 0.45, 0.475}                         (3)
B5: {0.3, 0.35, 0.375, 0.425, 0.45, 0.475}     (6)
B6: {0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7}    (7)
B7: {0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7}    (7)
Total: 29 new datasets
"""
import os, json, shutil, sys
import numpy as np
from scipy.stats import norm

P_MEAN, P_STD = -1.2, 1.2
SHARED_DIR = "/data/scratch/honjar/celeba_processed_v2b/shared_buckets_64"
ANNOTATED_DIR = "/data/scratch/honjar/annotated_datasets"
N_BUCKETS = 8
T_OFF = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))

def t_to_sigma_min(t):
    if t == 0: return 0.0
    if t >= 1.0: return T_OFF
    return float(np.exp(P_STD * norm.ppf(np.clip(t, 0.001, 0.999)) + P_MEAN))

def t_to_suffix(t_val):
    t_milli = round(t_val * 1000)
    if t_milli % 10 == 0:
        return "%03d" % (t_milli // 10)
    else:
        return "%04d" % t_milli

bucket_files = {}
for b in range(N_BUCKETS):
    prefix = "b%d_" % b
    bucket_files[b] = sorted([f for f in os.listdir(SHARED_DIR)
                               if f.startswith(prefix) and f.endswith(".jpg")
                               and not f.startswith("._")])
    print("  Bucket %d: %d images" % (b, len(bucket_files[b])))

new_points = {
    1: [0.475],
    2: [0.425, 0.475, 0.525],
    3: [0.475, 0.525],
    4: [0.35, 0.45, 0.475],
    5: [0.3, 0.35, 0.375, 0.425, 0.45, 0.475],
    6: [0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7],
    7: [0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7],
}

created = 0
skipped = 0
for b, t_values in sorted(new_points.items()):
    print("\nBucket %d:" % b)
    for t_val in t_values:
        suffix = t_to_suffix(t_val)
        name = "celeba_v2b_b%d_T%s" % (b, suffix)
        ddir = os.path.join(ANNOTATED_DIR, name)
        ann_path = os.path.join(ddir, "annotations.jsonl")
        if os.path.exists(ann_path):
            print("  SKIP: %s" % name)
            skipped += 1
            continue
        if os.path.exists(ddir):
            shutil.rmtree(ddir)
        os.makedirs(ddir)
        tvec = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
        tvec[b] = t_val
        anns = []
        for bucket in range(N_BUCKETS):
            sm = t_to_sigma_min(tvec[bucket])
            for fname in bucket_files[bucket]:
                os.symlink(os.path.join(SHARED_DIR, fname), os.path.join(ddir, fname))
                anns.append({"filename": fname, "sigma_min": sm, "sigma_max": 0.0})
        with open(ann_path, "w") as f:
            for a in anns:
                f.write(json.dumps(a) + "\n")
        print("  CREATED: %s" % name)
        created += 1

print("\nCreated: %d, Skipped: %d" % (created, skipped))
