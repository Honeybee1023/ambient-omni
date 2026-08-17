
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, shutil, numpy as np
from scipy.stats import norm

P_MEAN, P_STD = -1.2, 1.2
SHARED_DIR = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
ANNOTATED_DIR = f"{AMBIENT_BASE}/annotated_datasets"
N_BUCKETS = 8
T_OFF_SIGMA_MIN = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))

def t_to_sigma_min(t):
    if t == 0: return 0.0
    return float(np.exp(P_STD * norm.ppf(np.clip(t, 0.001, 0.999)) + P_MEAN))

def t_to_suffix(t):
    m = round(t * 1000)
    return "%03d" % (m // 10) if m % 10 == 0 else "%04d" % m

def get_bucket_files(b):
    pfx = "b%d_" % b
    return sorted([f for f in os.listdir(SHARED_DIR) if f.startswith(pfx) and f.endswith('.jpg')])

def create_dataset(name, tvec, all_files):
    d = os.path.join(ANNOTATED_DIR, name)
    if os.path.exists(os.path.join(d, "annotations.jsonl")):
        print("  SKIP:", name); return
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    anns = []
    for b in range(N_BUCKETS):
        t = tvec[b]
        sm = 0.0 if t == 0 else (T_OFF_SIGMA_MIN if t >= 1.0 else t_to_sigma_min(t))
        for f in all_files[b]:
            os.symlink(os.path.join(SHARED_DIR, f), os.path.join(d, f))
            anns.append({"filename": f, "sigma_min": sm, "sigma_max": 0.0})
    with open(os.path.join(d, "annotations.jsonl"), "w") as fh:
        for a in anns: fh.write(json.dumps(a) + "\n")
    print("  CREATED:", name, len(anns), "imgs")

all_files = [get_bucket_files(b) for b in range(N_BUCKETS)]

# 8-bucket tvec: all buckets at T=1 (off) except bucket 0 (T=0) and the active one
def make_tvec(bucket, t_val):
    v = [1.0] * N_BUCKETS
    v[0] = 0.0
    v[bucket] = t_val
    return v

DATASETS = {}
# Min hunt round 2
for t in [0.475, 0.525, 0.55]: DATASETS["celeba_v2b_b1_T%s" % t_to_suffix(t)] = make_tvec(1, t)
for t in [0.425, 0.475]: DATASETS["celeba_v2b_b2_T%s" % t_to_suffix(t)] = make_tvec(2, t)
for t in [0.475, 0.525]: DATASETS["celeba_v2b_b3_T%s" % t_to_suffix(t)] = make_tvec(3, t)
for t in [0.45, 0.475, 0.525]: DATASETS["celeba_v2b_b4_T%s" % t_to_suffix(t)] = make_tvec(4, t)
for t in [0.45, 0.475, 0.525]: DATASETS["celeba_v2b_b5_T%s" % t_to_suffix(t)] = make_tvec(5, t)
for t in [0.5, 0.7]: DATASETS["celeba_v2b_b6_T%s" % t_to_suffix(t)] = make_tvec(6, t)
for t in [0.5, 0.7]: DATASETS["celeba_v2b_b7_T%s" % t_to_suffix(t)] = make_tvec(7, t)
# T=0.99
for b in [2, 4, 6]: DATASETS["celeba_v2b_b%d_T099" % b] = make_tvec(b, 0.99)

print("Creating %d datasets..." % len(DATASETS))
for name in sorted(DATASETS):
    create_dataset(name, DATASETS[name], all_files)
print("Done!")
