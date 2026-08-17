"""
Create datasets for:
1. Confound test: axis-aligned multi-bucket (same format as BO, but 1-2 buckets active)
2. Context sweeps: fix other buckets at argmin, sweep one bucket's T
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, shutil
import numpy as np
from scipy.stats import norm

P_MEAN, P_STD = -1.2, 1.2
SHARED_DIR = f"{AMBIENT_BASE}/celeba_processed/shared_buckets_64"
ANNOTATED_DIR = f"{AMBIENT_BASE}/annotated_datasets"
METRICS_DIR = f"{AMBIENT_BASE}/generated"
BUCKETS = [1,2,3,4,5,6,7]

def t_to_sigma_min(t):
    if t <= 0.001: return 0.0
    if t >= 0.999: return float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))
    return float(np.exp(P_STD * norm.ppf(t) + P_MEAN))

def get_bucket_files(b):
    prefix = f"b{b}_"
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg') and not f.startswith('._')])

def create_dataset(name, t_vec):
    dataset_dir = os.path.join(ANNOTATED_DIR, name)
    ann_path = os.path.join(dataset_dir, "annotations.jsonl")
    if os.path.exists(ann_path):
        print(f"  SKIP (exists): {name}")
        return
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    annotations = []
    # Bucket 0 (clean)
    for fname in get_bucket_files(0):
        os.symlink(os.path.join(SHARED_DIR, fname), os.path.join(dataset_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})
    # Buckets 1-7
    for i, b in enumerate(BUCKETS):
        sigma_min = t_to_sigma_min(t_vec[i])
        for fname in get_bucket_files(b):
            os.symlink(os.path.join(SHARED_DIR, fname), os.path.join(dataset_dir, fname))
            annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    # Save T-vector sidecar
    with open(os.path.join(METRICS_DIR, f"tvec_{name}.json"), "w") as f:
        json.dump(t_vec, f)

    active = [(i+1, t_vec[i]) for i in range(7) if t_vec[i] < 0.999]
    print(f"  CREATED: {name} ({len(annotations)} imgs, {len(active)} active)")

# === 1. CONFOUND TESTS ===
print("=== Confound Tests (axis-aligned in multi-bucket format) ===")
confound_configs = {
    "celeba_confound_b2only":  [1.0, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0],
    "celeba_confound_b1only":  [0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "celeba_confound_b1b2":    [0.95, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0],
}
for name, tvec in confound_configs.items():
    create_dataset(name, tvec)

# === 2. CONTEXT SWEEPS ===
# 2k argmins: B1=0.95, B2=0.95, B3=1.0, B4=0.97, B5=0.99, B6=0.99, B7=1.0
ARGMIN_T = [0.95, 0.95, 1.0, 0.97, 0.99, 0.99, 1.0]

# B2 context sweep: fix others at argmin, sweep B2
print("\n=== B2 Context Sweep (others at argmin) ===")
b2_sweep_T = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
for t in b2_sweep_T:
    tvec = ARGMIN_T.copy()
    tvec[1] = t  # B2 = index 1
    suffix = f"{int(t*100):03d}" if t*100 == int(t*100) else f"{int(t*1000):04d}"
    name = f"celeba_ctx_b2_T{suffix}"
    create_dataset(name, tvec)

# B3 context sweep: fix others at argmin, sweep B3
print("\n=== B3 Context Sweep (others at argmin) ===")
b3_sweep_T = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
for t in b3_sweep_T:
    tvec = ARGMIN_T.copy()
    tvec[2] = t  # B3 = index 2
    suffix = f"{int(t*100):03d}" if t*100 == int(t*100) else f"{int(t*1000):04d}"
    name = f"celeba_ctx_b3_T{suffix}"
    create_dataset(name, tvec)

print("\n=== Done ===")
