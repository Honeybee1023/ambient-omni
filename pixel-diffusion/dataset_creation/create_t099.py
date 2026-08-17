
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
SHARED_DIR = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
ANNOTATED_DIR = f"{AMBIENT_BASE}/annotated_datasets"
N_BUCKETS = 8
T_OFF = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))

def t_to_sigma_min(t):
    if t == 0: return 0.0
    if t >= 1.0: return T_OFF
    return float(np.exp(P_STD * norm.ppf(np.clip(t, 0.001, 0.999)) + P_MEAN))

bucket_files = {}
for b in range(N_BUCKETS):
    prefix = "b%d_" % b
    bucket_files[b] = sorted([f for f in os.listdir(SHARED_DIR)
                               if f.startswith(prefix) and f.endswith(".jpg")
                               and not f.startswith("._")])

for b in [2, 3, 4, 6, 7]:
    name = "celeba_v2b_b%d_T099" % b
    ddir = os.path.join(ANNOTATED_DIR, name)
    ann_path = os.path.join(ddir, "annotations.jsonl")
    if os.path.exists(ann_path):
        print("SKIP (exists): %s" % name)
        continue
    if os.path.exists(ddir):
        shutil.rmtree(ddir)
    os.makedirs(ddir)
    tvec = {bb: (0.0 if bb == 0 else 1.0) for bb in range(N_BUCKETS)}
    tvec[b] = 0.99
    anns = []
    for bucket in range(N_BUCKETS):
        sm = t_to_sigma_min(tvec[bucket])
        for fname in bucket_files[bucket]:
            os.symlink(os.path.join(SHARED_DIR, fname), os.path.join(ddir, fname))
            anns.append({"filename": fname, "sigma_min": sm, "sigma_max": 0.0})
    with open(ann_path, "w") as f:
        for a in anns:
            f.write(json.dumps(a) + "\n")
    print("CREATED: %s (%d images)" % (name, len(anns)))
