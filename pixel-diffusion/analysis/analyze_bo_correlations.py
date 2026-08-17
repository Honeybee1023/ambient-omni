"""Extended correlation analysis: what predicts the independence delta?"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, numpy as np
from scipy.stats import pearsonr, spearmanr

data = json.load(open(f"{AMBIENT_BASE}/generated/bo_independence_analysis.json"))

deltas = np.array([d['delta'] for d in data])
n_active = np.array([d['n_active'] for d in data])
aggressiveness = np.array([d['aggressiveness'] for d in data])
actuals = np.array([d['actual'] for d in data])
additives = np.array([d['additive'] for d in data])
predicted_improvement = np.array([26.82 - d['additive'] for d in data])  # how much additive model thinks it'll help

# Load T-vectors from tvec files for more analysis
import os, glob
tvecs = {}
for d in data:
    name = d['name']
    tvp = f"{AMBIENT_BASE}/generated/tvec_{name}.json"
    if os.path.exists(tvp):
        tvecs[name] = json.load(open(tvp))
    elif 'indep2k_all_argmin' in name:
        tvecs[name] = [0.95, 0.95, 1.0, 0.97, 0.99, 0.99, 1.0]
    elif 'indep2k_argmin_low' in name:
        tvecs[name] = [0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif 'indep2k_argmin_high' in name:
        tvecs[name] = [1.0, 1.0, 1.0, 0.97, 0.99, 0.99, 1.0]

# Per-bucket T values
bucket_ts = {b: [] for b in range(7)}
for d in data:
    tv = tvecs.get(d['name'])
    if tv:
        for b in range(7):
            bucket_ts[b].append(tv[b])

min_t = np.array([min(tvecs[d['name']]) for d in data if d['name'] in tvecs])
mean_t = np.array([np.mean(tvecs[d['name']]) for d in data if d['name'] in tvecs])

print("=== Correlation with delta (positive = super-additive) ===\n")
candidates = [
    ("# active buckets", n_active),
    ("Aggressiveness (sum 1-T)", aggressiveness),
    ("Predicted improvement", predicted_improvement),
    ("Additive prediction", additives),
    ("Min T across buckets", min_t),
    ("Mean T across buckets", mean_t),
    ("Actual FID", actuals),
]

for label, arr in candidates:
    if len(arr) == len(deltas):
        r, p = pearsonr(arr, deltas)
        rs, ps = spearmanr(arr, deltas)
        print(f"  {label:<30s}  Pearson r={r:+.3f} (p={p:.3f})  Spearman r={rs:+.3f} (p={ps:.3f})")

# Per-bucket T vs delta
print("\n=== Per-bucket T vs delta ===")
for b in range(7):
    arr = np.array(bucket_ts[b])
    if len(arr) == len(deltas):
        r, p = pearsonr(arr, deltas)
        print(f"  B{b+1} T value vs delta:  r={r:+.3f} (p={p:.3f})")

# Group by #active and show mean delta
print("\n=== Mean delta by #active buckets ===")
for n in sorted(set(n_active)):
    mask = n_active == n
    print(f"  {n} active: mean delta={deltas[mask].mean():+.2f} ± {deltas[mask].std():.2f} (n={mask.sum()})")

# Save extended data for Jupyter plotting
extended = []
for d in data:
    tv = tvecs.get(d['name'], [None]*7)
    extended.append({**d, 't_vec': tv,
                     'predicted_improvement': 26.82 - d['additive'],
                     'min_t': min(tv) if tv[0] is not None else None,
                     'mean_t': np.mean(tv) if tv[0] is not None else None})
json.dump(extended, open(f"{AMBIENT_BASE}/generated/bo_extended_analysis.json", "w"), indent=2)
print("\nExtended data saved for Jupyter plotting.")
