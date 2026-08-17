"""
Per-point additive prediction vs actual for all BO + independence test points.
Computes delta = actual - additive_prediction for each point.
Positive delta = super-additive (worse than additivity predicts).
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, glob, re
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr

METRICS_DIR = f"{AMBIENT_BASE}/generated"

# --- 1. Build per-bucket FID(T) interpolators from 2-domain data ---
bucket_data = {b: {'T': [], 'FID': []} for b in range(1, 8)}

for fpath in sorted(glob.glob(os.path.join(METRICS_DIR, "metrics_celeba_2d_b*_T*_2000kimg.json"))):
    fname = os.path.basename(fpath)
    m = re.match(r'metrics_celeba_2d_b(\d+)_T(\w+)_2000kimg\.json', fname)
    if not m:
        continue
    bucket = int(m.group(1))
    suffix = m.group(2)
    n = int(suffix)
    t_val = n / 100.0 if len(suffix) == 3 else n / 1000.0

    with open(fpath) as f:
        fid = json.load(f)['fid_score']
    bucket_data[bucket]['T'].append(t_val)
    bucket_data[bucket]['FID'].append(fid)

bucket_interp = {}
bucket_t1_fid = {}
for b in range(1, 8):
    T = np.array(bucket_data[b]['T'])
    F = np.array(bucket_data[b]['FID'])
    idx = np.argsort(T)
    bucket_interp[b] = interp1d(T[idx], F[idx], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
    # T=1 baseline for this bucket
    i1 = np.argmin(np.abs(T - 1.0))
    bucket_t1_fid[b] = F[i1]

# Overall baseline = mean of all T=1 observations
all_t1 = list(bucket_t1_fid.values())
baseline_path = os.path.join(METRICS_DIR, "metrics_celeba_indep_baseline_2000kimg.json")
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        all_t1.append(json.load(f)['fid_score'])
baseline = np.mean(all_t1)

print(f"Baseline (mean T=1 FID): {baseline:.2f}")
print(f"Per-bucket T=1: {', '.join(f'B{b}={bucket_t1_fid[b]:.2f}' for b in range(1,8))}")
print()

# --- 2. Load all off-axis points (BO + independence) ---
points = []

# BO rounds
for r in [2, 3]:
    for i in range(15):
        name = f"celeba_bo_r{r}_p{i:02d}"
        tvp = os.path.join(METRICS_DIR, f"tvec_{name}.json")
        mp = os.path.join(METRICS_DIR, f"metrics_{name}_2000kimg.json")
        if os.path.exists(tvp) and os.path.exists(mp):
            with open(tvp) as f: t_vec = json.load(f)
            with open(mp) as f: actual = json.load(f)['fid_score']
            points.append({'name': name, 't_vec': t_vec, 'actual': actual})

# Independence tests
for iname, tvec in [
    ('indep2k_all_argmin',  [0.95, 0.95, 1.0, 0.97, 0.99, 0.99, 1.0]),
    ('indep2k_argmin_low',  [0.95, 0.95, 1.0, 1.0,  1.0,  1.0,  1.0]),
    ('indep2k_argmin_high', [1.0,  1.0,  1.0, 0.97, 0.99, 0.99, 1.0]),
]:
    mp = os.path.join(METRICS_DIR, f"metrics_celeba_{iname}_2000kimg.json")
    if os.path.exists(mp):
        with open(mp) as f: actual = json.load(f)['fid_score']
        points.append({'name': iname, 't_vec': tvec, 'actual': actual})

# --- 3. Compute per-point additive prediction and delta ---
print(f"{'Name':<28s} {'Actual':>6s} {'Addit':>6s} {'Delta':>7s} {'#Act':>4s} {'Aggr':>6s}  Per-bucket deltas")
print("-" * 110)

results = []
for pt in sorted(points, key=lambda x: x['actual']):
    t_vec = pt['t_vec']
    n_active = 0
    delta_sum = 0.0
    aggressiveness = 0.0
    bucket_deltas = []

    for i, b in enumerate(range(1, 8)):
        t_val = t_vec[i]
        if t_val >= 0.999:
            bucket_deltas.append("  --  ")
            continue
        n_active += 1
        aggressiveness += (1.0 - t_val)
        fid_interp = float(bucket_interp[b](t_val))
        bd = fid_interp - bucket_t1_fid[b]
        delta_sum += bd
        bucket_deltas.append(f"{bd:+5.2f}")

    additive_pred = baseline + delta_sum
    delta = pt['actual'] - additive_pred

    results.append({
        'name': pt['name'], 'actual': pt['actual'],
        'additive': additive_pred, 'delta': delta,
        'n_active': n_active, 'aggressiveness': aggressiveness
    })

    bd_str = " ".join(f"B{j+1}:{d}" for j, d in enumerate(bucket_deltas))
    print(f"{pt['name']:<28s} {pt['actual']:6.2f} {additive_pred:6.2f} {delta:+7.2f} {n_active:4d} {aggressiveness:6.3f}  {bd_str}")

# --- 4. Summary stats ---
deltas = [r['delta'] for r in results]
n_acts = [r['n_active'] for r in results]
aggs = [r['aggressiveness'] for r in results]

print()
print(f"Delta stats (n={len(deltas)}):")
print(f"  Mean: {np.mean(deltas):+.2f}")
print(f"  Std:  {np.std(deltas):.2f}")
print(f"  Min:  {np.min(deltas):+.2f}  Max: {np.max(deltas):+.2f}")
print(f"  All positive (super-additive): {all(d > 0 for d in deltas)}")
print(f"  Fraction positive: {sum(d > 0 for d in deltas)}/{len(deltas)}")

if len(set(n_acts)) > 1:
    r_n, p_n = pearsonr(n_acts, deltas)
    rs_n, ps_n = spearmanr(n_acts, deltas)
    print(f"\nCorrelation delta vs #active buckets:")
    print(f"  Pearson:  r={r_n:.3f} (p={p_n:.3f})")
    print(f"  Spearman: r={rs_n:.3f} (p={ps_n:.3f})")

r_a, p_a = pearsonr(aggs, deltas)
rs_a, ps_a = spearmanr(aggs, deltas)
print(f"\nCorrelation delta vs aggressiveness (sum of 1-T for active buckets):")
print(f"  Pearson:  r={r_a:.3f} (p={p_a:.3f})")
print(f"  Spearman: r={rs_a:.3f} (p={ps_a:.3f})")

# Save results for Jupyter
out_path = os.path.join(METRICS_DIR, "bo_independence_analysis.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
