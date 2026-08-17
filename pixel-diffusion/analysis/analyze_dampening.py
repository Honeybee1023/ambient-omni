"""
Analyze whether the r=0.815 (delta vs predicted_improvement) finding
survives removing the shared-variable confound.

Clean test: r(actual_FID, additive_prediction) — no shared variables.
Slope test: regress actual_improvement on predicted_improvement — slope ≈ 0.5 if dampening is real.
Permutation test: null distribution of r(delta, predicted_improvement) under shuffled actuals.

Reads: bo_independence_analysis.json
Saves: dampening_analysis.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json
import numpy as np
from scipy import stats
import os

METRICS_DIR = f"{AMBIENT_BASE}/generated"
INPUT = os.path.join(METRICS_DIR, "bo_independence_analysis.json")
OUTPUT = os.path.join(METRICS_DIR, "dampening_analysis.json")

# --- Load data ---
with open(INPUT) as f:
    data = json.load(f)

# Inspect structure
print("=== TOP-LEVEL KEYS ===")
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            if len(v) > 0:
                print(f"    first item type: {type(v[0])}")
                if isinstance(v[0], dict):
                    print(f"    first item keys: {list(v[0].keys())}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())[:10]}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
elif isinstance(data, list):
    print(f"  top-level list of {len(data)} items")
    if len(data) > 0 and isinstance(data[0], dict):
        print(f"  first item keys: {list(data[0].keys())}")

# --- Extract arrays ---
# Try to find the per-point records
if isinstance(data, dict) and "points" in data:
    points = data["points"]
elif isinstance(data, list):
    points = data
else:
    # Try to find any list of dicts
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            points = v
            print(f"\nUsing key '{k}' as points list")
            break
    else:
        print("ERROR: Cannot find per-point data. Printing full structure:")
        print(json.dumps(data, indent=2)[:3000])
        exit(1)

print(f"\n=== FOUND {len(points)} POINTS ===")
print(f"First point: {json.dumps(points[0], indent=2)}")

# Try to extract actual_fid and additive_prediction
# Field names might vary — try common patterns
def find_field(point, candidates):
    for c in candidates:
        if c in point:
            return c
    return None

fid_field = find_field(points[0], ["actual_fid", "actual", "fid", "actual_FID"])
pred_field = find_field(points[0], ["additive_prediction", "additive_pred", "predicted", "additive"])
delta_field = find_field(points[0], ["delta", "deviation"])
name_field = find_field(points[0], ["name", "dataset", "point", "config"])

print(f"\nField mapping: fid={fid_field}, pred={pred_field}, delta={delta_field}, name={name_field}")

if not fid_field or not pred_field:
    print("ERROR: Cannot find actual FID or additive prediction fields.")
    print(f"Available fields: {list(points[0].keys())}")
    exit(1)

# Extract arrays
actual_fids = np.array([p[fid_field] for p in points])
additive_preds = np.array([p[pred_field] for p in points])
names = [p.get(name_field, f"p{i}") for i, p in enumerate(points)]

if delta_field:
    deltas = np.array([p[delta_field] for p in points])
else:
    deltas = actual_fids - additive_preds

BASELINE = 26.82  # multi-bucket T=1 mean from 2-domain
# Check if baseline is stored in the data
if isinstance(data, dict) and "baseline" in data:
    BASELINE = data["baseline"]
    print(f"Using baseline from data: {BASELINE}")
else:
    print(f"Using hardcoded baseline: {BASELINE}")

predicted_improvements = BASELINE - additive_preds
actual_improvements = BASELINE - actual_fids

print(f"\n=== SUMMARY STATS ===")
print(f"Actual FID:    mean={actual_fids.mean():.2f}, std={actual_fids.std():.2f}, range=[{actual_fids.min():.2f}, {actual_fids.max():.2f}]")
print(f"Additive pred: mean={additive_preds.mean():.2f}, std={additive_preds.std():.2f}, range=[{additive_preds.min():.2f}, {additive_preds.max():.2f}]")
print(f"Delta:         mean={deltas.mean():.2f}, std={deltas.std():.2f}")
print(f"Pred improve:  mean={predicted_improvements.mean():.2f}, std={predicted_improvements.std():.2f}")
print(f"Act improve:   mean={actual_improvements.mean():.2f}, std={actual_improvements.std():.2f}")

# --- Correlation analysis ---
print(f"\n=== CORRELATION ANALYSIS ===")

# 1. Original: r(delta, predicted_improvement) — the r=0.815 with shared variable
r_orig, p_orig = stats.pearsonr(deltas, predicted_improvements)
print(f"r(delta, predicted_improvement)         = {r_orig:.3f}  (p={p_orig:.4f})  [ORIGINAL — shared variable]")

# 2. Clean: r(actual_FID, additive_prediction) — no shared variable
r_clean, p_clean = stats.pearsonr(actual_fids, additive_preds)
print(f"r(actual_FID, additive_prediction)      = {r_clean:.3f}  (p={p_clean:.4f})  [CLEAN — no shared var]")

# 3. Improvement space: r(actual_improvement, predicted_improvement)
r_improve, p_improve = stats.pearsonr(actual_improvements, predicted_improvements)
print(f"r(actual_improvement, predicted_improvement) = {r_improve:.3f}  (p={p_improve:.4f})")

# 4. Linear regression: actual_improvement = slope * predicted_improvement + intercept
slope, intercept, r_val, p_val, se = stats.linregress(predicted_improvements, actual_improvements)
print(f"\nLinear fit: actual_improve = {slope:.3f} * predicted_improve + {intercept:.3f}")
print(f"  slope = {slope:.3f} (SE={se:.3f}) — 1.0 = perfect additive, 0.0 = data useless")
print(f"  R² = {r_val**2:.3f}")

# 5. Linear regression: actual_FID = slope * additive_pred + intercept
slope2, intercept2, r_val2, p_val2, se2 = stats.linregress(additive_preds, actual_fids)
print(f"\nLinear fit: actual_FID = {slope2:.3f} * additive_pred + {intercept2:.3f}")
print(f"  slope = {slope2:.3f} (SE={se2:.3f}) — 1.0 = perfect additive")
print(f"  R² = {r_val2**2:.3f}")

# 6. Permutation test: null distribution of r(delta, predicted_improvement)
np.random.seed(42)
n_perm = 10000
null_rs = np.zeros(n_perm)
for i in range(n_perm):
    shuffled_actuals = np.random.permutation(actual_fids)
    shuffled_deltas = shuffled_actuals - additive_preds
    shuffled_pred_improve = BASELINE - additive_preds  # same for all permutations
    null_rs[i] = np.corrcoef(shuffled_deltas, shuffled_pred_improve)[0, 1]

null_mean = null_rs.mean()
null_std = null_rs.std()
null_p = (null_rs >= r_orig).mean()
print(f"\n=== PERMUTATION TEST (n={n_perm}) ===")
print(f"Null distribution: mean={null_mean:.3f}, std={null_std:.3f}")
print(f"Observed r={r_orig:.3f}, permutation p={null_p:.4f}")
print(f"Null 95th percentile: {np.percentile(null_rs, 95):.3f}")
print(f"Null 99th percentile: {np.percentile(null_rs, 99):.3f}")
print(f"Z-score vs null: {(r_orig - null_mean) / null_std:.2f}")

# --- Save results ---
results = {
    "n_points": len(points),
    "baseline_fid": BASELINE,
    "correlations": {
        "delta_vs_predicted_improvement": {"r": round(r_orig, 4), "p": round(p_orig, 6), "note": "shared variable — inflated"},
        "actual_fid_vs_additive_pred": {"r": round(r_clean, 4), "p": round(p_clean, 6), "note": "clean — no shared variable"},
        "actual_improvement_vs_predicted_improvement": {"r": round(r_improve, 4), "p": round(p_improve, 6)},
    },
    "regression_improvement_space": {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "se_slope": round(se, 4),
        "r_squared": round(r_val**2, 4),
        "interpretation": f"actual improvement ≈ {slope:.1%} of predicted improvement"
    },
    "regression_fid_space": {
        "slope": round(slope2, 4),
        "intercept": round(intercept2, 4),
        "se_slope": round(se2, 4),
        "r_squared": round(r_val2**2, 4),
    },
    "permutation_test": {
        "null_mean_r": round(null_mean, 4),
        "null_std_r": round(null_std, 4),
        "null_95th": round(float(np.percentile(null_rs, 95)), 4),
        "null_99th": round(float(np.percentile(null_rs, 99)), 4),
        "observed_r": round(r_orig, 4),
        "permutation_p": round(float(null_p), 4),
        "z_score": round((r_orig - null_mean) / null_std, 2),
    },
    "per_point": [
        {
            "name": names[i],
            "actual_fid": round(float(actual_fids[i]), 2),
            "additive_pred": round(float(additive_preds[i]), 2),
            "delta": round(float(deltas[i]), 2),
            "predicted_improvement": round(float(predicted_improvements[i]), 2),
            "actual_improvement": round(float(actual_improvements[i]), 2),
        }
        for i in range(len(points))
    ]
}

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
