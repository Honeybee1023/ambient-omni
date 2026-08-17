"""
Collect ALL available results into a single JSON for Jupyter plotting.
Reads: 2-domain metrics, BO R2/R3/R4, confound tests, context sweeps, independence tests.
Saves: all_results_summary.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json
import os
import glob
import numpy as np
from scipy import stats

METRICS_DIR = f"{AMBIENT_BASE}/generated"
OUTPUT = os.path.join(METRICS_DIR, "all_results_summary.json")

def read_fid(path):
    """Read FID from metrics JSON, handling both formats."""
    try:
        with open(path) as f:
            d = json.load(f)
        if "fid_score" in d:
            return d["fid_score"]
        elif "results" in d and "fid50k_full" in d["results"]:
            return d["results"]["fid50k_full"]
        else:
            # Try first float value
            for v in d.values():
                if isinstance(v, (int, float)) and v > 10:  # FID should be > 10
                    return v
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
    return None

# ============================================================
# 1. TWO-DOMAIN RESULTS (all 2k kimg)
# ============================================================
print("=== COLLECTING 2-DOMAIN RESULTS ===")
two_domain = {}
for b in range(1, 8):
    two_domain[f"B{b}"] = {}

# Map T suffixes to T values
T_SUFFIXES = {
    "000": 0.0, "0063": 0.063, "0125": 0.125, "019": 0.19, "0188": 0.188,
    "025": 0.25, "031": 0.31, "0313": 0.313, "0375": 0.375,
    "050": 0.5, "0625": 0.625, "075": 0.75,
    "080": 0.80, "085": 0.85, "088": 0.88,
    "090": 0.90, "091": 0.91, "092": 0.92, "0925": 0.925,
    "093": 0.93, "094": 0.94, "095": 0.95, "096": 0.96,
    "097": 0.97, "098": 0.98, "099": 0.99, "100": 1.0,
}

for f in sorted(glob.glob(os.path.join(METRICS_DIR, "metrics_celeba_2d_b*_*_2000kimg.json"))):
    basename = os.path.basename(f)
    # Parse: metrics_celeba_2d_b{B}_T{suffix}_2000kimg.json
    parts = basename.replace("metrics_celeba_2d_b", "").replace("_2000kimg.json", "")
    # parts like "1_T095" or "2_T0925"
    bp = parts.split("_T")
    if len(bp) != 2:
        continue
    bucket_num = bp[0]
    t_suffix = bp[1]
    
    fid = read_fid(f)
    if fid is None:
        continue
    
    t_val = T_SUFFIXES.get(t_suffix)
    if t_val is None:
        # Try to parse as float
        try:
            t_val = float("0." + t_suffix) if not t_suffix.startswith("1") else float(t_suffix) / (10 ** (len(t_suffix) - 1))
        except:
            print(f"  WARNING: unknown T suffix '{t_suffix}' in {basename}")
            continue
    
    key = f"B{bucket_num}"
    if key in two_domain:
        two_domain[key][t_val] = round(fid, 2)

for b in range(1, 8):
    key = f"B{b}"
    n = len(two_domain[key])
    print(f"  {key}: {n} T values")

# ============================================================
# 2. BO RESULTS (R2, R3, R4)
# ============================================================
print("\n=== COLLECTING BO RESULTS ===")
bo_results = {}
for r in [2, 3, 4]:
    bo_results[f"R{r}"] = {}
    for i in range(15):
        suffix = f"celeba_bo_r{r}_p{i:02d}"
        metrics_path = os.path.join(METRICS_DIR, f"metrics_{suffix}_2000kimg.json")
        tvec_path = os.path.join(METRICS_DIR, f"tvec_{suffix}.json")
        
        fid = read_fid(metrics_path) if os.path.exists(metrics_path) else None
        tvec = None
        if os.path.exists(tvec_path):
            with open(tvec_path) as f:
                tvec = json.load(f)
        
        if fid is not None:
            bo_results[f"R{r}"][f"p{i:02d}"] = {
                "fid": round(fid, 2),
                "tvec": tvec
            }
    print(f"  R{r}: {len(bo_results[f'R{r}'])} points")

# ============================================================
# 3. CONFOUND TESTS
# ============================================================
print("\n=== COLLECTING CONFOUND TESTS ===")
confound_results = {}
for name in ["celeba_confound_b2only", "celeba_confound_b1only", "celeba_confound_b1b2"]:
    path = os.path.join(METRICS_DIR, f"metrics_{name}_2000kimg.json")
    fid = read_fid(path) if os.path.exists(path) else None
    tvec_path = os.path.join(METRICS_DIR, f"tvec_{name}.json")
    tvec = None
    if os.path.exists(tvec_path):
        with open(tvec_path) as f:
            tvec = json.load(f)
    if fid is not None:
        confound_results[name] = {"fid": round(fid, 2), "tvec": tvec}
        print(f"  {name}: FID={fid:.2f}")

# ============================================================
# 4. CONTEXT SWEEPS
# ============================================================
print("\n=== COLLECTING CONTEXT SWEEPS ===")
context_sweeps = {"B2": {}, "B3": {}}
for b in [2, 3]:
    for f in sorted(glob.glob(os.path.join(METRICS_DIR, f"metrics_celeba_ctx_b{b}_T*_2000kimg.json"))):
        basename = os.path.basename(f)
        t_suffix = basename.split(f"_b{b}_T")[1].replace("_2000kimg.json", "")
        t_val = T_SUFFIXES.get(t_suffix)
        if t_val is None:
            try:
                t_val = int(t_suffix) / 100.0
            except:
                continue
        fid = read_fid(f)
        if fid is not None:
            context_sweeps[f"B{b}"][t_val] = round(fid, 2)
    print(f"  B{b} context sweep: {len(context_sweeps[f'B{b}'])} T values")

# ============================================================
# 5. INDEPENDENCE TEST RESULTS
# ============================================================
print("\n=== COLLECTING INDEPENDENCE TESTS ===")
independence = {}
for name in ["celeba_indep_argmin_baseline", "celeba_indep_argmin_low", 
             "celeba_indep_argmin_high", "celeba_indep_argmin_all"]:
    path = os.path.join(METRICS_DIR, f"metrics_{name}_2000kimg.json")
    if os.path.exists(path):
        fid = read_fid(path)
        if fid is not None:
            independence[name] = round(fid, 2)
            print(f"  {name}: FID={fid:.2f}")

# ============================================================
# 6. DAMPENING ANALYSIS (updated with R4)
# ============================================================
print("\n=== DAMPENING ANALYSIS (with R4) ===")

# Load the per-point independence analysis
indep_path = os.path.join(METRICS_DIR, "bo_independence_analysis.json")
if os.path.exists(indep_path):
    with open(indep_path) as f:
        indep_data = json.load(f)
    
    # These are the original 33 points (R2 + R3 + independence off-axis)
    existing_names = set()
    if isinstance(indep_data, list):
        existing_names = {p["name"] for p in indep_data}
    
    print(f"  Existing points in independence analysis: {len(existing_names)}")
    print(f"  R4 points available: {len(bo_results.get('R4', {}))}")
    # Note: R4 points need additive predictions to be included in dampening analysis
    # That requires the 2-domain interpolation curves — flag for later

# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================
print("\n=== SUMMARY ===")
all_bo_fids = []
for r in ["R2", "R3", "R4"]:
    for p, data in bo_results.get(r, {}).items():
        all_bo_fids.append(data["fid"])

print(f"Total BO points: {len(all_bo_fids)}")
print(f"BO FID range: [{min(all_bo_fids):.2f}, {max(all_bo_fids):.2f}]")
print(f"BO FID mean: {np.mean(all_bo_fids):.2f}")
print(f"Best BO: {min(all_bo_fids):.2f}")
print(f"Total 2-domain points: {sum(len(v) for v in two_domain.values())}")

# ============================================================
# SAVE
# ============================================================
results = {
    "two_domain": {k: {str(t): v for t, v in sorted(vals.items())} for k, vals in two_domain.items()},
    "bo_results": bo_results,
    "confound_tests": confound_results,
    "context_sweeps": {k: {str(t): v for t, v in sorted(vals.items())} for k, vals in context_sweeps.items()},
    "independence_tests": independence,
    "metadata": {
        "baseline_2domain_mean_T1": 26.82,
        "baseline_multibucket_T1": 28.35,
        "best_single_bucket": {"name": "b2_T096", "fid": 25.62},
        "best_bo_point": {"name": "r3_p00", "fid": 25.77},
        "noise_floor_2domain": 1.74,
    }
}

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
