"""
Functional form analysis: exponential vs power law for PickScore curves.
Also compares 2-domain vs 7-category curves (independence test).
Reads all data from JSON files on disk. Outputs results to JSON.
"""
import json, glob, re, os
import numpy as np

PS_WOLVES = 21.60
V_WOLVES = 3.35

def collect_data(prefix):
    """Read all metrics files for a given experiment prefix."""
    pattern = f'/data/scratch/honjar/generated/metrics_{prefix}_*_1000kimg.json'
    files = sorted(glob.glob(pattern))
    by_cat = {}
    for f in files:
        basename = os.path.basename(f)
        m = re.match(f'metrics_{prefix}_(\w+)_T(\w+)_1000kimg\.json', basename)
        if not m:
            continue
        cat = m.group(1)
        t_suffix = m.group(2)
        if len(t_suffix) == 3:
            t_val = int(t_suffix) / 100.0
        elif len(t_suffix) == 4:
            t_val = int(t_suffix) / 1000.0
        else:
            continue
        try:
            d = json.load(open(f))
            ps = d['pickscore']['mean']
            v = d['vendi']['score']
            by_cat.setdefault(cat, []).append((t_val, ps, v))
        except Exception as e:
            print(f'ERROR reading {basename}: {e}')
    # Sort each category by T
    for cat in by_cat:
        by_cat[cat] = sorted(by_cat[cat])
    return by_cat

def fit_linear(x, y):
    """Fit y = a*x + b, return (slope, intercept, R²)."""
    if len(x) < 3:
        return None, None, None
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    return coeffs[0], coeffs[1], r2

def analyze_pickscore_forms(T_arr, PS_arr, ceiling=PS_WOLVES):
    """Test exponential vs power law for PickScore 'damage' = ceiling - PS."""
    damage = ceiling - PS_arr
    one_minus_T = 1.0 - T_arr
    # Filter valid points (damage > 0, 1-T > 0)
    mask = (damage > 0.03) & (one_minus_T > 0.003)
    if mask.sum() < 4:
        return {'exp_r2': None, 'pow_r2': None, 'n_points': int(mask.sum())}
    
    T_f = T_arr[mask]
    dam_f = damage[mask]
    omt_f = one_minus_T[mask]
    
    # Exponential test: log(damage) vs T should be linear
    exp_slope, exp_int, exp_r2 = fit_linear(T_f, np.log(dam_f))
    
    # Power law test: log(damage) vs log(1-T) should be linear
    pow_slope, pow_int, pow_r2 = fit_linear(np.log(omt_f), np.log(dam_f))
    
    return {
        'exp_r2': round(float(exp_r2), 5),
        'exp_slope': round(float(exp_slope), 4),
        'exp_intercept': round(float(exp_int), 4),
        'pow_r2': round(float(pow_r2), 5),
        'pow_exponent': round(float(pow_slope), 4),
        'pow_intercept': round(float(pow_int), 4),
        'n_points': int(mask.sum()),
        'T_used': [round(float(t), 4) for t in T_f],
        'log_damage': [round(float(x), 4) for x in np.log(dam_f)],
        'log_1minusT': [round(float(x), 4) for x in np.log(omt_f)],
    }

# ===== Collect all data =====
data_2d = collect_data('pilot2d')
data_7c = collect_data('exp7d')

results = {'pickscore_form_test': {}, 'independence_test': {}, 'raw_data': {}}

# ===== PickScore functional form test (on 2-domain data) =====
print("=" * 60)
print("PICKSCORE FUNCTIONAL FORM: exponential vs power law")
print("=" * 60)
print(f"{'Category':>10s}  {'Pts':>4s}  {'Exp R²':>8s}  {'Pow R²':>8s}  {'Winner':>8s}  {'Pow α':>6s}")
print("-" * 55)

for cat in sorted(data_2d.keys()):
    points = data_2d[cat]
    T_arr = np.array([p[0] for p in points])
    PS_arr = np.array([p[1] for p in points])
    
    res = analyze_pickscore_forms(T_arr, PS_arr)
    results['pickscore_form_test'][cat] = res
    
    if res['exp_r2'] is not None:
        winner = 'EXP' if res['exp_r2'] > res['pow_r2'] else 'POWER'
        print(f"{cat:>10s}  {res['n_points']:>4d}  {res['exp_r2']:>8.4f}  {res['pow_r2']:>8.4f}  {winner:>8s}  {res.get('pow_exponent','?'):>6}")
    else:
        print(f"{cat:>10s}  insufficient data")

# ===== Independence test: 2-domain vs 7-category overlay =====
print("\n" + "=" * 60)
print("INDEPENDENCE TEST: 2-domain vs 7-category PickScore")
print("=" * 60)
print("(comparing PS values at matching T values)")
print(f"{'Category':>10s}  {'T':>6s}  {'2D PS':>8s}  {'7C PS':>8s}  {'Diff':>7s}")
print("-" * 50)

all_diffs = []
for cat in sorted(data_2d.keys()):
    if cat not in data_7c:
        continue
    d2 = {round(p[0], 4): p for p in data_2d[cat]}
    d7 = {round(p[0], 4): p for p in data_7c[cat]}
    cat_diffs = []
    for t_val in sorted(set(d2.keys()) & set(d7.keys())):
        ps_2d = d2[t_val][1]
        ps_7c = d7[t_val][1]
        v_2d = d2[t_val][2]
        v_7c = d7[t_val][2]
        diff = ps_7c - ps_2d
        cat_diffs.append({'T': t_val, 'ps_2d': ps_2d, 'ps_7c': ps_7c, 'diff': diff,
                          'v_2d': v_2d, 'v_7c': v_7c, 'v_diff': v_7c - v_2d})
        all_diffs.append(diff)
        print(f"{cat:>10s}  {t_val:>6.3f}  {ps_2d:>8.4f}  {ps_7c:>8.4f}  {diff:>+7.3f}")
    results['independence_test'][cat] = cat_diffs

if all_diffs:
    print(f"\nOverall: mean diff = {np.mean(all_diffs):+.4f}, std = {np.std(all_diffs):.4f}")
    print(f"PS training noise σ ≈ 0.03, so diffs within ±0.10 are noise-level")
    results['independence_summary'] = {
        'mean_diff': round(float(np.mean(all_diffs)), 4),
        'std_diff': round(float(np.std(all_diffs)), 4),
        'n_comparisons': len(all_diffs)
    }

# ===== Save raw data for plotting =====
for prefix, d in [('pilot2d', data_2d), ('exp7d', data_7c)]:
    results['raw_data'][prefix] = {}
    for cat, points in d.items():
        results['raw_data'][prefix][cat] = {
            'T': [p[0] for p in points],
            'PS': [p[1] for p in points],
            'V': [p[2] for p in points],
        }

outpath = '/data/scratch/honjar/generated/functional_form_analysis.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to: {outpath}")
