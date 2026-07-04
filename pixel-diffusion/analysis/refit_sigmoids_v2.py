"""
Refit sigmoids to full T range with per-bucket initial conditions.
Previous fit failed because optimizer got stuck at k=1, t_mid=0.5.
Fix: use knowledge of where the transition actually is per bucket.
Also try piecewise: linear(T<cutoff) + sigmoid(T>=cutoff) for continuous interpolation.
"""
import json
import numpy as np
from scipy.optimize import curve_fit, minimize
import warnings
warnings.filterwarnings('ignore')

ANALYSIS_PATH = "/data/scratch/honjar/generated/celeba_2k_analysis.json"
with open(ANALYSIS_PATH) as f:
    data = json.load(f)

BLUR_SIGMAS = {str(b): data['blur_sigmas'][str(b)] for b in range(1, 8)}

def sigmoid(T, fid_base, fid_plat, k, t_mid):
    return fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))

def sigmoid_slope(T, fid_base, fid_plat, k, t_mid, slope):
    """Sigmoid where the upper plateau has a linear slope.
    At T << t_mid: FID ≈ fid_base + slope*T
    At T >> t_mid: FID ≈ fid_plat
    Continuous everywhere, 5 parameters."""
    return fid_plat + (fid_base + slope * T - fid_plat) / (1.0 + np.exp(k * (T - t_mid)))

# Per-bucket priors based on what we SEE in the plots
BUCKET_PRIORS = {
    2: {'fid_base': 31, 'fid_plat': 26, 'k': 20, 't_mid': 0.65, 'slope': 0,
        'k_bounds': (5, 100), 'tmid_bounds': (0.4, 0.9)},
    3: {'fid_base': 53, 'fid_plat': 26, 'k': 40, 't_mid': 0.84, 'slope': 5,
        'k_bounds': (10, 200), 'tmid_bounds': (0.7, 0.95)},
    4: {'fid_base': 75, 'fid_plat': 27, 'k': 30, 't_mid': 0.90, 'slope': 20,
        'k_bounds': (10, 200), 'tmid_bounds': (0.8, 0.98)},
    5: {'fid_base': 92, 'fid_plat': 26, 'k': 30, 't_mid': 0.93, 'slope': 30,
        'k_bounds': (10, 200), 'tmid_bounds': (0.85, 0.99)},
    6: {'fid_base': 109, 'fid_plat': 27, 'k': 30, 't_mid': 0.95, 'slope': 30,
        'k_bounds': (10, 200), 'tmid_bounds': (0.88, 1.0)},
    7: {'fid_base': 140, 'fid_plat': 26, 'k': 40, 't_mid': 0.96, 'slope': 30,
        'k_bounds': (15, 200), 'tmid_bounds': (0.90, 1.02)},
}

new_fits = {}
print(f"{'Bucket':<8} {'σ':<5} {'Model':<15} {'R²':<8} {'RMSE':<8} {'k':<8} {'t_mid':<8} {'slope':<8}")
print("=" * 75)

for b in range(1, 8):
    d = data['data_2k'][str(b)]
    Ts = np.array(d['T'])
    fids = np.array(d['FID'])
    n = len(Ts)
    ss_tot = np.sum((fids - np.mean(fids)) ** 2)
    
    if b == 1:
        new_fits[str(b)] = None
        print(f"  B{b:<5} {BLUR_SIGMAS[str(b)]:<5} {'SKIP (noise)':<15}")
        continue
    
    pr = BUCKET_PRIORS[b]
    results = {}
    
    # --- Standard sigmoid with good initial conditions ---
    try:
        p0 = [pr['fid_base'], pr['fid_plat'], pr['k'], pr['t_mid']]
        bounds_lo = [pr['fid_plat'] - 5, pr['fid_plat'] - 5, pr['k_bounds'][0], pr['tmid_bounds'][0]]
        bounds_hi = [pr['fid_base'] * 1.5, pr['fid_plat'] + 5, pr['k_bounds'][1], pr['tmid_bounds'][1]]
        popt, _ = curve_fit(sigmoid, Ts, fids, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000)
        pred = sigmoid(Ts, *popt)
        ss_res = np.sum((fids - pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / n)
        results['sigmoid'] = {'popt': popt, 'r2': r2, 'rmse': rmse}
        print(f"  B{b:<5} {BLUR_SIGMAS[str(b)]:<5} {'sigmoid':<15} {r2:<8.4f} {rmse:<8.2f} "
              f"{popt[2]:<8.1f} {popt[3]:<8.4f} {'—':<8}")
    except Exception as e:
        print(f"  B{b} sigmoid FAILED: {e}")
    
    # --- Sigmoid with sloped upper plateau ---
    try:
        p0s = [pr['fid_base'], pr['fid_plat'], pr['k'], pr['t_mid'], pr['slope']]
        bounds_lo_s = [pr['fid_plat'] - 5, pr['fid_plat'] - 5, pr['k_bounds'][0], pr['tmid_bounds'][0], -50]
        bounds_hi_s = [pr['fid_base'] * 2, pr['fid_plat'] + 5, pr['k_bounds'][1], pr['tmid_bounds'][1], 200]
        popt_s, _ = curve_fit(sigmoid_slope, Ts, fids, p0=p0s, bounds=(bounds_lo_s, bounds_hi_s), maxfev=20000)
        pred_s = sigmoid_slope(Ts, *popt_s)
        ss_res_s = np.sum((fids - pred_s) ** 2)
        r2_s = 1 - ss_res_s / ss_tot
        rmse_s = np.sqrt(ss_res_s / n)
        results['sigmoid_slope'] = {'popt': popt_s, 'r2': r2_s, 'rmse': rmse_s}
        print(f"  {'':8}{'':5} {'sigmoid+slope':<15} {r2_s:<8.4f} {rmse_s:<8.2f} "
              f"{popt_s[2]:<8.1f} {popt_s[3]:<8.4f} {popt_s[4]:<8.1f}")
    except Exception as e:
        print(f"  B{b} sigmoid+slope FAILED: {e}")
    
    # Pick the better model
    if results:
        best_name = max(results, key=lambda m: results[m]['r2'])
        best = results[best_name]
        
        if best_name == 'sigmoid':
            p = best['popt']
            new_fits[str(b)] = {
                'fid_base': round(float(p[0]), 2), 'fid_plat': round(float(p[1]), 2),
                'k': round(float(p[2]), 1), 't_mid': round(float(p[3]), 4),
                'r2': round(best['r2'], 4), 'rmse': round(best['rmse'], 2),
                'model': 'sigmoid', 'n_points': n,
            }
        else:
            p = best['popt']
            new_fits[str(b)] = {
                'fid_base': round(float(p[0]), 2), 'fid_plat': round(float(p[1]), 2),
                'k': round(float(p[2]), 1), 't_mid': round(float(p[3]), 4),
                'slope': round(float(p[4]), 1),
                'r2': round(best['r2'], 4), 'rmse': round(best['rmse'], 2),
                'model': 'sigmoid_slope', 'n_points': n,
            }
        
        # Also store both for comparison
        new_fits[str(b)]['alt_r2'] = {m: round(results[m]['r2'], 4) for m in results}

data['fits_2k'] = new_fits
with open(ANALYSIS_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print("\n=== t_mid vs σ (transition point vs blur level) ===")
for b in range(2, 8):
    f = new_fits.get(str(b))
    if f:
        print(f"  B{b} σ={BLUR_SIGMAS[str(b)]}: t_mid={f['t_mid']:.3f}, k={f['k']:.0f}, "
              f"R²={f['r2']:.3f} ({f['model']})")
