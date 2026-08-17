"""
Refit sigmoid curves to 2k kimg 2-domain data with all dense points.
Also tries alternative forms to see if anything fits better.
Updates celeba_2k_analysis.json with new fits.
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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

ANALYSIS_PATH = f"{AMBIENT_BASE}/generated/celeba_2k_analysis.json"

with open(ANALYSIS_PATH) as f:
    data = json.load(f)

BLUR_SIGMAS = {str(b): data['blur_sigmas'][str(b)] for b in range(1, 8)}

# === Model definitions ===

def sigmoid(T, fid_base, fid_plat, k, t_mid):
    """Standard sigmoid: high FID at low T, drops to plateau at high T."""
    return fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))

def richards(T, fid_base, fid_plat, k, t_mid, nu):
    """Generalized logistic (Richards curve) — asymmetric sigmoid."""
    return fid_base + (fid_plat - fid_base) / (1.0 + nu * np.exp(k * (T - t_mid))) ** (1.0 / nu)

def sigmoid_linear(T, fid_base, fid_plat, k, t_mid, slope):
    """Sigmoid + linear trend in the plateau region."""
    sig = fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))
    return sig + slope * (T - t_mid)

# === Fit each bucket ===

new_fits = {}
for b in range(1, 8):
    d = data['data_2k'][str(b)]
    Ts = np.array(d['T'])
    fids = np.array(d['FID'])
    n = len(Ts)
    
    print(f"\n=== Bucket {b} (σ={BLUR_SIGMAS[str(b)]}, n={n}) ===")
    
    if b == 1:
        print(f"  FID range: {fids.min():.2f} – {fids.max():.2f} (spread={fids.max()-fids.min():.2f})")
        print(f"  Skipping fit — B1 is noise (spread < 2 FID)")
        new_fits[str(b)] = None
        continue
    
    results = {}
    
    # --- Sigmoid ---
    try:
        fid_high = np.mean(fids[Ts < 0.5])
        fid_low = np.mean(fids[Ts > 0.95])
        p0 = [fid_high, fid_low, 20.0, 0.9]
        bounds = ([0, 0, 1, 0.5], [300, 50, 200, 1.05])
        popt, _ = curve_fit(sigmoid, Ts, fids, p0=p0, bounds=bounds, maxfev=10000)
        pred = sigmoid(Ts, *popt)
        ss_res = np.sum((fids - pred) ** 2)
        ss_tot = np.sum((fids - np.mean(fids)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results['sigmoid'] = {
            'params': {'fid_base': popt[0], 'fid_plat': popt[1], 'k': popt[2], 't_mid': popt[3]},
            'r2': r2, 'rmse': np.sqrt(ss_res / n)
        }
        print(f"  Sigmoid:      R²={r2:.4f}  RMSE={results['sigmoid']['rmse']:.2f}  "
              f"k={popt[2]:.1f}  t_mid={popt[3]:.3f}")
    except Exception as e:
        print(f"  Sigmoid FAILED: {e}")
    
    # --- Richards (asymmetric) ---
    try:
        p0_r = [fid_high, fid_low, 20.0, 0.9, 1.0]
        bounds_r = ([0, 0, 1, 0.5, 0.01], [300, 50, 200, 1.05, 10.0])
        popt_r, _ = curve_fit(richards, Ts, fids, p0=p0_r, bounds=bounds_r, maxfev=10000)
        pred_r = richards(Ts, *popt_r)
        ss_res_r = np.sum((fids - pred_r) ** 2)
        r2_r = 1 - ss_res_r / ss_tot
        results['richards'] = {
            'params': {'fid_base': popt_r[0], 'fid_plat': popt_r[1], 'k': popt_r[2], 
                       't_mid': popt_r[3], 'nu': popt_r[4]},
            'r2': r2_r, 'rmse': np.sqrt(ss_res_r / n)
        }
        print(f"  Richards:     R²={r2_r:.4f}  RMSE={results['richards']['rmse']:.2f}  "
              f"k={popt_r[2]:.1f}  t_mid={popt_r[3]:.3f}  nu={popt_r[4]:.2f}")
    except Exception as e:
        print(f"  Richards FAILED: {e}")
    
    # --- Sigmoid + linear ---
    try:
        p0_sl = [fid_high, fid_low, 20.0, 0.9, 0.0]
        bounds_sl = ([0, 0, 1, 0.5, -100], [300, 50, 200, 1.05, 100])
        popt_sl, _ = curve_fit(sigmoid_linear, Ts, fids, p0=p0_sl, bounds=bounds_sl, maxfev=10000)
        pred_sl = sigmoid_linear(Ts, *popt_sl)
        ss_res_sl = np.sum((fids - pred_sl) ** 2)
        r2_sl = 1 - ss_res_sl / ss_tot
        results['sigmoid_linear'] = {
            'params': {'fid_base': popt_sl[0], 'fid_plat': popt_sl[1], 'k': popt_sl[2],
                       't_mid': popt_sl[3], 'slope': popt_sl[4]},
            'r2': r2_sl, 'rmse': np.sqrt(ss_res_sl / n)
        }
        print(f"  Sig+Linear:   R²={r2_sl:.4f}  RMSE={results['sigmoid_linear']['rmse']:.2f}  "
              f"slope={popt_sl[4]:.1f}")
    except Exception as e:
        print(f"  Sig+Linear FAILED: {e}")
    
    # Pick best
    best_model = max(results.keys(), key=lambda m: results[m]['r2'])
    best = results[best_model]
    print(f"  >>> Best: {best_model} (R²={best['r2']:.4f})")
    
    # Store sigmoid params for backward compat with Cell 3
    if 'sigmoid' in results:
        sp = results['sigmoid']['params']
        new_fits[str(b)] = {
            'fid_base': round(sp['fid_base'], 2),
            'fid_plat': round(sp['fid_plat'], 2),
            'k': round(sp['k'], 1),
            't_mid': round(sp['t_mid'], 4),
            'r2': round(results['sigmoid']['r2'], 4),
            'n_points': n,
            'alt_fits': {m: {'r2': round(r['r2'], 4), 'rmse': round(r['rmse'], 2)} 
                         for m, r in results.items()}
        }

# Save
data['fits_2k'] = new_fits
with open(ANALYSIS_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print("\n=== SUMMARY ===")
print(f"{'Bucket':<8} {'σ':<5} {'Sigmoid R²':<12} {'Richards R²':<12} {'Sig+Lin R²':<12} {'Best':<12} {'t_mid':<8} {'k':<6}")
for b in range(2, 8):
    fits = new_fits.get(str(b))
    if fits and 'alt_fits' in fits:
        af = fits['alt_fits']
        best_m = max(af.keys(), key=lambda m: af[m]['r2'])
        print(f"  B{b:<5} {BLUR_SIGMAS[str(b)]:<5} "
              f"{af.get('sigmoid',{}).get('r2','—'):<12} "
              f"{af.get('richards',{}).get('r2','—'):<12} "
              f"{af.get('sigmoid_linear',{}).get('r2','—'):<12} "
              f"{best_m:<12} "
              f"{fits['t_mid']:<8} {fits['k']:<6}")

print("\nUpdated celeba_2k_analysis.json with new fits.")
