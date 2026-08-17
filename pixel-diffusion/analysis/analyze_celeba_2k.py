"""
CelebA 2-domain analysis at 2k kimg.
Reads FID from all metrics JSONs, fits sigmoids, saves results.
Plotting happens in Jupyter.
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, os
import numpy as np
from scipy.optimize import curve_fit

BLUR_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 8.0}
T_SUFFIXES = ['000','0125','025','0375','050','0625','075','080','085','090','095','097','099','100']

def suffix_to_T(s):
    return int(s) / (100.0 if len(s) == 3 else 1000.0)

def load_all_fid(kimg_label):
    data = {}
    for b in range(1, 8):
        points = []
        for ts in T_SUFFIXES:
            T = suffix_to_T(ts)
            path = f'{AMBIENT_BASE}/generated/metrics_celeba_2d_b{b}_T{ts}_{kimg_label}.json'
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                points.append((T, d['fid_score']))
            else:
                print(f'MISSING: {path}')
        data[b] = sorted(points)
    return data

def flipped_sigmoid(T, fid_base, fid_plat, k, t_mid):
    return fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))

def fit_sigmoid(T_arr, fid_arr):
    try:
        p0 = [np.min(fid_arr), np.max(fid_arr), 30.0, 0.8]
        bounds = ([0, 0, 0.1, 0.0], [200, 300, 500, 1.0])
        popt, _ = curve_fit(flipped_sigmoid, T_arr, fid_arr, p0=p0, bounds=bounds, maxfev=10000)
        pred = flipped_sigmoid(T_arr, *popt)
        ss_res = np.sum((fid_arr - pred)**2)
        ss_tot = np.sum((fid_arr - fid_arr.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
        return {'fid_base': float(popt[0]), 'fid_plat': float(popt[1]),
                'k': float(popt[2]), 't_mid': float(popt[3]), 'r2': float(r2)}
    except Exception as e:
        print(f'Sigmoid fit failed: {e}')
        return None

# === Load data ===
data_2k = load_all_fid('2000kimg')
data_1k = load_all_fid('1000kimg')

# === Fit sigmoids to 2k data ===
fits_2k = {}
print('=== SIGMOID FITS (2k kimg) ===')
print(f'{"Bucket":>7s}  {"σ":>4s}  {"FID_base":>9s}  {"FID_plat":>9s}  {"k":>7s}  {"T_mid":>6s}  {"R²":>6s}')
print('-' * 55)
for b in range(1, 8):
    Ts = np.array([p[0] for p in data_2k[b]])
    fids = np.array([p[1] for p in data_2k[b]])
    fit = fit_sigmoid(Ts, fids)
    if fit:
        fits_2k[b] = fit
        print(f'B{b:>5d}  {BLUR_SIGMAS[b]:>4.1f}  {fit["fid_base"]:>9.2f}  {fit["fid_plat"]:>9.2f}  {fit["k"]:>7.1f}  {fit["t_mid"]:>6.3f}  {fit["r2"]:>6.3f}')

# === Summary ===
print('\n=== BEST T PER BUCKET (2k kimg) ===')
print(f'{"Bucket":>7s}  {"σ":>4s}  {"Best T":>7s}  {"Best FID":>9s}  {"T=1 FID":>8s}  {"Δ":>6s}')
print('-' * 48)
for b in range(1, 8):
    Ts = np.array([p[0] for p in data_2k[b]])
    fids = np.array([p[1] for p in data_2k[b]])
    best_idx = np.argmin(fids)
    t1_fid = fids[-1]
    delta = t1_fid - fids[best_idx]
    print(f'B{b:>5d}  {BLUR_SIGMAS[b]:>4.1f}  {Ts[best_idx]:>7.3f}  {fids[best_idx]:>9.2f}  {t1_fid:>8.2f}  {delta:>+6.2f}')

t1_all = [data_2k[b][-1][1] for b in range(1, 8)]
print(f'\nT=1 noise floor: {min(t1_all):.1f} – {max(t1_all):.1f} (spread: {max(t1_all)-min(t1_all):.1f})')

# === Save everything ===
out = {
    'blur_sigmas': {str(b): BLUR_SIGMAS[b] for b in range(1, 8)},
    'fits_2k': {str(b): v for b, v in fits_2k.items()},
    'data_2k': {str(b): {'T': [p[0] for p in data_2k[b]], 'FID': [p[1] for p in data_2k[b]]} for b in range(1, 8)},
    'data_1k': {str(b): {'T': [p[0] for p in data_1k[b]], 'FID': [p[1] for p in data_1k[b]]} for b in range(1, 8)},
}
outpath = f'{AMBIENT_BASE}/generated/celeba_2k_analysis.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {outpath}')
