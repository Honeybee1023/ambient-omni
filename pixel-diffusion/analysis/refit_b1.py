"""Add B1 fit to the JSON. Even if it's mostly noise, fit something."""

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
import warnings
warnings.filterwarnings('ignore')

ANALYSIS_PATH = f"{AMBIENT_BASE}/generated/celeba_2k_analysis.json"
with open(ANALYSIS_PATH) as f:
    data = json.load(f)

def sigmoid_slope(T, fid_base, fid_plat, k, t_mid, slope):
    return fid_plat + (fid_base + slope * T - fid_plat) / (1.0 + np.exp(k * (T - t_mid)))

d = data['data_2k']['1']
Ts = np.array(d['T'])
fids = np.array(d['FID'])
ss_tot = np.sum((fids - np.mean(fids)) ** 2)

# Try sigmoid_slope with gentle bounds
p0 = [27.2, 26.2, 5.0, 0.7, -1.0]
bounds = ([25, 24, 1, 0.3, -10], [29, 28, 50, 1.05, 10])
popt, _ = curve_fit(sigmoid_slope, Ts, fids, p0=p0, bounds=bounds, maxfev=20000)
pred = sigmoid_slope(Ts, *popt)
ss_res = np.sum((fids - pred) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(ss_res / len(Ts))

print(f"B1 sigmoid_slope fit:")
print(f"  fid_base={popt[0]:.2f}, fid_plat={popt[1]:.2f}, k={popt[2]:.1f}, t_mid={popt[3]:.3f}, slope={popt[4]:.1f}")
print(f"  R²={r2:.4f}, RMSE={rmse:.2f}")

data['fits_2k']['1'] = {
    'fid_base': round(float(popt[0]), 2),
    'fid_plat': round(float(popt[1]), 2),
    'k': round(float(popt[2]), 1),
    't_mid': round(float(popt[3]), 4),
    'slope': round(float(popt[4]), 1),
    'r2': round(r2, 4),
    'rmse': round(rmse, 2),
    'model': 'sigmoid_slope',
    'n_points': len(Ts),
}

with open(ANALYSIS_PATH, 'w') as f:
    json.dump(data, f, indent=2)
print("Updated JSON with B1 fit.")
