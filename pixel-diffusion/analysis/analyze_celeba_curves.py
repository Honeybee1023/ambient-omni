#!/usr/bin/env python3
"""
analyze_celeba_curves.py — v2 (double sigmoid)
Two competing effects: corruption damage (decreasing) + data scarcity (increasing).
Natural minimum = optimal T. Reads CelebA 2-domain FID metrics, fits curves,
extracts parameters, collects independence results. Outputs JSON for Jupyter.
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, glob, re, os, sys
import numpy as np
from scipy.optimize import curve_fit

BLUR_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 8.0}
FID_NOISE_FLOOR = 3.0

# === Data I/O ===
def read_fid(filepath):
    d = json.load(open(filepath))
    if 'fid_score' in d: return float(d['fid_score'])
    elif 'fid' in d: return float(d['fid'])
    else: raise KeyError(f"No fid key. Keys: {list(d.keys())}")

def collect_celeba_fid(kimg=1000):
    pattern = f'{AMBIENT_BASE}/generated/metrics_celeba_2d_b*_T*_{kimg}kimg.json'
    files = sorted(glob.glob(pattern))
    by_bucket = {}
    for f in files:
        basename = os.path.basename(f)
        m = re.match(rf'metrics_celeba_2d_b(\d+)_T(\w+)_{kimg}kimg\.json', basename)
        if not m: continue
        bucket = int(m.group(1))
        t_suffix = m.group(2)
        if len(t_suffix) == 3: t_val = int(t_suffix) / 100.0
        elif len(t_suffix) == 4: t_val = int(t_suffix) / 1000.0
        else: continue
        try:
            by_bucket.setdefault(bucket, []).append((t_val, read_fid(f)))
        except Exception as e:
            print(f'ERROR reading {basename}: {e}')
    for b in by_bucket:
        by_bucket[b] = sorted(by_bucket[b])
    return by_bucket

# === Functional forms ===
def double_sigmoid(T, C, A, k1, T1, B, k2, T2):
    """Two competing effects with natural minimum.
    C = floor, A = corruption amplitude, B = scarcity amplitude.
    Damage drops around T1, scarcity rises around T2.
    Minimum occurs between T1 and T2."""
    damage = A / (1.0 + np.exp(k1 * (T - T1)))
    scarcity = B / (1.0 + np.exp(-k2 * (T - T2)))
    return C + damage + scarcity

def sigmoid_only(T, fid_base, fid_plat, k, t_mid):
    """Simple flipped sigmoid (for comparison)."""
    return fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))

def fit_bucket(T_arr, FID_arr, bucket_num):
    """Fit double sigmoid and simple sigmoid, compare."""
    fid_min = float(np.min(FID_arr))
    fid_at_0 = float(FID_arr[T_arr == 0.0][0]) if np.any(T_arr == 0.0) else float(FID_arr[0])
    fid_at_1 = float(FID_arr[T_arr == 1.0][0]) if np.any(T_arr == 1.0) else float(FID_arr[-1])

    ss_tot = float(np.sum((FID_arr - np.mean(FID_arr))**2))

    # --- Simple sigmoid (for comparison R²) ---
    r2_sig = -999.0
    try:
        p0 = [fid_min, fid_at_0, 30.0, 0.9]
        bounds = ([25, 30, 1, 0.2], [55, 250, 300, 1.0])
        popt_s, _ = curve_fit(sigmoid_only, T_arr, FID_arr, p0=p0, bounds=bounds, maxfev=10000)
        r2_sig = 1.0 - np.sum((FID_arr - sigmoid_only(T_arr, *popt_s))**2) / ss_tot if ss_tot > 0 else 0
    except:
        pass

    # --- Double sigmoid ---
    r2_ds = -999.0
    popt_ds = None
    # Multiple initial guesses to avoid local minima
    best_cost = np.inf
    init_guesses = [
        # C, A, k1, T1, B, k2, T2
        [fid_min - 1, fid_at_0 - fid_min + 1, 40, 0.90, fid_at_1 - fid_min + 1, 80, 0.99],
        [fid_min - 1, fid_at_0 - fid_min + 1, 30, 0.85, fid_at_1 - fid_min + 1, 50, 0.98],
        [fid_min,     fid_at_0 - fid_min,     50, 0.92, fid_at_1 - fid_min + 2, 100, 0.995],
        [fid_min - 2, fid_at_0 - fid_min + 2, 20, 0.80, 3,                      60, 0.97],
    ]
    bounds_lo = [20, 0.5, 3, 0.2, 0.1, 3, 0.90]
    bounds_hi = [50, 200, 300, 0.98, 30, 300, 1.005]

    for p0 in init_guesses:
        try:
            # Clip p0 to bounds
            p0_clipped = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, bounds_lo, bounds_hi)]
            popt, _ = curve_fit(double_sigmoid, T_arr, FID_arr, p0=p0_clipped,
                                bounds=(bounds_lo, bounds_hi), maxfev=20000)
            cost = np.sum((FID_arr - double_sigmoid(T_arr, *popt))**2)
            if cost < best_cost:
                best_cost = cost
                popt_ds = popt
        except:
            pass

    if popt_ds is not None:
        r2_ds = 1.0 - best_cost / ss_tot if ss_tot > 0 else 0

    if popt_ds is None:
        return {'form': 'FAILED', 'r2': None, 'params': None,
                'r2_sigmoid': round(float(r2_sig), 4) if r2_sig > -900 else None}

    # --- Extract quantities from double sigmoid fit ---
    T_fine = np.linspace(0, 1, 10001)
    FID_fine = double_sigmoid(T_fine, *popt_ds)

    C, A, k1, T1, B, k2, T2 = popt_ds

    # Fit-based argmin
    fit_argmin_idx = np.argmin(FID_fine)
    fit_argmin_T = float(T_fine[fit_argmin_idx])
    fit_argmin_FID = float(FID_fine[fit_argmin_idx])

    # Fit-based threshold (lowest T where FID within noise_floor of T=1 value)
    fit_fid_at_1 = float(FID_fine[-1])
    within_noise = FID_fine <= fit_fid_at_1 + FID_NOISE_FLOOR
    fit_thresh_T = float(T_fine[np.argmax(within_noise)]) if np.any(within_noise) else 1.0

    # Raw-data argmin
    raw_idx = np.argmin(FID_arr)
    raw_argmin_T = float(T_arr[raw_idx])
    raw_argmin_FID = float(FID_arr[raw_idx])

    # Raw-data threshold
    within_raw = FID_arr <= fid_at_1 + FID_NOISE_FLOOR
    raw_thresh_T = float(T_arr[within_raw][0]) if np.any(within_raw) else 1.0

    # R² on transition+dip region only (T >= 0.8)
    dip_mask = T_arr >= 0.80
    if np.sum(dip_mask) >= 3:
        ss_res_dip = np.sum((FID_arr[dip_mask] - double_sigmoid(T_arr[dip_mask], *popt_ds))**2)
        ss_tot_dip = np.sum((FID_arr[dip_mask] - np.mean(FID_arr[dip_mask]))**2)
        r2_dip = 1.0 - ss_res_dip / ss_tot_dip if ss_tot_dip > 0 else 0
    else:
        r2_dip = None

    params = {
        'C': round(float(C), 2), 'A': round(float(A), 2),
        'k1': round(float(k1), 1), 'T1': round(float(T1), 4),
        'B': round(float(B), 2), 'k2': round(float(k2), 1),
        'T2': round(float(T2), 4),
        'fit_argmin_T': round(fit_argmin_T, 4),
        'fit_argmin_FID': round(fit_argmin_FID, 2),
        'fit_threshold_T': round(fit_thresh_T, 4),
        'fit_fid_at_T1': round(fit_fid_at_1, 2),
        'raw_argmin_T': round(raw_argmin_T, 4),
        'raw_argmin_FID': round(raw_argmin_FID, 2),
        'raw_threshold_T': round(raw_thresh_T, 4),
        'raw_fid_at_T1': round(fid_at_1, 2),
    }

    # Dense curve for plotting
    plot_T = np.linspace(0, 1, 201).tolist()
    plot_FID = double_sigmoid(np.array(plot_T), *popt_ds).tolist()

    # Also compute the two components for visualization
    plot_damage = [float(C + A / (1 + np.exp(k1 * (t - T1)))) for t in plot_T]
    plot_scarcity = [float(C + B / (1 + np.exp(-k2 * (t - T2)))) for t in plot_T]

    return {
        'form': 'double_sigmoid',
        'r2': round(float(r2_ds), 4),
        'r2_sigmoid': round(float(r2_sig), 4) if r2_sig > -900 else None,
        'r2_dip_region': round(float(r2_dip), 4) if r2_dip is not None else None,
        'params': params,
        'plot_curve': {'T': plot_T, 'FID': [round(f, 2) for f in plot_FID],
                       'damage_component': [round(f, 2) for f in plot_damage],
                       'scarcity_component': [round(f, 2) for f in plot_scarcity]},
    }

# === Independence ===
def collect_independence():
    tests = [
        'celeba_indep_baseline',
        'celeba_indep_all_tmid', 'celeba_indep_lowblur_tmid', 'celeba_indep_highblur_tmid',
        'celeba_indep_all_argmin',
        'celeba_indep_argmin_low', 'celeba_indep_argmin_high',
        'celeba_indep_thresh_all', 'celeba_indep_thresh_low', 'celeba_indep_thresh_high',
    ]
    results = {}
    for name in tests:
        f = f'{AMBIENT_BASE}/generated/metrics_{name}_1000kimg.json'
        if os.path.exists(f):
            try: results[name] = round(read_fid(f), 2)
            except: pass
    return results

def analyze_independence(indep):
    analysis = {}
    baseline = indep.get('celeba_indep_baseline')
    if baseline is None: return analysis

    for test_name, keys in [
        ('tmid', ('celeba_indep_all_tmid', 'celeba_indep_lowblur_tmid', 'celeba_indep_highblur_tmid')),
        ('argmin', ('celeba_indep_all_argmin', 'celeba_indep_argmin_low', 'celeba_indep_argmin_high')),
        ('threshold', ('celeba_indep_thresh_all', 'celeba_indep_thresh_low', 'celeba_indep_thresh_high')),
    ]:
        k_all, k_low, k_high = keys
        entry = {'baseline': baseline}
        if k_all in indep: entry['all'] = indep[k_all]; entry['delta_all'] = round(indep[k_all] - baseline, 2)
        if k_low in indep: entry['low'] = indep[k_low]; entry['delta_low'] = round(indep[k_low] - baseline, 2)
        if k_high in indep: entry['high'] = indep[k_high]; entry['delta_high'] = round(indep[k_high] - baseline, 2)
        if all(k in indep for k in [k_all, k_low, k_high]):
            predicted = baseline + (indep[k_low] - baseline) + (indep[k_high] - baseline)
            entry['predicted_additive'] = round(predicted, 2)
            entry['actual_minus_predicted'] = round(indep[k_all] - predicted, 2)
        if entry.keys() - {'baseline'}:
            analysis[test_name] = entry
    return analysis

# === Main ===
if __name__ == '__main__':
    kimg = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"=== CelebA Curve Analysis v2 — Double Sigmoid ({kimg} kimg) ===\n")

    data = collect_celeba_fid(kimg=kimg)
    print(f"Found data for buckets: {sorted(data.keys())} ({sum(len(v) for v in data.values())} total points)\n")

    results = {'kimg': kimg, 'blur_sigmas': {str(k): v for k, v in BLUR_SIGMAS.items()},
               'raw_data': {}, 'fits': {}, 'independence': {}, 'independence_analysis': {},
               'meta_curves': {}}

    # Per-bucket fits
    header = f"{'Bkt':>3s} {'σ':>4s} {'R²':>6s} {'R²dip':>6s} {'R²sig':>6s}  {'C':>5s} {'A':>5s} {'k1':>5s} {'T1':>5s} {'B':>4s} {'k2':>5s} {'T2':>5s}  {'argT':>5s} {'argF':>5s} {'thrT':>5s}"
    print(header)
    print("-" * len(header))

    for bucket in sorted(data.keys()):
        points = data[bucket]
        T_arr = np.array([p[0] for p in points])
        FID_arr = np.array([p[1] for p in points])
        sigma = BLUR_SIGMAS.get(bucket, -1)

        results['raw_data'][str(bucket)] = {
            'T': [round(float(t), 4) for t in T_arr],
            'FID': [round(float(f), 2) for f in FID_arr],
            'sigma_blur': sigma
        }

        fit = fit_bucket(T_arr, FID_arr, bucket)
        results['fits'][str(bucket)] = fit

        p = fit['params']
        if p:
            r2d = fit['r2_dip_region'] or 0
            r2s = fit['r2_sigmoid'] or 0
            print(f"{bucket:>3d} {sigma:>4.1f} {fit['r2']:>6.3f} {r2d:>6.3f} {r2s:>6.3f}  "
                  f"{p['C']:>5.1f} {p['A']:>5.1f} {p['k1']:>5.1f} {p['T1']:>5.4f} "
                  f"{p['B']:>4.1f} {p['k2']:>5.1f} {p['T2']:>5.4f}  "
                  f"{p['fit_argmin_T']:>5.3f} {p['fit_argmin_FID']:>5.1f} {p['fit_threshold_T']:>5.3f}")
        else:
            print(f"{bucket:>3d} {sigma:>4.1f} *** FIT FAILED ***")

    # Independence
    print(f"\n{'='*60}\nINDEPENDENCE TEST RESULTS\n{'='*60}")
    indep = collect_independence()
    results['independence'] = indep
    baseline = indep.get('celeba_indep_baseline')
    if baseline:
        print(f"Baseline: {baseline:.1f}")
        for test, label in [('tmid', 'T_mid (flawed)'), ('argmin', 'Argmin'), ('threshold', 'Threshold')]:
            prefix = {'tmid': 'celeba_indep_', 'argmin': 'celeba_indep_argmin_',
                      'threshold': 'celeba_indep_thresh_'}[test]
            # Print whatever we have
            if test == 'tmid':
                keys = [('All', 'celeba_indep_all_tmid'), ('Low', 'celeba_indep_lowblur_tmid'),
                        ('High', 'celeba_indep_highblur_tmid')]
            elif test == 'argmin':
                keys = [('All', 'celeba_indep_all_argmin'), ('Low', 'celeba_indep_argmin_low'),
                        ('High', 'celeba_indep_argmin_high')]
            else:
                keys = [('All', 'celeba_indep_thresh_all'), ('Low', 'celeba_indep_thresh_low'),
                        ('High', 'celeba_indep_thresh_high')]
            vals = [(role, indep.get(k)) for role, k in keys]
            if any(v is not None for _, v in vals):
                print(f"\n  {label}:")
                for role, v in vals:
                    if v is not None:
                        print(f"    {role:>5s}: {v:>6.1f} (Δ={v-baseline:+.1f})")
                    else:
                        print(f"    {role:>5s}: PENDING")

    analysis = analyze_independence(indep)
    results['independence_analysis'] = analysis

    # Meta-curves
    print(f"\n{'='*60}\nMETA-CURVES\n{'='*60}")
    mc = {k: [] for k in ['sigma', 'T1', 'T2', 'fit_argmin_T', 'fit_argmin_FID',
                           'fit_threshold_T', 'raw_argmin_T', 'raw_argmin_FID',
                           'raw_threshold_T', 'A', 'B', 'k1', 'k2', 'C']}
    for bucket in sorted(data.keys()):
        fit = results['fits'][str(bucket)]
        if fit['params'] is None: continue
        p = fit['params']
        mc['sigma'].append(BLUR_SIGMAS[bucket])
        for key in mc:
            if key != 'sigma':
                mc[key].append(p.get(key, 0))
    results['meta_curves'] = mc

    print(f"{'σ':>4s} {'T1(dmg)':>7s} {'T2(scar)':>8s} {'fitArgT':>7s} {'fitArgF':>7s} {'rawArgT':>7s} {'rawArgF':>7s} {'thrT':>5s} {'A':>5s} {'B':>4s}")
    for i in range(len(mc['sigma'])):
        print(f"{mc['sigma'][i]:>4.1f} {mc['T1'][i]:>7.4f} {mc['T2'][i]:>8.4f} "
              f"{mc['fit_argmin_T'][i]:>7.4f} {mc['fit_argmin_FID'][i]:>7.1f} "
              f"{mc['raw_argmin_T'][i]:>7.4f} {mc['raw_argmin_FID'][i]:>7.1f} "
              f"{mc['fit_threshold_T'][i]:>5.3f} {mc['A'][i]:>5.1f} {mc['B'][i]:>4.1f}")

    outpath = f'{AMBIENT_BASE}/generated/celeba_curve_analysis_{kimg}kimg.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {outpath}")
