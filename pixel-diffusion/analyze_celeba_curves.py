#!/usr/bin/env python3
"""
analyze_celeba_curves.py
Reads CelebA 2-domain FID results, fits sigmoid+bump curves,
extracts parameters (argmin, threshold, T_mid), collects independence
test results, computes meta-curves. Outputs JSON for Jupyter.
"""
import json, glob, re, os, sys
import numpy as np
from scipy.optimize import curve_fit

# === Constants ===
BLUR_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 8.0}
FID_NOISE_FLOOR = 3.0  # seed variance in FID

# === Data collection ===
def read_fid(filepath):
    """Read FID from a metric JSON, handling different key structures."""
    d = json.load(open(filepath))
    if 'fid_score' in d:
        return float(d['fid_score'])
    elif 'fid' in d:
        return float(d['fid'])
    else:
        raise KeyError(f"No fid_score or fid key. Keys: {list(d.keys())}")

def collect_celeba_fid(kimg=1000):
    """Read all celeba 2-domain FID metrics."""
    pattern = f'/data/scratch/honjar/generated/metrics_celeba_2d_b*_T*_{kimg}kimg.json'
    files = sorted(glob.glob(pattern))
    by_bucket = {}
    for f in files:
        basename = os.path.basename(f)
        m = re.match(rf'metrics_celeba_2d_b(\d+)_T(\w+)_{kimg}kimg\.json', basename)
        if not m:
            continue
        bucket = int(m.group(1))
        t_suffix = m.group(2)
        if len(t_suffix) == 3:
            t_val = int(t_suffix) / 100.0
        elif len(t_suffix) == 4:
            t_val = int(t_suffix) / 1000.0
        else:
            continue
        try:
            fid = read_fid(f)
            by_bucket.setdefault(bucket, []).append((t_val, fid))
        except Exception as e:
            print(f'ERROR reading {basename}: {e}')
    for b in by_bucket:
        by_bucket[b] = sorted(by_bucket[b])
    return by_bucket

# === Functional forms ===
def sigmoid_only(T, fid_base, fid_plat, k, t_mid):
    """Flipped sigmoid: high at low T, low at high T."""
    return fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))

def sigmoid_plus_bump(T, fid_base, fid_plat, k, t_mid, A, t_bump, w):
    """Sigmoid + Gaussian bump for the hump above plateau."""
    sig = fid_base + (fid_plat - fid_base) / (1.0 + np.exp(k * (T - t_mid)))
    bump = A * np.exp(-0.5 * ((T - t_bump) / w) ** 2)
    return sig + bump

def fit_bucket(T_arr, FID_arr, bucket_num):
    """Fit sigmoid and sigmoid+bump, pick the better one."""
    fid_min = float(np.min(FID_arr))
    low_T_mask = T_arr <= 0.25
    fid_plat_init = float(np.mean(FID_arr[low_T_mask])) if np.sum(low_T_mask) >= 2 else float(FID_arr[0])

    # --- Sigmoid only ---
    r2_sig = -999.0
    popt_sig = None
    try:
        p0 = [fid_min, fid_plat_init, 30.0, 0.9]
        bounds_lo = [25, 30, 1, 0.2]
        bounds_hi = [55, 250, 300, 1.0]
        popt_sig, _ = curve_fit(sigmoid_only, T_arr, FID_arr, p0=p0,
                                bounds=(bounds_lo, bounds_hi), maxfev=10000)
        pred = sigmoid_only(T_arr, *popt_sig)
        ss_res = np.sum((FID_arr - pred)**2)
        ss_tot = np.sum((FID_arr - np.mean(FID_arr))**2)
        r2_sig = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    except Exception as e:
        print(f"  Bucket {bucket_num} sigmoid fit failed: {e}")

    # --- Sigmoid + bump ---
    r2_full = -999.0
    popt_full = None
    try:
        # Estimate bump from sigmoid residuals
        A_init, t_bump_init = 5.0, 0.4
        if popt_sig is not None:
            resid = FID_arr - sigmoid_only(T_arr, *popt_sig)
            mid_mask = (T_arr > 0.1) & (T_arr < 0.85)
            if np.any(mid_mask) and np.max(resid[mid_mask]) > 2:
                A_init = float(np.max(resid[mid_mask]))
                t_bump_init = float(T_arr[mid_mask][np.argmax(resid[mid_mask])])

        p0 = [fid_min, fid_plat_init, 30.0, 0.9, A_init, t_bump_init, 0.15]
        bounds_lo = [25, 30, 1, 0.2, 0, 0.05, 0.03]
        bounds_hi = [55, 250, 300, 1.0, 150, 0.85, 0.50]
        popt_full, _ = curve_fit(sigmoid_plus_bump, T_arr, FID_arr, p0=p0,
                                 bounds=(bounds_lo, bounds_hi), maxfev=20000)
        pred = sigmoid_plus_bump(T_arr, *popt_full)
        ss_res = np.sum((FID_arr - pred)**2)
        ss_tot = np.sum((FID_arr - np.mean(FID_arr))**2)
        r2_full = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    except Exception as e:
        print(f"  Bucket {bucket_num} sigmoid+bump fit failed: {e}")

    # --- Pick winner: bump must improve R² meaningfully and have A > 2 ---
    use_bump = (popt_full is not None and r2_full > r2_sig + 0.01 and popt_full[4] > 2.0)

    if use_bump:
        popt = popt_full
        r2 = r2_full
        form = 'sigmoid+bump'
    elif popt_sig is not None:
        popt = popt_sig
        r2 = r2_sig
        form = 'sigmoid_only'
    else:
        return {'form': 'FAILED', 'r2': None, 'r2_sigmoid_only': None,
                'r2_sigmoid_bump': None, 'params': None}

    # --- Extract analytic quantities from the fit ---
    T_fine = np.linspace(0, 1, 10001)
    if form == 'sigmoid+bump':
        FID_fine = sigmoid_plus_bump(T_fine, *popt)
        params = dict(fid_base=popt[0], fid_plat=popt[1], k=popt[2],
                      t_mid=popt[3], A=popt[4], t_bump=popt[5], w=popt[6])
    else:
        FID_fine = sigmoid_only(T_fine, *popt)
        params = dict(fid_base=popt[0], fid_plat=popt[1], k=popt[2],
                      t_mid=popt[3], A=0.0, t_bump=0.0, w=0.0)

    # Argmin (best T)
    params['argmin_T'] = float(T_fine[np.argmin(FID_fine)])
    params['argmin_FID'] = float(np.min(FID_fine))

    # FID at T=1
    params['fid_at_T1'] = float(FID_fine[-1])

    # Threshold T: lowest T where FID <= baseline + noise_floor
    baseline_fid = float(FID_fine[-1])  # T=1 value from fit
    within_noise = FID_fine <= baseline_fid + FID_NOISE_FLOOR
    if np.any(within_noise):
        params['threshold_T'] = float(T_fine[np.argmax(within_noise)])
    else:
        params['threshold_T'] = 1.0

    # T_95: where sigmoid transition is 95% complete
    params['T_95'] = float(params['t_mid'] + 2.944 / params['k']) if params['k'] > 0.1 else 1.0

    # Hump peak (max FID above the plateau, only for sigmoid+bump)
    if form == 'sigmoid+bump' and params['A'] > 2:
        hump_mask = (T_fine > 0.05) & (T_fine < params['t_mid'] - 0.05)
        if np.any(hump_mask):
            params['hump_peak_T'] = float(T_fine[hump_mask][np.argmax(FID_fine[hump_mask])])
            params['hump_peak_FID'] = float(np.max(FID_fine[hump_mask]))
        else:
            params['hump_peak_T'] = 0.0
            params['hump_peak_FID'] = 0.0
    else:
        params['hump_peak_T'] = 0.0
        params['hump_peak_FID'] = 0.0

    # Round everything
    params = {k: round(float(v), 4) for k, v in params.items()}

    # Dense curve for plotting
    plot_T = np.linspace(0, 1, 201).tolist()
    if form == 'sigmoid+bump':
        plot_FID = sigmoid_plus_bump(np.array(plot_T), *popt).tolist()
    else:
        plot_FID = sigmoid_only(np.array(plot_T), *popt).tolist()

    return {
        'form': form,
        'r2': round(float(r2), 4),
        'r2_sigmoid_only': round(float(r2_sig), 4) if r2_sig > -900 else None,
        'r2_sigmoid_bump': round(float(r2_full), 4) if r2_full > -900 else None,
        'params': params,
        'plot_curve': {'T': plot_T, 'FID': [round(f, 2) for f in plot_FID]}
    }

# === Independence tests ===
def collect_independence():
    """Read all independence test FID metrics."""
    tests = [
        'celeba_indep_baseline',
        'celeba_indep_all_tmid', 'celeba_indep_lowblur_tmid', 'celeba_indep_highblur_tmid',
        'celeba_indep_all_argmin',
        'celeba_indep_argmin_low', 'celeba_indep_argmin_high',
        'celeba_indep_thresh_all', 'celeba_indep_thresh_low', 'celeba_indep_thresh_high',
    ]
    results = {}
    for name in tests:
        f = f'/data/scratch/honjar/generated/metrics_{name}_1000kimg.json'
        if os.path.exists(f):
            try:
                results[name] = read_fid(f)
            except Exception as e:
                print(f"  WARNING: couldn't read {name}: {e}")
    return results

def analyze_independence(indep):
    """Compute additivity tests from independence results."""
    analysis = {}
    baseline = indep.get('celeba_indep_baseline')
    if baseline is None:
        return analysis

    # T_mid test
    tmid_all = indep.get('celeba_indep_all_tmid')
    tmid_low = indep.get('celeba_indep_lowblur_tmid')
    tmid_high = indep.get('celeba_indep_highblur_tmid')
    if all(v is not None for v in [tmid_all, tmid_low, tmid_high]):
        d_low = tmid_low - baseline
        d_high = tmid_high - baseline
        predicted = baseline + d_low + d_high
        analysis['tmid'] = {
            'baseline': baseline, 'all': tmid_all,
            'low': tmid_low, 'high': tmid_high,
            'delta_low': round(d_low, 2), 'delta_high': round(d_high, 2),
            'predicted_additive': round(predicted, 2),
            'actual_minus_predicted': round(tmid_all - predicted, 2),
            'verdict': 'sub-additive' if tmid_all < predicted else 'super-additive'
        }

    # Argmin test
    arg_all = indep.get('celeba_indep_all_argmin')
    arg_low = indep.get('celeba_indep_argmin_low')
    arg_high = indep.get('celeba_indep_argmin_high')
    if arg_all is not None:
        entry = {'baseline': baseline, 'all': arg_all,
                 'delta_all': round(arg_all - baseline, 2)}
        if arg_low is not None and arg_high is not None:
            d_low = arg_low - baseline
            d_high = arg_high - baseline
            predicted = baseline + d_low + d_high
            entry.update({
                'low': arg_low, 'high': arg_high,
                'delta_low': round(d_low, 2), 'delta_high': round(d_high, 2),
                'predicted_additive': round(predicted, 2),
                'actual_minus_predicted': round(arg_all - predicted, 2)
            })
        analysis['argmin'] = entry

    # Threshold test
    th_all = indep.get('celeba_indep_thresh_all')
    th_low = indep.get('celeba_indep_thresh_low')
    th_high = indep.get('celeba_indep_thresh_high')
    if any(v is not None for v in [th_all, th_low, th_high]):
        entry = {'baseline': baseline}
        if th_all is not None: entry['all'] = th_all; entry['delta_all'] = round(th_all - baseline, 2)
        if th_low is not None: entry['low'] = th_low; entry['delta_low'] = round(th_low - baseline, 2)
        if th_high is not None: entry['high'] = th_high; entry['delta_high'] = round(th_high - baseline, 2)
        if all(v is not None for v in [th_all, th_low, th_high]):
            predicted = baseline + (th_low - baseline) + (th_high - baseline)
            entry['predicted_additive'] = round(predicted, 2)
            entry['actual_minus_predicted'] = round(th_all - predicted, 2)
        analysis['threshold'] = entry

    return analysis

# === Main ===
if __name__ == '__main__':
    kimg = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print(f"=== CelebA Curve Analysis ({kimg} kimg) ===\n")
    data = collect_celeba_fid(kimg=kimg)
    print(f"Found data for buckets: {sorted(data.keys())} ({sum(len(v) for v in data.values())} total points)\n")

    results = {'kimg': kimg, 'blur_sigmas': {str(k): v for k, v in BLUR_SIGMAS.items()},
               'raw_data': {}, 'fits': {}, 'independence': {}, 'independence_analysis': {},
               'meta_curves': {}}

    # --- Per-bucket fits ---
    header = f"{'Bkt':>3s} {'σ':>4s} {'Pts':>3s} {'Form':>13s} {'R²':>6s} {'R²sig':>6s} {'R²s+b':>6s} {'T_mid':>5s} {'A':>5s} {'argT':>5s} {'argFID':>6s} {'thrT':>5s} {'humpT':>5s}"
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
            r2s = fit['r2_sigmoid_only'] or 0
            r2sb = fit['r2_sigmoid_bump'] or 0
            print(f"{bucket:>3d} {sigma:>4.1f} {len(points):>3d} {fit['form']:>13s} "
                  f"{fit['r2']:>6.3f} {r2s:>6.3f} {r2sb:>6.3f} "
                  f"{p['t_mid']:>5.3f} {p['A']:>5.1f} {p['argmin_T']:>5.3f} "
                  f"{p['argmin_FID']:>6.1f} {p['threshold_T']:>5.3f} {p['hump_peak_T']:>5.3f}")
        else:
            print(f"{bucket:>3d} {sigma:>4.1f} {len(points):>3d} *** FIT FAILED ***")

    # --- Independence ---
    print(f"\n{'='*60}\nINDEPENDENCE TEST RESULTS\n{'='*60}")
    indep = collect_independence()
    results['independence'] = {k: round(v, 2) for k, v in indep.items()}

    baseline = indep.get('celeba_indep_baseline')
    if baseline:
        print(f"\nBaseline (all T=1): FID = {baseline:.1f}")

        if 'celeba_indep_all_tmid' in indep:
            print(f"\n--- T_mid test (FLAWED — T_mid is middle, not end of transition) ---")
            for role, key in [('Low', 'celeba_indep_lowblur_tmid'),
                              ('High', 'celeba_indep_highblur_tmid'),
                              ('All', 'celeba_indep_all_tmid')]:
                if key in indep:
                    print(f"  {role:>5s}: {indep[key]:>6.1f}  (Δ = {indep[key]-baseline:+.1f})")

        if 'celeba_indep_all_argmin' in indep:
            print(f"\n--- Argmin test ---")
            for role, key in [('All', 'celeba_indep_all_argmin'),
                              ('Low', 'celeba_indep_argmin_low'),
                              ('High', 'celeba_indep_argmin_high')]:
                if key in indep:
                    print(f"  {role:>5s}: {indep[key]:>6.1f}  (Δ = {indep[key]-baseline:+.1f})")

        if any(k in indep for k in ['celeba_indep_thresh_all','celeba_indep_thresh_low','celeba_indep_thresh_high']):
            print(f"\n--- Threshold test ---")
            for role, key in [('All', 'celeba_indep_thresh_all'),
                              ('Low', 'celeba_indep_thresh_low'),
                              ('High', 'celeba_indep_thresh_high')]:
                if key in indep:
                    print(f"  {role:>5s}: {indep[key]:>6.1f}  (Δ = {indep[key]-baseline:+.1f})")

    analysis = analyze_independence(indep)
    results['independence_analysis'] = analysis
    if 'tmid' in analysis:
        a = analysis['tmid']
        print(f"\n  T_mid additivity: predicted={a['predicted_additive']:.1f}, actual={a['all']:.1f}, "
              f"deviation={a['actual_minus_predicted']:+.1f} ({a['verdict']})")
    if 'argmin' in analysis and 'predicted_additive' in analysis['argmin']:
        a = analysis['argmin']
        print(f"  Argmin additivity: predicted={a['predicted_additive']:.1f}, actual={a['all']:.1f}, "
              f"deviation={a['actual_minus_predicted']:+.1f}")

    # --- Meta-curves ---
    print(f"\n{'='*60}\nMETA-CURVES\n{'='*60}")
    mc = {'sigma': [], 't_mid': [], 'argmin_T': [], 'argmin_FID': [],
          'threshold_T': [], 'T_95': [], 'bump_A': [], 'hump_peak_T': [],
          'fid_plat': [], 'fid_base': []}

    for bucket in sorted(data.keys()):
        fit = results['fits'][str(bucket)]
        if fit['params'] is None:
            continue
        p = fit['params']
        mc['sigma'].append(BLUR_SIGMAS[bucket])
        for key in ['t_mid', 'argmin_T', 'argmin_FID', 'threshold_T', 'T_95',
                     'fid_plat', 'fid_base', 'hump_peak_T']:
            mc[key].append(p.get(key, 0))
        mc['bump_A'].append(p.get('A', 0))

    results['meta_curves'] = mc

    print(f"{'σ':>4s} {'T_mid':>5s} {'argT':>5s} {'argFID':>6s} {'thrT':>5s} {'T_95':>5s} {'plat':>5s} {'base':>5s} {'bmpA':>5s} {'hmpT':>5s}")
    for i in range(len(mc['sigma'])):
        print(f"{mc['sigma'][i]:>4.1f} {mc['t_mid'][i]:>5.3f} {mc['argmin_T'][i]:>5.3f} "
              f"{mc['argmin_FID'][i]:>6.1f} {mc['threshold_T'][i]:>5.3f} {mc['T_95'][i]:>5.3f} "
              f"{mc['fid_plat'][i]:>5.0f} {mc['fid_base'][i]:>5.1f} {mc['bump_A'][i]:>5.1f} "
              f"{mc['hump_peak_T'][i]:>5.3f}")

    # Save
    outpath = f'/data/scratch/honjar/generated/celeba_curve_analysis_{kimg}kimg.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {outpath}")
