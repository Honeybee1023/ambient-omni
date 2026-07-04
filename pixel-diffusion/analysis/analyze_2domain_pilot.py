"""
Analyze 2-domain pilot results.
For each category: plot PickScore and Vendi vs T.
Check if log(metric - c) vs T is linear (would justify exponential form).
"""

import json, os
import numpy as np

GENERATED_DIR = '/data/scratch/honjar/generated'
SUPPLEMENTARY = ['dog', 'cat', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']
T_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

print('=' * 70)
print('2-Domain Pilot Analysis')
print('=' * 70)

all_results = {}

for cat in SUPPLEMENTARY:
    results = []
    for t_val in T_VALUES:
        name = 'pilot2d_%s_T%03d' % (cat, int(t_val * 100))
        p = os.path.join(GENERATED_DIR, 'metrics_%s_1000kimg.json' % name)
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            results.append({
                't': t_val,
                'pickscore': d['pickscore']['mean'],
                'vendi': d['vendi']['score'],
            })
        else:
            results.append({'t': t_val, 'pickscore': None, 'vendi': None})
    all_results[cat] = results

    print('\n=== %s ===' % cat.upper())
    print('  T      PickScore    Vendi')
    print('  ' + '-' * 30)
    for r in results:
        if r['pickscore'] is not None:
            print('  %.2f   %9.4f   %7.4f' % (r['t'], r['pickscore'], r['vendi']))
        else:
            print('  %.2f   MISSING     MISSING' % r['t'])

    # Check log-linearity for complete data
    complete = [r for r in results if r['pickscore'] is not None]
    if len(complete) >= 3:
        ts = np.array([r['t'] for r in complete])
        ps = np.array([r['pickscore'] for r in complete])
        vs = np.array([r['vendi'] for r in complete])

        print('\n  Trends:')
        print('    PickScore range: %.4f - %.4f (spread: %.4f)' % (ps.min(), ps.max(), ps.max()-ps.min()))
        print('    Vendi range:     %.4f - %.4f (spread: %.4f)' % (vs.min(), vs.max(), vs.max()-vs.min()))

        # Linear fit
        for metric_name, y in [('PickScore', ps), ('Vendi', vs)]:
            slope, intercept = np.polyfit(ts, y, 1)
            pred_lin = intercept + slope * ts
            ss_res = np.sum((y - pred_lin)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Log-linear check: fit log(|y - c|) vs T
            # Estimate c as the value at the asymptote
            # Try c = y at T=1 for metrics that decrease, c = y at T=0 for metrics that increase
            c_candidates = [y.min() - 0.01, y.max() + 0.01]
            best_r2_log = -np.inf
            best_c = None
            for c in c_candidates:
                shifted = np.abs(y - c)
                if np.all(shifted > 0):
                    log_y = np.log(shifted)
                    slope_log, intercept_log = np.polyfit(ts, log_y, 1)
                    pred_log = intercept_log + slope_log * ts
                    ss_res_log = np.sum((log_y - pred_log)**2)
                    ss_tot_log = np.sum((log_y - np.mean(log_y))**2)
                    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 0
                    if r2_log > best_r2_log:
                        best_r2_log = r2_log
                        best_c = c

            print('    %s: linear R2=%.4f, log-linear R2=%.4f' % (metric_name, r2_lin, best_r2_log))
            if best_r2_log > r2_lin + 0.05:
                print('      -> Log-linear fits BETTER: exponential form justified')
            elif r2_lin > best_r2_log + 0.05:
                print('      -> Linear fits BETTER: exponential may not be needed')
            else:
                print('      -> Similar fit: hard to distinguish with 5 points')

# Summary
print('\n' + '=' * 70)
print('SUMMARY: Does the exponential form hold?')
print('=' * 70)
print('Look for categories where log-linear R2 >> linear R2.')
print('If most categories show this, exponential form is justified for diffusion.')
print('If linear fits just as well, simpler model may suffice.')
print()
print('Key plots to make (in Jupyter):')
print('  1. For each category: T vs PickScore and T vs Vendi (raw curves)')
print('  2. For each category: T vs log(|metric - c|) (should be linear if exponential)')
print('  3. Compare dog curve to cat curve (should have opposite slopes)')

# Save for Jupyter
save_path = os.path.join(GENERATED_DIR, 'pilot2d_results.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nResults saved to %s' % save_path)
