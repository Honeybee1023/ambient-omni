"""
Per-category regression analysis for Phase 3.

Fits linear regressions:
  1. T -> PickScore (sanity check: should want to exclude non-wolves)
  2. T -> Vendi (what categories help/hurt diversity)
  3. Optimal T to maximize PickScore (Phase 1 sanity check)
  4. Optimal T to maximize Vendi with PickScore constraint (Phase 2 goal)

Reads: percat_r1_assignments.npz + 20 metric JSONs
"""

import json
import os
import numpy as np
from itertools import product as iterproduct

GENERATED_DIR = '/data/scratch/honjar/generated'
ANNOTATED_DIR = '/data/scratch/honjar/annotated_datasets'

# --- Load data ---
assignments = np.load(os.path.join(ANNOTATED_DIR, 'percat_r1_assignments.npz'), allow_pickle=True)
t_matrix = assignments['t_matrix']  # (20, 7)
categories = list(assignments['categories'])

pickscore = []
vendi = []
aesthetic = []
for i in range(20):
    p = os.path.join(GENERATED_DIR, 'metrics_percat_r1_model_0%02d_1000kimg.json' % i)
    with open(p) as f:
        d = json.load(f)
    pickscore.append(d['pickscore']['mean'])
    vendi.append(d['vendi']['score'])
    aesthetic.append(d['aesthetic']['mean'])

pickscore = np.array(pickscore)
vendi = np.array(vendi)
aesthetic = np.array(aesthetic)

n, p = t_matrix.shape
print('=' * 60)
print('Per-Category Regression Analysis')
print('=' * 60)
print('Models: %d, Categories: %d' % (n, p))
print('Categories: %s' % categories)
print()

# --- Helper: OLS regression ---
def fit_ols(X, y):
    """Fit OLS with intercept. Returns intercept, coefficients, R2, residuals."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    residuals = y - y_pred
    return beta[0], beta[1:], r2, adj_r2, residuals, y_pred

def predict_ols(intercept, coefs, X):
    return intercept + X @ coefs

# --- 1. PickScore regression ---
print('=== Regression 1: T -> PickScore ===')
intercept_ps, coefs_ps, r2_ps, adj_r2_ps, resid_ps, pred_ps = fit_ols(t_matrix, pickscore)
print('R2 = %.4f, Adjusted R2 = %.4f' % (r2_ps, adj_r2_ps))
print('Intercept: %.4f' % intercept_ps)
print()
print('Coefficients (effect of increasing T, i.e., EXCLUDING more):')
sorted_idx = np.argsort(coefs_ps)
for idx in sorted_idx:
    direction = 'excluding HURTS' if coefs_ps[idx] < 0 else 'excluding HELPS'
    print('  %-10s: %+.4f  (%s quality)' % (categories[idx], coefs_ps[idx], direction))
print()
print('Residual std: %.4f' % np.std(resid_ps))

# --- 2. Vendi regression ---
print()
print('=== Regression 2: T -> Vendi ===')
intercept_v, coefs_v, r2_v, adj_r2_v, resid_v, pred_v = fit_ols(t_matrix, vendi)
print('R2 = %.4f, Adjusted R2 = %.4f' % (r2_v, adj_r2_v))
print('Intercept: %.4f' % intercept_v)
print()
print('Coefficients (effect of increasing T, i.e., EXCLUDING more):')
sorted_idx = np.argsort(coefs_v)
for idx in sorted_idx:
    direction = 'excluding HURTS' if coefs_v[idx] < 0 else 'excluding HELPS'
    print('  %-10s: %+.4f  (%s diversity)' % (categories[idx], coefs_v[idx], direction))
print()
print('Residual std: %.4f' % np.std(resid_v))

# --- 3. Optimal T to maximize PickScore ---
print()
print('=== Optimal T: Maximize PickScore (sanity check) ===')
# For linear model, optimal is at boundary: T=0 if coef<0, T=1 if coef>0
opt_t_ps = np.array([0.0 if c < 0 else 1.0 for c in coefs_ps])
pred_opt_ps = predict_ols(intercept_ps, coefs_ps, opt_t_ps.reshape(1, -1))[0]
print('Optimal T per category:')
for cat, t_val in zip(categories, opt_t_ps):
    label = 'INCLUDE' if t_val == 0 else 'EXCLUDE'
    print('  %-10s: T=%.1f (%s)' % (cat, t_val, label))
print('Predicted PickScore at optimum: %.4f' % pred_opt_ps)
print()
print('Sanity check: does the model want to exclude everything non-wolf?')
all_exclude = np.all(opt_t_ps == 1.0)
print('  All supplementary excluded: %s' % all_exclude)
if not all_exclude:
    included = [categories[i] for i in range(len(categories)) if opt_t_ps[i] == 0]
    print('  Kept categories: %s' % included)

# --- 4. Optimal T: Maximize Vendi with PickScore constraint ---
print()
print('=== Optimal T: Maximize Vendi with PickScore constraint ===')
print()

# Grid search over T values for each category
# With 7 categories and 5 values each = 5^7 = 78125 combos — fast for regression
t_grid_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# Test several PickScore thresholds
thresholds = {
    'aggressive (18.50)': 18.50,  # above median
    'moderate (18.35)': 18.35,    # near median
    'relaxed (18.20)': 18.20,     # near minimum
    'very relaxed (18.00)': 18.00,
}

# Build grid
grid = np.array(list(iterproduct(t_grid_values, repeat=p)))
pred_ps_grid = predict_ols(intercept_ps, coefs_ps, grid)
pred_v_grid = predict_ols(intercept_v, coefs_v, grid)

# Also find unconstrained Vendi optimum
best_v_idx = np.argmax(pred_v_grid)
print('Unconstrained Vendi optimum:')
print('  Predicted Vendi: %.4f' % pred_v_grid[best_v_idx])
print('  Predicted PickScore: %.4f' % pred_ps_grid[best_v_idx])
for cat, t_val in zip(categories, grid[best_v_idx]):
    label = 'INCLUDE' if t_val == 0 else ('EXCLUDE' if t_val == 1.0 else 'T=%.2f' % t_val)
    print('  %-10s: T=%.2f (%s)' % (cat, t_val, label))
print()

for name, threshold in thresholds.items():
    mask = pred_ps_grid >= threshold
    if not np.any(mask):
        print('Threshold %s: NO feasible solution' % name)
        print()
        continue
    
    feasible_v = pred_v_grid[mask]
    feasible_idx = np.where(mask)[0]
    best_idx = feasible_idx[np.argmax(feasible_v)]
    
    print('Threshold %s:' % name)
    print('  Predicted Vendi: %.4f' % pred_v_grid[best_idx])
    print('  Predicted PickScore: %.4f' % pred_ps_grid[best_idx])
    for cat, t_val in zip(categories, grid[best_idx]):
        label = 'INCLUDE' if t_val == 0 else ('EXCLUDE' if t_val == 1.0 else 'T=%.2f' % t_val)
        print('  %-10s: T=%.2f (%s)' % (cat, t_val, label))
    print()

# --- 5. Penalty-based combined metric ---
print('=== Combined metric: Vendi - lambda * max(0, threshold - PickScore)^2 ===')
print()
threshold_penalty = np.median(pickscore)
print('Using threshold = median PickScore = %.4f' % threshold_penalty)

for lam in [1.0, 5.0, 10.0, 50.0]:
    penalty = lam * np.maximum(0, threshold_penalty - pred_ps_grid) ** 2
    combined = pred_v_grid - penalty
    best_idx = np.argmax(combined)
    
    print('  lambda=%.1f -> Vendi=%.4f, PickScore=%.4f' % (lam, pred_v_grid[best_idx], pred_ps_grid[best_idx]))
    t_str = ', '.join(['%s=%.2f' % (cat, t) for cat, t in zip(categories, grid[best_idx])])
    print('    T: %s' % t_str)

# --- 6. Summary ---
print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print()
print('PickScore regression R2: %.4f (adj: %.4f)' % (r2_ps, adj_r2_ps))
print('Vendi regression R2:     %.4f (adj: %.4f)' % (r2_v, adj_r2_v))
print()
if adj_r2_ps > 0.7:
    print('PickScore R2 is HIGH — linear model captures most variance.')
elif adj_r2_ps > 0.4:
    print('PickScore R2 is MODERATE — linear model captures some signal, may benefit from more data.')
else:
    print('PickScore R2 is LOW — linear model may not be appropriate, or need more models.')
    
if adj_r2_v > 0.7:
    print('Vendi R2 is HIGH — linear model captures most variance.')
elif adj_r2_v > 0.4:
    print('Vendi R2 is MODERATE — linear model captures some signal, may benefit from more data.')
else:
    print('Vendi R2 is LOW — linear model may not be appropriate, or need more models.')

print()
print('Observed data ranges:')
print('  PickScore: %.4f - %.4f' % (pickscore.min(), pickscore.max()))
print('  Vendi:     %.4f - %.4f' % (vendi.min(), vendi.max()))
print('  Wolves-only baseline: PickScore=21.60, Vendi=3.35')
