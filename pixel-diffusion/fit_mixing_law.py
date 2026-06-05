"""
Fit data mixing laws (Ye et al., 2024) to per-category threshold experiment.

Equation: metric(T) = c + k * exp(sum_j t_j * T_j)

Where T_j is the noise threshold for category j (0=included, 1=excluded).
Fits separate laws for PickScore and Vendi, then optimizes constrained objective.

9 parameters per metric (c, k, t_1...t_7) from 20 data points = 11 DOF.
"""

import json
import os
import numpy as np
from scipy.optimize import minimize, differential_evolution

GENERATED_DIR = '/data/scratch/honjar/generated'
ANNOTATED_DIR = '/data/scratch/honjar/annotated_datasets'

# --- Load data ---
assignments = np.load(os.path.join(ANNOTATED_DIR, 'percat_r1_assignments.npz'), allow_pickle=True)
t_matrix = assignments['t_matrix']  # (20, 7)
categories = list(assignments['categories'])

pickscore = []
vendi = []
for i in range(20):
    p = os.path.join(GENERATED_DIR, 'metrics_percat_r1_model_0%02d_1000kimg.json' % i)
    with open(p) as f:
        d = json.load(f)
    pickscore.append(d['pickscore']['mean'])
    vendi.append(d['vendi']['score'])

pickscore = np.array(pickscore)
vendi = np.array(vendi)
n, M = t_matrix.shape

print('=' * 60)
print('Data Mixing Law Fitting (Ye et al. 2024, Eq. 7)')
print('=' * 60)
print('Models: %d, Categories: %d' % (n, M))
print('Parameters per metric: %d (c, k, t_1...t_%d)' % (M + 2, M))
print('Degrees of freedom: %d' % (n - M - 2))
print()

# --- Exponential model: y = c + k * exp(sum_j t_j * T_j) ---

def exp_model(params, T):
    c = params[0]
    k = params[1]
    t = params[2:]
    return c + k * np.exp(T @ t)

def huber_loss(params, T, y, delta=0.1):
    pred = exp_model(params, T)
    resid = y - pred
    abs_r = np.abs(resid)
    loss = np.where(abs_r <= delta, 0.5 * resid**2, delta * (abs_r - 0.5 * delta))
    return np.sum(loss)

def fit_exp_law(T, y, name, n_restarts=50):
    """Fit exponential mixing law with multiple random restarts (following DML paper)."""
    best_loss = np.inf
    best_params = None
    
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        # Initialize: c near mean(y), k small, t near 0
        c0 = np.mean(y) + rng.randn() * np.std(y) * 0.5
        k0 = rng.randn() * np.std(y) * 0.5
        t0 = rng.randn(M) * 0.5
        params0 = np.concatenate([[c0, k0], t0])
        
        try:
            result = minimize(huber_loss, params0, args=(T, y), method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-12})
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Compute fit quality
    pred = exp_model(best_params, T)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(y - pred))
    
    c, k, t = best_params[0], best_params[1], best_params[2:]
    
    print('=== Exponential Law: %s ===' % name)
    print('  c = %.6f (irreducible component)' % c)
    print('  k = %.6f (scale)' % k)
    print('  R2 = %.4f, MAE = %.4f' % (r2, mae))
    print('  Coefficients t_j (effect of T on metric via exp):')
    for j, cat in enumerate(categories):
        print('    %-10s: t = %+.4f' % (cat, t[j]))
    print()
    
    return best_params, r2, mae

# --- Fit both metrics ---
params_ps, r2_ps, mae_ps = fit_exp_law(t_matrix, pickscore, 'PickScore')
params_v, r2_v, mae_v = fit_exp_law(t_matrix, vendi, 'Vendi')

# --- Compare to linear regression R2 ---
print('=== Model Comparison ===')
print('  PickScore: Linear R2=0.8594, Exponential R2=%.4f' % r2_ps)
print('  Vendi:     Linear R2=0.7016, Exponential R2=%.4f' % r2_v)
print()

# --- Optimize: maximize PickScore (sanity check) ---
print('=== Optimal T: Maximize PickScore (sanity check) ===')

def neg_pickscore(T_flat):
    return -exp_model(params_ps, T_flat)

bounds = [(0, 1)] * M
result_ps = differential_evolution(neg_pickscore, bounds, seed=42, maxiter=1000)
opt_T_ps = result_ps.x
pred_ps_opt = exp_model(params_ps, opt_T_ps)

print('Optimal T per category:')
for j, cat in enumerate(categories):
    print('  %-10s: T=%.3f' % (cat, opt_T_ps[j]))
print('Predicted PickScore: %.4f' % pred_ps_opt)
print()

# --- Optimize: maximize Vendi (unconstrained) ---
print('=== Optimal T: Maximize Vendi (unconstrained) ===')

def neg_vendi(T_flat):
    return -exp_model(params_v, T_flat)

result_v = differential_evolution(neg_vendi, bounds, seed=42, maxiter=1000)
opt_T_v = result_v.x
pred_v_opt = exp_model(params_v, opt_T_v)
pred_ps_at_v_opt = exp_model(params_ps, opt_T_v)

print('Optimal T per category:')
for j, cat in enumerate(categories):
    print('  %-10s: T=%.3f' % (cat, opt_T_v[j]))
print('Predicted Vendi: %.4f' % pred_v_opt)
print('Predicted PickScore at this T: %.4f' % pred_ps_at_v_opt)
print()

# --- Optimize: maximize Vendi subject to PickScore constraint ---
print('=== Optimal T: Maximize Vendi with PickScore constraint ===')
print()

for threshold in [18.50, 18.35, 18.20]:
    def neg_vendi_constrained(T_flat):
        ps = exp_model(params_ps, T_flat)
        v = exp_model(params_v, T_flat)
        # Heavy penalty for PickScore below threshold
        penalty = 1000.0 * max(0, threshold - ps) ** 2
        return -v + penalty
    
    result = differential_evolution(neg_vendi_constrained, bounds, seed=42, maxiter=1000)
    opt_T = result.x
    pred_v = exp_model(params_v, opt_T)
    pred_ps = exp_model(params_ps, opt_T)
    
    print('Threshold PickScore >= %.2f:' % threshold)
    print('  Predicted Vendi: %.4f, PickScore: %.4f' % (pred_v, pred_ps))
    for j, cat in enumerate(categories):
        print('  %-10s: T=%.3f' % (cat, opt_T[j]))
    print()

# --- Summary ---
print('=' * 60)
print('KEY RESULT: Do we get interior (non-binary) T values?')
print('=' * 60)
all_binary = all(t < 0.05 or t > 0.95 for t in opt_T)
if all_binary:
    print('All optimal T values are near 0 or 1 — exponential law')
    print('also gives corner solutions. May need more data or different form.')
else:
    interior = [(cat, t) for cat, t in zip(categories, opt_T) if 0.05 < t < 0.95]
    print('Found interior T values (not 0 or 1):')
    for cat, t in interior:
        print('  %-10s: T=%.3f' % (cat, t))
    print('The exponential form captures nonlinearity that linear missed!')
