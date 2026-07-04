"""
Find optimal T at multiple PickScore thresholds using fitted mixing law,
then create validation datasets for each.
"""

import json
import os
import numpy as np
from scipy.optimize import differential_evolution

GENERATED_DIR = '/data/scratch/honjar/generated'
ANNOTATED_DIR = '/data/scratch/honjar/annotated_datasets'
CLASSIFIED_DIR = '/data/scratch/honjar/afhq_classified'
DATA_ROOT = '/data/scratch/honjar/afhq/afhq'
SHARED_DIR = os.path.join(ANNOTATED_DIR, 'shared_all_categories_64')

from scipy.stats import norm
P_MEAN, P_STD = -1.2, 1.2

def t_to_sigma_min(t_value):
    t_clipped = np.clip(t_value, 0.001, 0.999)
    return float(np.exp(P_STD * norm.ppf(t_clipped) + P_MEAN))

# --- Load data and refit (fast) ---
assignments = np.load(os.path.join(ANNOTATED_DIR, 'percat_r1_assignments.npz'), allow_pickle=True)
t_matrix = assignments['t_matrix']
categories = list(assignments['categories'])
M = len(categories)

pickscore, vendi = [], []
for i in range(20):
    p = os.path.join(GENERATED_DIR, 'metrics_percat_r1_model_0%02d_1000kimg.json' % i)
    with open(p) as f:
        d = json.load(f)
    pickscore.append(d['pickscore']['mean'])
    vendi.append(d['vendi']['score'])
pickscore, vendi = np.array(pickscore), np.array(vendi)

def exp_model(params, T):
    return params[0] + params[1] * np.exp(T @ params[2:])

def huber_loss(params, T, y, delta=0.1):
    pred = exp_model(params, T)
    resid = y - pred
    abs_r = np.abs(resid)
    return np.sum(np.where(abs_r <= delta, 0.5 * resid**2, delta * (abs_r - 0.5 * delta)))

def fit_exp(T, y, n_restarts=50):
    best_loss, best_params = np.inf, None
    for trial in range(n_restarts):
        rng = np.random.RandomState(trial)
        params0 = np.concatenate([[np.mean(y) + rng.randn()*np.std(y)*0.5,
                                    rng.randn()*np.std(y)*0.5], rng.randn(M)*0.5])
        try:
            from scipy.optimize import minimize
            result = minimize(huber_loss, params0, args=(T, y), method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12})
            if result.fun < best_loss:
                best_loss, best_params = result.fun, result.x
        except:
            continue
    return best_params

print('Fitting exponential laws...')
params_ps = fit_exp(t_matrix, pickscore)
params_v = fit_exp(t_matrix, vendi)
print('Done.')
print()

# --- Sweep thresholds ---
WILD_CATEGORIES = ['wolf', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']
DOMESTIC_CATEGORIES = ['dog', 'cat']
ALL_CATEGORIES = WILD_CATEGORIES + DOMESTIC_CATEGORIES

thresholds = [18.50, 18.60, 18.70, 18.80]
results = {}

print('=' * 70)
print('Threshold sweep: maximize Vendi subject to PickScore >= threshold')
print('=' * 70)

for thresh in thresholds:
    def objective(T_flat):
        ps = exp_model(params_ps, T_flat)
        v = exp_model(params_v, T_flat)
        penalty = 1000.0 * max(0, thresh - ps) ** 2
        return -v + penalty

    result = differential_evolution(objective, [(0,1)]*M, seed=42, maxiter=1000)
    opt_T = result.x
    pred_v = exp_model(params_v, opt_T)
    pred_ps = exp_model(params_ps, opt_T)

    name = 'validate_thresh_%s' % str(thresh).replace('.', '')
    results[name] = {'threshold': thresh, 'opt_T': opt_T, 'pred_vendi': pred_v, 'pred_pickscore': pred_ps}

    print('\nThreshold >= %.2f:' % thresh)
    print('  Predicted Vendi=%.4f, PickScore=%.4f' % (pred_v, pred_ps))
    for j, cat in enumerate(categories):
        print('  %-10s: T=%.3f' % (cat, opt_T[j]))

# Also add unconstrained Vendi optimum
def neg_v(T_flat):
    return -exp_model(params_v, T_flat)
result = differential_evolution(neg_v, [(0,1)]*M, seed=42, maxiter=1000)
opt_T = result.x
results['validate_unconstrained'] = {
    'threshold': 0, 'opt_T': opt_T,
    'pred_vendi': exp_model(params_v, opt_T),
    'pred_pickscore': exp_model(params_ps, opt_T)
}
print('\nUnconstrained Vendi optimum:')
print('  Predicted Vendi=%.4f, PickScore=%.4f' % (results['validate_unconstrained']['pred_vendi'],
                                                    results['validate_unconstrained']['pred_pickscore']))
for j, cat in enumerate(categories):
    print('  %-10s: T=%.3f' % (cat, opt_T[j]))

# --- Create validation datasets ---
print('\n' + '=' * 70)
print('Creating validation datasets')
print('=' * 70)

# Load file lists
def load_file_lists():
    file_lists = {}
    for cat in WILD_CATEGORIES:
        list_path = os.path.join(CLASSIFIED_DIR, '%s_files.json' % cat)
        with open(list_path) as f:
            file_lists[cat] = ['%s_%s' % (cat, fn) for fn in json.load(f)]
    for cat in DOMESTIC_CATEGORIES:
        cat_dir = os.path.join(DATA_ROOT, 'train/%s' % cat)
        original = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        file_lists[cat] = ['%s_%s' % (cat, fn) for fn in original]
    return file_lists

file_lists = load_file_lists()

import shutil

for name, info in results.items():
    opt_T = info['opt_T']
    dataset_dir = os.path.join(ANNOTATED_DIR, name)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    # Build T dict: wolves=0, supplementary categories from optimization
    t_values = {'wolf': 0.0}
    for j, cat in enumerate(categories):
        t_values[cat] = float(opt_T[j])

    annotations = []
    for cat in ALL_CATEGORIES:
        t_val = t_values[cat]
        sigma_min = 0.0 if t_val < 0.001 else t_to_sigma_min(t_val)
        for fname in file_lists[cat]:
            src = os.path.join(SHARED_DIR, fname)
            dst = os.path.join(dataset_dir, fname)
            os.symlink(src, dst)
            annotations.append({'filename': fname, 'sigma_min': sigma_min, 'sigma_max': 0.0})

    with open(os.path.join(dataset_dir, 'annotations.jsonl'), 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print('Created %s: %d images, pred_V=%.3f, pred_PS=%.3f' %
          (name, len(annotations), info['pred_vendi'], info['pred_pickscore']))

# Save results for later analysis
with open(os.path.join(ANNOTATED_DIR, 'validation_sweep_results.json'), 'w') as f:
    save_results = {}
    for name, info in results.items():
        save_results[name] = {
            'threshold': info['threshold'],
            'opt_T': {cat: float(info['opt_T'][j]) for j, cat in enumerate(categories)},
            'pred_vendi': float(info['pred_vendi']),
            'pred_pickscore': float(info['pred_pickscore']),
        }
    json.dump(save_results, f, indent=2)

print('\nAll datasets created. Submit training with:')
for name in results:
    print('  sbatch /data/scratch/honjar/ambient-omni/pixel-diffusion/run_train_percat.sh %s' % name)
