"""
2-domain pilot v2: Additional T values for finer curve resolution.
Midpoints (0.125, 0.375, 0.625) + fine-grained near T=1 (0.80-0.99).
Same structure as v1: wolves + one supplementary category.
Total: 7 categories x 9 T values = 63 new datasets.
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)


import json, os, shutil
import numpy as np
from scipy.stats import norm

P_MEAN, P_STD = -1.2, 1.2
CLASSIFIED_DIR = f'{AMBIENT_BASE}/afhq_classified'
DATA_ROOT = f'{AMBIENT_BASE}/afhq/afhq'
ANNOTATED_DIR = f'{AMBIENT_BASE}/annotated_datasets'
SHARED_DIR = os.path.join(ANNOTATED_DIR, 'shared_all_categories_64')

WILD_CATEGORIES = ['wolf', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']
SUPPLEMENTARY = ['dog', 'cat', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']

NEW_T_VALUES = [0.125, 0.375, 0.625, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

def t_to_sigma_min(t_value):
    t_clipped = np.clip(t_value, 0.001, 0.999)
    return float(np.exp(P_STD * norm.ppf(t_clipped) + P_MEAN))

def t_to_suffix(t_val):
    """T value -> name suffix. Matches original 3-digit format where possible."""
    t_milli = round(t_val * 1000)
    if t_milli % 10 == 0:
        return '%03d' % (t_milli // 10)
    else:
        return '%04d' % t_milli

def load_file_list(cat):
    if cat in WILD_CATEGORIES:
        with open(os.path.join(CLASSIFIED_DIR, '%s_files.json' % cat)) as f:
            return ['%s_%s' % (cat, fn) for fn in json.load(f)]
    else:
        cat_dir = os.path.join(DATA_ROOT, 'train/%s' % cat)
        original = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        return ['%s_%s' % (cat, fn) for fn in original]

wolf_files = load_file_list('wolf')
print('Wolves: %d images (always T=0)' % len(wolf_files))
print()

created = []

for cat in SUPPLEMENTARY:
    cat_files = load_file_list(cat)
    print('=== %s: %d images ===' % (cat, len(cat_files)))

    for t_val in NEW_T_VALUES:
        suffix = t_to_suffix(t_val)
        name = 'pilot2d_%s_T%s' % (cat, suffix)
        dataset_dir = os.path.join(ANNOTATED_DIR, name)

        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)

        sigma_min = 0.0 if t_val < 0.001 else t_to_sigma_min(t_val)
        annotations = []

        for fname in wolf_files:
            src = os.path.join(SHARED_DIR, fname)
            dst = os.path.join(dataset_dir, fname)
            os.symlink(src, dst)
            annotations.append({'filename': fname, 'sigma_min': 0.0, 'sigma_max': 0.0})

        for fname in cat_files:
            src = os.path.join(SHARED_DIR, fname)
            dst = os.path.join(dataset_dir, fname)
            os.symlink(src, dst)
            annotations.append({'filename': fname, 'sigma_min': sigma_min, 'sigma_max': 0.0})

        with open(os.path.join(dataset_dir, 'annotations.jsonl'), 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')

        print('  %s: %d images, T=%.3f, sigma_min=%.6f' % (name, len(annotations), t_val, sigma_min))
        created.append(name)

print('\n=== Summary ===')
print('Created %d datasets' % len(created))
