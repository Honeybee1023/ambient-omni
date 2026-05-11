"""
7-category extra T values: {0.86, 0.87, 0.88, 0.89, 0.96, 0.98}
These were missing from the original create_7cat_datasets.py.
Same structure: all categories present, one varies, rest at T=1.
"""

import json, os, shutil
import numpy as np
from scipy.stats import norm

P_MEAN, P_STD = -1.2, 1.2
CLASSIFIED_DIR = '/data/scratch/honjar/afhq_classified'
DATA_ROOT = '/data/scratch/honjar/afhq/afhq'
ANNOTATED_DIR = '/data/scratch/honjar/annotated_datasets'
SHARED_DIR = os.path.join(ANNOTATED_DIR, 'shared_all_categories_64')

WILD_CATEGORIES = ['wolf', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']
SUPPLEMENTARY = ['dog', 'cat', 'tiger', 'lion', 'fox', 'leopard', 'cheetah']
EXTRA_T_VALUES = [0.86, 0.87, 0.88, 0.89, 0.96, 0.98]
SIGMA_MIN_T1 = float(np.exp(P_STD * norm.ppf(0.999) + P_MEAN))

def t_to_sigma_min(t_value):
    t_clipped = np.clip(t_value, 0.001, 0.999)
    return float(np.exp(P_STD * norm.ppf(t_clipped) + P_MEAN))

def t_to_suffix(t_val):
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

all_files = {}
for cat in SUPPLEMENTARY:
    all_files[cat] = load_file_list(cat)
wolf_files = load_file_list('wolf')
total = len(wolf_files) + sum(len(v) for v in all_files.values())
print('Total per dataset: %d images' % total)

created = []

for varying_cat in SUPPLEMENTARY:
    print('=== Varying: %s ===' % varying_cat)
    for t_val in EXTRA_T_VALUES:
        suffix = t_to_suffix(t_val)
        name = 'exp7d_%s_T%s' % (varying_cat, suffix)
        dataset_dir = os.path.join(ANNOTATED_DIR, name)
        if os.path.exists(dataset_dir):
            print('  SKIP (exists): %s' % name)
            continue
        os.makedirs(dataset_dir)
        annotations = []
        for fname in wolf_files:
            src = os.path.join(SHARED_DIR, fname)
            os.symlink(src, os.path.join(dataset_dir, fname))
            annotations.append({'filename': fname, 'sigma_min': 0.0, 'sigma_max': 0.0})
        for cat in SUPPLEMENTARY:
            if cat == varying_cat:
                sigma_min = 0.0 if t_val < 0.001 else t_to_sigma_min(t_val)
            else:
                sigma_min = SIGMA_MIN_T1
            for fname in all_files[cat]:
                src = os.path.join(SHARED_DIR, fname)
                dst = os.path.join(dataset_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                annotations.append({'filename': fname, 'sigma_min': sigma_min, 'sigma_max': 0.0})
        with open(os.path.join(dataset_dir, 'annotations.jsonl'), 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')
        print('  %s: %d images, %s T=%.3f' % (name, len(annotations), varying_cat, t_val))
        created.append(name)

print('\nCreated %d new datasets' % len(created))
