"""
Create additional CelebA datasets:
- 64 new 2-domain datasets (finer T resolution at peaks/valleys)
- 3 independence test datasets (2k kimg argmin T values)
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
from scipy.stats import norm

BASE = f'{AMBIENT_BASE}'
ANNO_DIR = f'{BASE}/annotated_datasets'
SHARED_DIR = f'{BASE}/celeba_processed/shared_buckets_64'

NEW_2D = {
    1: [0.06, 0.19, 0.31, 0.44, 0.56, 0.69, 0.825, 0.875, 0.925, 0.96, 0.98, 0.995],
    2: [0.19, 0.31, 0.44, 0.56, 0.775, 0.825, 0.875, 0.925, 0.96, 0.98, 0.995],
    3: [0.31, 0.44, 0.56, 0.69, 0.775, 0.96, 0.98, 0.995],
    4: [0.44, 0.56, 0.69, 0.96, 0.98],
    5: [0.06, 0.19, 0.31, 0.56, 0.69, 0.775, 0.825, 0.98, 0.995],
    6: [0.06, 0.19, 0.44, 0.56, 0.69, 0.775, 0.825, 0.98, 0.995],
    7: [0.06, 0.19, 0.31, 0.44, 0.56, 0.69, 0.775, 0.825, 0.875, 0.925],
}

INDEP_CONFIGS = {
    'celeba_indep2k_argmin_low':  {1:0.95, 2:0.95, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0},
    'celeba_indep2k_argmin_high': {1:1.0, 2:1.0, 3:1.0, 4:0.97, 5:0.99, 6:0.99, 7:1.0},
    'celeba_indep2k_all_argmin':  {1:0.95, 2:0.95, 3:1.0, 4:0.97, 5:0.99, 6:0.99, 7:1.0},
}

def T_to_suffix(T):
    if abs(T * 100 - round(T * 100)) < 0.001:
        return f'{int(round(T * 100)):03d}'
    else:
        return f'{int(round(T * 1000)):04d}'

# ============================================================
# STEP 1: Learn and VERIFY T -> sigma_min mapping
# ============================================================
print("=== LEARNING T -> sigma_min MAPPING ===")

# Inspect format
sample_path = f'{ANNO_DIR}/celeba_2d_b4_T050/annotations.jsonl'
with open(sample_path) as f:
    for line in f:
        e = json.loads(line)
        if e['filename'].startswith('b0_'):
            print(f"Clean sample:  {e}")
            break
with open(sample_path) as f:
    for line in f:
        e = json.loads(line)
        if e['filename'].startswith('b4_'):
            print(f"Supp sample:   {e}")
            break

# Read all existing (T, sigma_min, sigma_max) from bucket 4
known = {}
for suffix in ['000','0125','025','0375','050','0625','075','080','085','090','095','097','099','100']:
    T = int(suffix) / (100.0 if len(suffix) == 3 else 1000.0)
    anno = f'{ANNO_DIR}/celeba_2d_b4_T{suffix}/annotations.jsonl'
    if not os.path.exists(anno):
        print(f'  WARNING: missing {anno}')
        continue
    with open(anno) as f:
        for line in f:
            e = json.loads(line)
            if e['filename'].startswith('b4_'):
                known[T] = (e['sigma_min'], e.get('sigma_max', 0.0))
                break

print(f"\nExisting mappings ({len(known)} points):")
for T in sorted(known.keys()):
    sm, sx = known[T]
    print(f"  T={T:.4f} -> sigma_min={sm:.6f}, sigma_max={sx:.6f}")

# Verify formula: sigma_min = exp(-1.2 + 1.2 * norm.ppf(T))
P_MEAN, P_STD = -1.2, 1.2
print("\n=== VERIFYING FORMULA: sigma_min = exp(-1.2 + 1.2 * norm.ppf(T)) ===")
formula_ok = True
for T in sorted(known.keys()):
    actual = known[T][0]
    if T <= 0.001 or T >= 0.999:
        print(f"  T={T:.4f}: actual={actual:.6f} (edge case, skipping verification)")
        continue
    predicted = np.exp(P_MEAN + P_STD * norm.ppf(T))
    diff = abs(actual - predicted)
    status = "OK" if diff < 0.01 else "MISMATCH"
    if status == "MISMATCH":
        formula_ok = False
    print(f"  T={T:.4f}: actual={actual:.6f}, formula={predicted:.6f}, diff={diff:.6f} [{status}]")

if formula_ok:
    print("\n*** Formula VERIFIED. Using exact formula for new T values. ***")
else:
    print("\n*** WARNING: Formula MISMATCH. Using interpolation instead. ***")

known_Ts = np.array(sorted(known.keys()))
known_sms = np.array([known[t][0] for t in known_Ts])

def get_sigma_min(T):
    if T <= 0.001:
        return 0.0
    if T >= 0.999:
        return float(np.exp(P_MEAN + P_STD * norm.ppf(0.999)))
    if formula_ok:
        return float(np.exp(P_MEAN + P_STD * norm.ppf(T)))
    else:
        return float(np.interp(T, known_Ts, known_sms))

# Check sigma_max pattern
sigma_maxes = set(known[T][1] for T in known)
SUPP_SIGMA_MAX = sigma_maxes.pop() if len(sigma_maxes) == 1 else 0.0
print(f"sigma_max for supp images: {SUPP_SIGMA_MAX}" +
      ("" if len(sigma_maxes) <= 1 else f" (WARNING: varies: {sigma_maxes})"))

# Print all new T -> sigma_min values
print("\n=== NEW T VALUES ===")
all_new = sorted(set(T for ts in NEW_2D.values() for T in ts))
for T in all_new:
    print(f"  T={T:.4f} (T{T_to_suffix(T)}) -> sigma_min={get_sigma_min(T):.6f}")

# ============================================================
# STEP 2: Create 2-domain datasets
# ============================================================
print("\n=== CREATING 2-DOMAIN DATASETS ===")
created_2d = []
for b, t_list in sorted(NEW_2D.items()):
    ref = f'{ANNO_DIR}/celeba_2d_b{b}_T050'
    with open(f'{ref}/annotations.jsonl') as f:
        ref_entries = [json.loads(line) for line in f]
    # Get sigma_max for this bucket's supp images
    bsm = SUPP_SIGMA_MAX
    for entry in ref_entries:
        if entry['filename'].startswith(f'b{b}_'):
            bsm = entry.get('sigma_max', 0.0)
            break

    for T in t_list:
        suffix = T_to_suffix(T)
        ds_name = f'celeba_2d_b{b}_T{suffix}'
        ds_dir = f'{ANNO_DIR}/{ds_name}'
        if os.path.exists(ds_dir):
            print(f'  EXISTS: {ds_name}')
            continue
        os.makedirs(ds_dir)
        sm = get_sigma_min(T)
        with open(f'{ds_dir}/annotations.jsonl', 'w') as fout:
            for entry in ref_entries:
                fn = entry['filename']
                os.symlink(f'{SHARED_DIR}/{fn}', f'{ds_dir}/{fn}')
                if fn.startswith(f'b{b}_'):
                    fout.write(json.dumps({'filename': fn, 'sigma_min': sm, 'sigma_max': bsm}) + '\n')
                else:
                    fout.write(json.dumps({'filename': fn, 'sigma_min': 0.0, 'sigma_max': 0.0}) + '\n')
        print(f'  CREATED: {ds_name} (T={T}, sigma_min={sm:.6f})')
        created_2d.append(ds_name)

# ============================================================
# STEP 3: Create independence test datasets
# ============================================================
print("\n=== CREATING INDEPENDENCE TEST DATASETS ===")
# Get sigma_max per bucket from reference datasets
bucket_smax = {}
for b in range(1, 8):
    ref = f'{ANNO_DIR}/celeba_2d_b{b}_T050/annotations.jsonl'
    with open(ref) as f:
        for line in f:
            e = json.loads(line)
            if e['filename'].startswith(f'b{b}_'):
                bucket_smax[b] = e.get('sigma_max', 0.0)
                break

all_files = sorted([f for f in os.listdir(SHARED_DIR)
                     if f.endswith('.jpg') and not f.startswith('._')])
print(f"Total shared images: {len(all_files)}")

created_indep = []
for ds_name, bucket_T_map in INDEP_CONFIGS.items():
    ds_dir = f'{ANNO_DIR}/{ds_name}'
    if os.path.exists(ds_dir):
        print(f'  EXISTS: {ds_name}')
        continue
    os.makedirs(ds_dir)
    with open(f'{ds_dir}/annotations.jsonl', 'w') as fout:
        for fname in all_files:
            bnum = int(fname[1])
            if bnum == 0:
                sm, sx = 0.0, 0.0
            else:
                T = bucket_T_map.get(bnum, 1.0)
                sm = get_sigma_min(T)
                sx = bucket_smax.get(bnum, 0.0)
            os.symlink(f'{SHARED_DIR}/{fname}', f'{ds_dir}/{fname}')
            fout.write(json.dumps({'filename': fname, 'sigma_min': sm, 'sigma_max': sx}) + '\n')
    print(f'  CREATED: {ds_name}')
    for bb in range(1, 8):
        print(f'    B{bb}: T={bucket_T_map[bb]} -> sigma_min={get_sigma_min(bucket_T_map[bb]):.6f}')
    created_indep.append(ds_name)

print(f"\n=== SUMMARY ===")
print(f"New 2-domain: {len(created_2d)}")
print(f"New independence: {len(created_indep)}")
print(f"Total: {len(created_2d) + len(created_indep)}")
print("\nDataset names:")
for n in created_2d + created_indep:
    print(f"  {n}")
