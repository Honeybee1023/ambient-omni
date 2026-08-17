"""
Prepare all data needed for 7D visualization Jupyter cells.
Reads: all_results_summary.json, celeba_2k_analysis.json, raw metrics files.
Saves: 7d_visualization_data.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, glob, os
import numpy as np

GEN_DIR = f'{AMBIENT_BASE}/generated'

# ============================================================
# 1. Load existing data
# ============================================================
with open(f'{GEN_DIR}/all_results_summary.json') as f:
    summary = json.load(f)
with open(f'{GEN_DIR}/celeba_2k_analysis.json') as f:
    analysis = json.load(f)

# ============================================================
# 2. Add 3 missing R4 points to summary
# ============================================================
for p in ['p05', 'p12', 'p14']:
    if p not in summary['bo_results']['R4']:
        mf = f'{GEN_DIR}/metrics_celeba_bo_r4_{p}_2000kimg.json'
        tf = f'{GEN_DIR}/tvec_celeba_bo_r4_{p}.json'
        if os.path.exists(mf) and os.path.exists(tf):
            fid = json.load(open(mf))['fid_score']
            tvec = json.load(open(tf))
            summary['bo_results']['R4'][p] = {
                'fid': round(fid, 2),
                'tvec': tvec
            }
            print(f"Added R4 {p}: FID={fid:.2f}")

# ============================================================
# 3. Add independence test data to summary
# ============================================================
indep_data = {
    '1k': {
        'baseline': {'name': 'celeba_indep_baseline', 'fid': None},
        'tmid_low': {'name': 'celeba_indep_lowblur_tmid', 'fid': None},
        'tmid_high': {'name': 'celeba_indep_highblur_tmid', 'fid': None},
        'tmid_all': {'name': 'celeba_indep_all_tmid', 'fid': None},
        'argmin_low': {'name': 'celeba_indep_argmin_low', 'fid': None},
        'argmin_high': {'name': 'celeba_indep_argmin_high', 'fid': None},
        'argmin_all': {'name': 'celeba_indep_all_argmin', 'fid': None},
        'thresh_low': {'name': 'celeba_indep_thresh_low', 'fid': None},
        'thresh_high': {'name': 'celeba_indep_thresh_high', 'fid': None},
        'thresh_all': {'name': 'celeba_indep_thresh_all', 'fid': None},
    },
    '2k': {
        'baseline': {'name': 'celeba_indep_baseline', 'fid': None},
        'argmin_low': {'name': 'celeba_indep2k_argmin_low', 'fid': None},
        'argmin_high': {'name': 'celeba_indep2k_argmin_high', 'fid': None},
        'argmin_all': {'name': 'celeba_indep2k_all_argmin', 'fid': None},
    }
}

# Read 1k kimg independence FIDs
for key, entry in indep_data['1k'].items():
    mf = f'{GEN_DIR}/metrics_{entry["name"]}_1000kimg.json'
    if os.path.exists(mf):
        entry['fid'] = json.load(open(mf))['fid_score']

# Read 2k kimg independence FIDs
for key, entry in indep_data['2k'].items():
    mf = f'{GEN_DIR}/metrics_{entry["name"]}_2000kimg.json'
    if os.path.exists(mf):
        entry['fid'] = json.load(open(mf))['fid_score']

# Compute independence predictions from sub-groups (method 1: predicted_all = baseline + delta_low + delta_high)
def compute_independence(data_dict, kimg_label):
    results = {}
    bl = data_dict['baseline']['fid']
    if bl is None:
        return results
    results['baseline'] = bl
    
    for regime in ['tmid', 'argmin', 'thresh']:
        low_key = f'{regime}_low'
        high_key = f'{regime}_high'
        all_key = f'{regime}_all'
        if low_key not in data_dict or data_dict[low_key]['fid'] is None:
            continue
        fid_low = data_dict[low_key]['fid']
        fid_high = data_dict[high_key]['fid']
        fid_all = data_dict[all_key]['fid']
        delta_low = fid_low - bl
        delta_high = fid_high - bl
        predicted = bl + delta_low + delta_high
        deviation = fid_all - predicted
        
        results[regime] = {
            'fid_low': round(fid_low, 2),
            'fid_high': round(fid_high, 2),
            'fid_all': round(fid_all, 2),
            'delta_low': round(delta_low, 2),
            'delta_high': round(delta_high, 2),
            'predicted_all': round(predicted, 2),
            'deviation': round(deviation, 2),
            'interaction_type': 'sub-additive' if deviation < -1 else ('super-additive' if deviation > 1 else 'approximately additive')
        }
    return results

independence_results = {
    '1k': compute_independence(indep_data['1k'], '1k'),
    '2k': compute_independence(indep_data['2k'], '2k'),
}
print(f"\nIndependence results computed:")
for kimg, res in independence_results.items():
    for regime in ['tmid', 'argmin', 'thresh']:
        if regime in res:
            r = res[regime]
            print(f"  {kimg} {regime}: predicted={r['predicted_all']:.2f}, actual={r['fid_all']:.2f}, "
                  f"deviation={r['deviation']:+.2f} ({r['interaction_type']})")

# ============================================================
# 4. Build interpolators for 2-domain curves (2k kimg)
# ============================================================
# Per-bucket T=1 FID and interpolation function
bucket_t1_fid = {}
bucket_interp = {}  # will store (Ts, FIDs) sorted arrays for np.interp

for b in range(1, 8):
    bs = str(b)
    d = analysis['data_2k'][bs]
    Ts = np.array(d['T'])
    fids = np.array(d['FID'])
    # Sort by T
    sort_idx = np.argsort(Ts)
    Ts = Ts[sort_idx]
    fids = fids[sort_idx]
    bucket_interp[b] = (Ts, fids)
    # T=1 FID: use the actual T=1.0 value
    t1_idx = np.argmin(np.abs(Ts - 1.0))
    bucket_t1_fid[b] = fids[t1_idx]

baseline_2domain = summary['metadata']['baseline_2domain_mean_T1']  # 26.82
print(f"\nBaseline (2-domain mean T=1): {baseline_2domain}")
print(f"Per-bucket T=1 FIDs: {', '.join(f'B{b}={bucket_t1_fid[b]:.2f}' for b in range(1,8))}")

def additive_prediction(tvec, baseline=baseline_2domain):
    """Compute additive prediction for a T-vector using 2-domain curves."""
    total_delta = 0.0
    for b in range(1, 8):
        T_b = tvec[b-1]
        Ts, fids = bucket_interp[b]
        fid_at_T = float(np.interp(T_b, Ts, fids))
        delta_b = fid_at_T - bucket_t1_fid[b]
        total_delta += delta_b
    return baseline + total_delta

# ============================================================
# 5. Compute additive predictions for ALL BO points
# ============================================================
bo_points = []
for rnd in ['R2', 'R3', 'R4']:
    for pname, pdata in sorted(summary['bo_results'][rnd].items()):
        tvec = pdata['tvec']
        fid = pdata['fid']
        pred = additive_prediction(tvec)
        n_active = sum(1 for t in tvec if t < 0.999)
        bo_points.append({
            'name': f'{rnd.lower()}_{pname}',
            'round': int(rnd[1]),
            'tvec': tvec,
            'fid': fid,
            'additive_pred': round(pred, 3),
            'delta': round(fid - pred, 3),
            'n_active': n_active,
        })

# Also add the independence test points (2k only - argmin_low, argmin_high, all_argmin)
indep_tvecs_2k = {
    'argmin_low': [0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0],
    'argmin_high': [1.0, 1.0, 1.0, 0.97, 0.99, 0.99, 1.0],
    'all_argmin': [0.95, 0.95, 1.0, 0.97, 0.99, 0.99, 1.0],
}
for iname, tvec in indep_tvecs_2k.items():
    fid_val = indep_data['2k'][iname.replace('all_argmin', 'argmin_all')]['fid']
    if fid_val is not None:
        pred = additive_prediction(tvec)
        bo_points.append({
            'name': f'indep_{iname}',
            'round': 0,  # independence test
            'tvec': tvec,
            'fid': round(fid_val, 2),
            'additive_pred': round(pred, 3),
            'delta': round(fid_val - pred, 3),
            'n_active': sum(1 for t in tvec if t < 0.999),
        })

# Add confound tests
for cname, cdata in summary['confound_tests'].items():
    tvec = cdata['tvec']
    fid = cdata['fid']
    pred = additive_prediction(tvec)
    bo_points.append({
        'name': cname.replace('celeba_', ''),
        'round': -1,  # confound
        'tvec': tvec,
        'fid': fid,
        'additive_pred': round(pred, 3),
        'delta': round(fid - pred, 3),
        'n_active': sum(1 for t in tvec if t < 0.999),
    })

# Sort by FID
bo_points.sort(key=lambda x: x['fid'])

print(f"\n{len(bo_points)} total points (BO + independence + confound)")
print(f"Best: {bo_points[0]['name']} = {bo_points[0]['fid']}")
print(f"Worst: {bo_points[-1]['name']} = {bo_points[-1]['fid']}")

# Summary stats for BO-only points
bo_only = [p for p in bo_points if p['round'] >= 2]
deltas = [p['delta'] for p in bo_only]
print(f"\nBO-only stats ({len(bo_only)} points):")
print(f"  Mean delta: {np.mean(deltas):+.2f}")
print(f"  Std delta: {np.std(deltas):.2f}")
print(f"  Super-additive: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

# ============================================================
# 6. Organize context sweep data with matched 2-domain curves
# ============================================================
context_sweep_data = {}
for bucket_name in ['B2', 'B3']:
    b = int(bucket_name[1])
    ctx = summary['context_sweeps'][bucket_name]
    
    # Get 2-domain data for this bucket
    d2 = analysis['data_2k'][str(b)]
    td_Ts = d2['T']
    td_FIDs = d2['FID']
    
    context_sweep_data[bucket_name] = {
        'context_T': sorted([float(t) for t in ctx.keys()]),
        'context_FID': [ctx[t] for t in sorted(ctx.keys(), key=float)],
        'twodomain_T': td_Ts,
        'twodomain_FID': td_FIDs,
    }

# ============================================================
# 7. Best configurations ranking
# ============================================================
rankings = [
    {'name': 'Best single-bucket (b2_T096)', 'fid': 25.62, 'source': '2-domain'},
]
# Add top 5 BO points
for p in bo_only[:5]:
    rankings.append({'name': f'BO {p["name"]}', 'fid': p['fid'], 'source': 'BO'})
rankings.append({'name': 'All-argmin independence', 'fid': 26.22, 'source': 'independence'})
rankings.append({'name': 'Baseline (mean T=1)', 'fid': baseline_2domain, 'source': 'baseline'})
rankings.append({'name': 'Multi-bucket baseline (all T=1)', 'fid': 28.35, 'source': 'multi-bucket baseline'})
rankings.sort(key=lambda x: x['fid'])

# ============================================================
# 8. Save everything
# ============================================================
output = {
    'bo_points': bo_points,
    'independence_results': independence_results,
    'context_sweep_data': context_sweep_data,
    'confound_summary': {
        'b2only': {'multi_fid': 26.19, 'twodomain_fid': 25.62, 'gap': 0.57},
        'b1only': {'multi_fid': 27.15, 'twodomain_fid': 25.76, 'gap': 1.39},
        'b1b2': {'multi_fid': 26.20, 'b2only_fid': 26.19, 'gap': 0.01},
    },
    'rankings': rankings,
    'metadata': {
        'baseline_2domain': baseline_2domain,
        'baseline_multibucket': 28.35,
        'best_single_bucket_fid': 25.62,
        'n_bo_points': len(bo_only),
        'noise_floor': 1.74,
    }
}

outpath = f'{GEN_DIR}/7d_visualization_data.json'
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {outpath}")

# Also save updated summary
with open(f'{GEN_DIR}/all_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Updated all_results_summary.json (R4 now has {len(summary['bo_results']['R4'])} points)")
