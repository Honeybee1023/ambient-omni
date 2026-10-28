"""The 8 short-run schedules for the ImageNet search (same space as the AFHQ/CelebA searches, 3 bad buckets).

One jump per bad bucket: before its jump time w the bucket is used as clean (level 0), from w on only above noise
level l. Monotone (non-crossing): the worse a bucket, the earlier AND at least as high it jumps,
  w3 <= w2 <= w1 in WHEN_RANGE,  l1 <= l2 <= l3 in LEVELS.
Space-filling design: one scrambled Sobol sequence (seed SOBOL_SEED, not used before), jump times sorted into monotone
order, levels stratified from the Sobol coordinates then sorted (pixel-diffusion/short_search.py space_filling, K=3).

Usage: python imagenet_search_design.py <outdir> [--n 8]
Writes <outdir>/s00.json ... (bucket_schedule specs for train_edm2.py --bucket_schedule=@file) and design.json.
"""
import argparse, json, os

import numpy as np
from scipy.stats import qmc

K = 3
LEVELS = (0.85, 0.90, 0.95, 1.0)
WHEN_RANGE = (0.10, 0.95)
SOBOL_SEED = 20261028


def space_filling(seed, n):
    s = qmc.Sobol(d=2 * K, scramble=True, seed=seed).random(n)
    w = np.sort(WHEN_RANGE[0] + (WHEN_RANGE[1] - WHEN_RANGE[0]) * s[:, :K], axis=1)[:, ::-1]
    lv = np.sort(np.array(LEVELS)[np.minimum((s[:, K:] * len(LEVELS)).astype(int), len(LEVELS) - 1)], axis=1)
    return w, lv


def spec(w, lv):
    return {'type': 'per_bucket',
            'buckets': {str(b + 1): {'phases': [[0, 0.0], [round(float(w[b]), 3), float(lv[b])]]} for b in range(K)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir'); ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    W, L = space_filling(SOBOL_SEED, a.n)
    design = []
    for i, (w, lv) in enumerate(zip(W, L)):
        sp = spec(w, lv)
        json.dump(sp, open(f'{a.outdir}/s{i:02d}.json', 'w'))
        design.append({'name': f's{i:02d}', 'when': [round(float(x), 3) for x in w], 'level': [float(x) for x in lv]})
        print(f's{i:02d}  jump at ' + ', '.join(f'b{b + 1} {w[b]:.2f}' for b in range(K))
              + '  | level ' + ', '.join(f'b{b + 1} {lv[b]:.2f}' for b in range(K)))
    json.dump({'sobol_seed': SOBOL_SEED, 'levels': LEVELS, 'when_range': WHEN_RANGE, 'runs': design},
              open(f'{a.outdir}/design.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
