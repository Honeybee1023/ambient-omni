"""Build the ImageNet bucket file from per-image quality scores (Q-Align, see scripts/qalign_score.py).

Per class (ImageNet-1k label = sorted synset index, as dataset_tool assigns it):
  - clean (bucket 0) = the top 10% of the class by quality score (round(0.1 * n), ties broken by index);
  - the other 90% split by 1-D k-means (k = 3) on the score, buckets ordered best -> worst (1 = highest centre).
Short-run subset (1/10 of the data, stratified): within every (class, bucket) cell a seeded random round(n / 10)
images. Held-out clean = clean images outside the subset (the FID reference for picking among short runs).

Dataset index i is the i-th image of the sorted file list (EDM2 dataset_tool order), the same index the training
dataset uses (sorted archive names). Output npz:
  bucket int8[N], score float32[N], label int16[N], subset_short bool[N], heldout_clean_short bool[N]

Usage: python imagenet_buckets.py <qalign_chunk_dir> <class_counts.txt> <out.npz> [--frac-clean 0.1] [--k 3]
"""
import argparse, glob, json

import numpy as np

N_IMAGENET = 1281167


def kmeans_1d(x, k, iters=300):
    """Lloyd's algorithm in 1-D from quantile starts (deterministic). Returns labels ordered by centre, highest = 0."""
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    rank = np.argsort(-c)                      # centre order, highest first
    remap = np.empty(k, int); remap[rank] = np.arange(k)
    return remap[lab], c[rank]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chunks'); ap.add_argument('counts'); ap.add_argument('out')
    ap.add_argument('--frac-clean', type=float, default=0.1)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--short-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=20261028)
    a = ap.parse_args()

    score = np.full(N_IMAGENET, np.nan, np.float32)
    for f in sorted(glob.glob(f'{a.chunks}/chunk_*.npz')):
        z = np.load(f); score[z['index']] = z['score']
    if np.isnan(score).any():
        raise SystemExit(f'{int(np.isnan(score).sum())} images have no score yet')
    counts = [int(l.split()[1]) for l in open(a.counts)]
    assert len(counts) == 1000 and sum(counts) == N_IMAGENET
    label = np.repeat(np.arange(1000), counts).astype(np.int16)

    bucket = np.zeros(N_IMAGENET, np.int8)
    centres = []
    for c in range(1000):
        idx = np.flatnonzero(label == c)
        s = score[idx]
        order = np.lexsort((idx, -s))              # best first; ties -> lower index first
        n_clean = int(round(a.frac_clean * len(idx)))
        rest = idx[order[n_clean:]]
        lab, cen = kmeans_1d(score[rest].astype(np.float64), a.k)
        bucket[rest] = (lab + 1).astype(np.int8)
        centres.append(cen)

    rng = np.random.RandomState(a.seed)
    subset = np.zeros(N_IMAGENET, bool)
    for c in range(1000):
        for b in range(a.k + 1):
            cell = np.flatnonzero((label == c) & (bucket == b))
            m = int(round(a.short_frac * len(cell)))
            subset[rng.choice(cell, m, replace=False)] = True
    heldout = (bucket == 0) & ~subset

    np.savez(a.out, bucket=bucket, score=score, label=label, subset_short=subset, heldout_clean_short=heldout)
    summ = {'buckets': {int(b): int((bucket == b).sum()) for b in range(a.k + 1)},
            'bucket_share': {int(b): round(float((bucket == b).mean()), 4) for b in range(a.k + 1)},
            'subset_short': {int(b): int((subset & (bucket == b)).sum()) for b in range(a.k + 1)},
            'heldout_clean_short': int(heldout.sum()),
            'mean_score_per_bucket': {int(b): round(float(score[bucket == b].mean()), 3) for b in range(a.k + 1)},
            'kmeans_centre_quantiles': np.quantile(np.array(centres), [0.05, 0.5, 0.95], axis=0).round(3).tolist(),
            'per_class_bucket_share_min_med_max': {int(b): np.quantile(np.bincount(label[bucket == b], minlength=1000)
                                                                        / np.bincount(label), [0, .5, 1]).round(3).tolist()
                                                   for b in range(1, a.k + 1)}}
    json.dump(summ, open(a.out.replace('.npz', '_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
