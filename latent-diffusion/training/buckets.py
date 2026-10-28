"""Per-bucket noise floors for ImageNet: our time-axis method and the matched Ambient-o baseline.

Every training image belongs to one bucket: 0 = clean, 1..K = bad buckets ordered best -> worst
(per class: top 10% by quality = clean, the rest split by 1-D k-means; see scripts/imagenet_buckets.py).
A bucket's noise floor sigma_min says the image is used only at sigma_t > sigma_min (ambient loss with
x_tn = x_0 + sigma_min * n); sigma_min = 0 means "used as a clean image", inf means "never used".

Schedule specs (JSON):
  {"type": "per_bucket", "buckets": {"1": {"phases": [[0, 0.0], [0.42, 0.9]]}, ...}}
      Step function per bucket over training progress = cur_nimg / horizon_nimg. Each phase value is a
      noise LEVEL l in [0, 1] (a quantile of the training noise distribution), mapped to
      sigma = exp(P_std * Phi^-1(l) + P_mean); l = 0 -> sigma 0 (clean), l = 1 -> never used.
      Same definition as the pixel-space AFHQ/CelebA searches (pixel-diffusion/short_search.py).
  {"type": "fixed", "sigma_min": {"1": 0.2, "2": 0.2, "3": 0.2}}
      Constant floors (the Ambient-o baseline: every bad image at sigma_min = 0.2 for the whole run).
Bucket 0 is always clean and must not appear in a spec.
"""

import collections
import json
import math

import numpy as np
import torch
from scipy.stats import norm


def level_to_sigma(level, P_mean, P_std):
    level = float(level)
    if not 0.0 <= level <= 1.0:
        raise ValueError(f'noise level must be in [0, 1], got {level}')
    if level == 0.0:
        return 0.0
    if level == 1.0:
        return math.inf
    return float(math.exp(P_std * norm.ppf(level) + P_mean))


class BucketSchedule:
    def __init__(self, spec, num_buckets, P_mean, P_std, horizon_nimg=None):
        if isinstance(spec, str):
            spec = json.loads(spec)
        self.spec, self.K, self.P_mean, self.P_std = spec, int(num_buckets), P_mean, P_std
        self.horizon_nimg = horizon_nimg
        bad = [str(b) for b in range(1, self.K + 1)]
        if spec.get('type') == 'fixed':
            floors = spec.get('sigma_min', {})
            if sorted(floors) != sorted(bad):
                raise ValueError(f'fixed spec must give sigma_min for buckets {bad}, got {sorted(floors)}')
            self._fixed = np.array([0.0] + [float(floors[b]) for b in bad])
            self._phases = None
        elif spec.get('type') == 'per_bucket':
            groups = spec.get('buckets', {})
            if sorted(groups) != sorted(bad):
                raise ValueError(f'per_bucket spec must list buckets {bad}, got {sorted(groups)}')
            if not horizon_nimg:
                raise ValueError('per_bucket schedules need horizon_nimg (the run length the fractions refer to)')
            self._phases = []
            for b in bad:
                ph = groups[b]['phases']
                fr = [float(p[0]) for p in ph]
                if not ph or fr[0] != 0.0 or any(y < x for x, y in zip(fr, fr[1:])):
                    raise ValueError(f'bucket {b}: phases must start at 0 and be sorted, got {fr}')
                self._phases.append([(float(f), level_to_sigma(l, P_mean, P_std)) for f, l in ph])
            self._fixed = None
        else:
            raise ValueError(f'unknown schedule type: {spec.get("type")}')

    def floors(self, cur_nimg):
        """sigma_min for buckets 0..K at training position cur_nimg (array of K+1 floats)."""
        if self._fixed is not None:
            return self._fixed
        p = cur_nimg / self.horizon_nimg
        out = [0.0]
        for ph in self._phases:
            s = ph[0][1]
            for f, v in ph:
                if p >= f:
                    s = v
            out.append(s)
        return np.array(out)


def load_buckets(path, dataset_size):
    """Bucket file (npz from scripts/imagenet_buckets.py): bucket[i] for dataset index i (sorted dataset_tool order),
    optional subset_<name> boolean masks. Returns (bucket int8 array, dict of masks)."""
    z = np.load(path)
    bucket = z['bucket'].astype(np.int8)
    if len(bucket) != dataset_size:
        raise ValueError(f'bucket file has {len(bucket)} entries, dataset has {dataset_size}')
    masks = {k[len('subset_'):]: z[k].astype(bool) for k in z.files if k.startswith('subset_')}
    return bucket, masks


class BucketSampler(torch.utils.data.Sampler):
    """Ambient-o's keep_schedule sampler (torch_utils.misc.InfiniteSampler) with per-bucket, time-varying floors.

    As in Ambient-o: draw sigma ~ exp(N(P_mean, P_std)) first, then walk the shuffled order to the next image whose
    floor is below sigma, and train that image at that sigma. The noise-level distribution of training batches is
    therefore exactly EDM2's, and an image is skipped (for this pass) when it is not usable at the drawn sigma.

    Differences from InfiniteSampler, all deliberate:
      - floors come from BucketSchedule at the trainer's current position (self.cur_nimg, set every step);
      - the sigma and floor used for each yielded index are queued, so the trainer applies exactly the pair the
        sampler decided on even when the loader prefetches across a schedule change;
      - optional index subset (the 1/10 short-run data);
      - own RandomState (seed, rank, resume position) instead of the global numpy state;
      - resumable position: the trainer checkpoints the walk position of the last consumed sample, so a resumed run
        continues the same pass instead of restarting at cur_nimg (which, with skipping, is far behind the walk).
    """

    def __init__(self, dataset_size, bucket, schedule, rank=0, num_replicas=1, seed=0, P_mean=-0.4, P_std=1.0,
                 subset=None, start_pos=None, cur_nimg=0):
        super().__init__()
        self.N = int(dataset_size)
        self.bucket = np.asarray(bucket, dtype=np.int8)
        self.schedule = schedule
        self.rank, self.stride, self.seed = rank, num_replicas, seed
        self.P_mean, self.P_std = P_mean, P_std
        self.items = np.arange(self.N) if subset is None else np.flatnonzero(subset)
        self.cur_nimg = cur_nimg
        self.start = rank if start_pos is None else int(start_pos)
        # (raw_idx, sigma, sigma_min, walk position) per yielded index, in yield order. The DataLoader returns
        # samples in sampler order, so the trainer pops from the left and checks raw_idx matches.
        self.decided = collections.deque()

    def __iter__(self):
        rng = np.random.RandomState((self.seed * 1_000_003 + self.rank * 7919 + self.start) % (1 << 31))
        M = len(self.items)
        pos, epoch, order = self.start, None, None
        sigma = math.exp(rng.normal() * self.P_std + self.P_mean)
        floors, floors_at = None, None
        while True:
            if epoch != pos // M:
                epoch = pos // M
                order = self.items[np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).permutation(M)]
            if floors_at != self.cur_nimg:
                floors, floors_at = self.schedule.floors(self.cur_nimg), self.cur_nimg
            i = int(order[pos % M])
            fl = floors[self.bucket[i]]
            if sigma > fl:
                self.decided.append((i, sigma, fl, pos))
                yield i
                sigma = math.exp(rng.normal() * self.P_std + self.P_mean)
            pos += self.stride
