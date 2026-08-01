"""Diagnostic: does the dynamic-T schedule actually reach the sampler?

Background
----------
`training_loop.py` builds a local `annotations` dict, then does:

    dataset_obj.annotations = dict(annotations)      # <-- COPY, made once

Inside the training loop the dynamic schedule writes the newly scheduled
sigma_min back into the *local* `annotations` dict:

    annotations[fname] = (current_sigma_min, 0.0)

But `InfiniteSampler.__iter__` (torch_utils/misc.py) decides which images are
*eligible* at the currently sampled sigma by reading `self.dataset.annotations`
-- i.e. the frozen copy, which still holds the 999.0 sentinel.

This script checks two things without needing a GPU or a real dataset:
  1. Are corrupt images (sigma_min=999) ever yielded by the sampler?
  2. Do mutations of the local dict propagate to `dataset.annotations`?

Run:  python tests/test_dynamic_t_sampler.py
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch_utils.misc import InfiniteSampler


class FakeDataset:
    """Minimal stand-in for the real dataset: the sampler only needs
    __len__, _image_fnames and .annotations."""

    def __init__(self, fnames):
        self._image_fnames = list(fnames)
        self.annotations = {}

    def __len__(self):
        return len(self._image_fnames)


SENTINEL = 999.0
S_MAX = 4  # matches the --s_max=4 used by the training scripts


def build(n_clean=50, n_corrupt=450):
    fnames = [f"b0_{i}.jpg" for i in range(n_clean)]
    fnames += [f"b5_{i}.jpg" for i in range(n_corrupt)]
    ds = FakeDataset(fnames)

    # This mirrors how training_loop.py loads annotations.jsonl for the
    # dynamic-T datasets: clean -> 0.0, corrupt -> 999.0 sentinel.
    annotations = defaultdict(lambda: (0.0, 0.0))
    for f in fnames:
        annotations[f] = (0.0, 0.0) if f.startswith("b0_") else (SENTINEL, 0.0)

    ds.annotations = dict(annotations)  # the copy made at training_loop.py:248
    return ds, annotations


def main():
    ds, annotations = build()
    n_draw = 20000

    # ---- Check 1: does the sentinel keep corrupt images out of the sampler? --
    sampler = InfiniteSampler(ds, rank=0, num_replicas=1, shuffle=True,
                              seed=0, window_size=0.5, s_max=S_MAX)
    it = iter(sampler)
    seen = [ds._image_fnames[next(it)] for _ in range(n_draw)]
    n_corrupt_seen = sum(1 for f in seen if f.startswith("b5_"))

    print("=" * 68)
    print("CHECK 1 -- sampler eligibility with sigma_min=999 sentinel")
    print("=" * 68)
    print(f"  drew {n_draw} samples from a set that is 90% corrupt")
    print(f"  corrupt images actually yielded: {n_corrupt_seen}")
    print(f"  clean   images actually yielded: {n_draw - n_corrupt_seen}")

    # Analytic sanity check on the odds.
    buffer_factor = (1 + 1 / (S_MAX - 1)) ** 0.5
    # sigma = exp(z * 1.2 - 1.2); need sigma > 999 * buffer_factor
    z_needed = (np.log(SENTINEL * buffer_factor) + 1.2) / 1.2
    from scipy.stats import norm
    p = norm.sf(z_needed)
    print(f"  -> a corrupt image needs sigma > {SENTINEL * buffer_factor:.1f},")
    print(f"     i.e. a {z_needed:.2f}-sigma draw; P = {p:.2e} per attempt")

    # ---- Check 2: does the dynamic update reach the sampler? -----------------
    print()
    print("=" * 68)
    print("CHECK 2 -- does the scheduled sigma_min propagate to the sampler?")
    print("=" * 68)
    scheduled = 0.301  # what T=0.5 maps to
    for f in list(annotations):
        if annotations[f][0] >= 900:
            annotations[f] = (scheduled, 0.0)

    a_local = annotations["b5_0.jpg"][0]
    a_sampler = ds.annotations["b5_0.jpg"][0]
    print(f"  after schedule update, local annotations['b5_0.jpg'] = {a_local}")
    print(f"  but dataset.annotations['b5_0.jpg']                  = {a_sampler}")
    print(f"  propagated: {a_local == a_sampler}")

    # Re-run the sampler now that the *dataset* copy is fixed, to show the
    # eligibility filter works fine once it sees the real value.
    ds2, ann2 = build()
    for f in list(ann2):
        if ann2[f][0] >= 900:
            ann2[f] = (scheduled, 0.0)
    ds2.annotations = dict(ann2)
    sampler2 = InfiniteSampler(ds2, rank=0, num_replicas=1, shuffle=True,
                               seed=0, window_size=0.5, s_max=S_MAX)
    it2 = iter(sampler2)
    seen2 = [ds2._image_fnames[next(it2)] for _ in range(n_draw)]
    n_corrupt_seen2 = sum(1 for f in seen2 if f.startswith("b5_"))
    print()
    print(f"  control: with sigma_min={scheduled} in dataset.annotations,")
    print(f"           corrupt images yielded = {n_corrupt_seen2} / {n_draw}")

    # ---- Check 3: the fix -- share one dict instead of copying ---------------
    print()
    print("=" * 68)
    print("CHECK 3 -- the fix: dataset.annotations shares the local dict")
    print("=" * 68)
    ds3, ann3 = build()
    ds3.annotations = ann3  # <-- the fix in training_loop.py (no dict() copy)
    sampler3 = InfiniteSampler(ds3, rank=0, num_replicas=1, shuffle=True,
                               seed=0, window_size=0.5, s_max=S_MAX)
    it3 = iter(sampler3)

    # phase 1: schedule still at the sentinel -> corrupt images excluded
    before = sum(1 for _ in range(2000)
                 if ds3._image_fnames[next(it3)].startswith("b5_"))

    # phase 2: schedule fires mid-training, exactly as the training loop does
    for f in list(ann3):
        if ann3[f][0] >= 900:
            ann3[f] = (scheduled, 0.0)

    after = sum(1 for _ in range(2000)
                if ds3._image_fnames[next(it3)].startswith("b5_"))

    print(f"  corrupt yielded before schedule update: {before} / 2000")
    print(f"  corrupt yielded after  schedule update: {after} / 2000")
    print(f"  schedule reaches a live sampler: {before == 0 and after > 0}")

    print()
    print("=" * 68)
    verdict_1 = n_corrupt_seen == 0
    verdict_2 = a_local != a_sampler
    verdict_3 = before == 0 and after > 0
    if verdict_1 and verdict_2:
        print("VERDICT: BUG CONFIRMED (checks 1 and 2 reproduce the failure).")
    else:
        print("VERDICT: could not reproduce the bug; investigate further.")
    print(f"         Fix validated by check 3: {verdict_3}")
    print("=" * 68)
    return 0 if (verdict_1 and verdict_2 and verdict_3) else 1


if __name__ == "__main__":
    sys.exit(main())
