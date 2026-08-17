"""Does the restricted-bucket band actually gate the sampler?

Background
----------
Before restricted buckets, `InfiniteSampler` gated on

    (sigma > sigma_min*buffer) or (sigma < sigma_max)

which is an OR: `sigma_max` can only ever *widen* eligibility (it adds the
low-noise window used by ambient-crops). There was no way to say "this image is
usable between two noise levels and nowhere else", which is what the
single-bottleneck validation experiment needs.

The third annotation field `sigma_band_max` is AND'd on top:

    ((sigma > sigma_min*buffer) or (sigma < sigma_max)) and (sigma < band_max*buffer)

This checks:
  1. A banded image is *never* yielded at a sigma outside its band.
  2. Two consecutive buckets tile the sigma axis -- no gap, no overlap.
  3. A degenerate band (band_max == sigma_min) makes the bucket unusable.
  4. Annotations without the field behave exactly as they did before.

Run:  python tests/test_restricted_bucket_sampler.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch_utils.misc import InfiniteSampler

S_MAX = 4.0
BUFFER = (1 + 1 / (S_MAX - 1)) ** 0.5  # 1.1547, what --s_max=4 gives in training


class FakeDataset:
    def __init__(self, fnames):
        self._image_fnames = list(fnames)
        self.annotations = {}

    def __len__(self):
        return len(self._image_fnames)


def collect(annotations, n_draws=200000, seed=0):
    """Run the sampler and record, per filename, the sigmas it was used at."""
    # InfiniteSampler draws sigma from the *global* numpy RNG, not from its own
    # seeded RandomState (which only shuffles the order). Without this, two
    # collect() calls continue the same stream and are not comparable.
    np.random.seed(seed)
    ds = FakeDataset(sorted(annotations.keys()))
    ds.annotations = dict(annotations)
    sampler = InfiniteSampler(dataset=ds, rank=0, num_replicas=1, shuffle=True,
                              seed=seed, window_size=0, s_max=S_MAX)
    seen = {f: [] for f in annotations}
    it = iter(sampler)
    for _ in range(n_draws):
        idx = next(it)
        fname = ds._image_fnames[idx]
        seen[fname].append(sampler.sampled_sigmas[fname])
    return {f: np.array(v) for f, v in seen.items()}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    ok = True

    # sigma(T) = exp(1.2 * Phi^-1(T) - 1.2); values for the real sweep.
    sig_055 = 0.3251   # T = 0.55, the fixed B2 threshold
    sig_080 = 1.0464   # T = 0.80, a swept threshold
    sig_off = 12.3287  # T = 0.999, "inactive"

    print("\n1. Banded image is confined to its band")
    anns = {
        "clean.jpg": (0.0, 0.0, 1e6),                # always eligible
        "fixed.jpg": (sig_055, 0.0, sig_080),        # band [0.55, 0.80)
        "swept.jpg": (sig_080, 0.0, 1e6),            # [0.80, inf)
    }
    seen = collect(anns)
    lo, hi = sig_055 * BUFFER, sig_080 * BUFFER
    f = seen["fixed.jpg"]
    ok &= check("fixed.jpg never below its band", f.size and f.min() > lo,
                f"min={f.min():.4f} vs lo={lo:.4f}")
    ok &= check("fixed.jpg never above its band", f.size and f.max() < hi,
                f"max={f.max():.4f} vs hi={hi:.4f}")
    ok &= check("fixed.jpg was actually used", f.size > 100, f"n={f.size}")

    print("\n2. Consecutive buckets tile the axis (no gap, no overlap)")
    s = seen["swept.jpg"]
    ok &= check("swept.jpg starts where fixed.jpg stops", s.min() > hi * 0.999,
                f"swept min={s.min():.4f} vs fixed hi={hi:.4f}")
    ok &= check("no overlap between the two", f.max() < s.min(),
                f"fixed max={f.max():.4f} < swept min={s.min():.4f}")
    # No sigma in the bulk of the distribution falls between the two buckets.
    # (Only the bulk: out in the lognormal tail the draws are genuinely sparse,
    # so a large raw gap there says nothing about coverage.)
    both = np.sort(np.concatenate([f, s]))
    bulk = both[both < np.quantile(both, 0.99)]
    ok &= check("no coverage hole across the seam",
                bool(((bulk[1:] - bulk[:-1]) < 0.05).all()),
                f"largest gap in bulk={float((bulk[1:] - bulk[:-1]).max()):.5f}")

    print("\n3. Degenerate band (band_max == sigma_min) is unusable")
    anns_deg = {
        "clean.jpg": (0.0, 0.0, 1e6),
        "fixed.jpg": (sig_055, 0.0, sig_055),        # empty band
        "swept.jpg": (sig_055, 0.0, 1e6),
    }
    seen_deg = collect(anns_deg)
    ok &= check("empty-band image never yielded", seen_deg["fixed.jpg"].size == 0,
                f"n={seen_deg['fixed.jpg'].size}")
    ok &= check("swept bucket still covers [T*, inf)", seen_deg["swept.jpg"].size > 100,
                f"n={seen_deg['swept.jpg'].size}")

    print("\n4. Backward compatibility: 2-tuples gate exactly as before")
    legacy = {
        "clean.jpg": (0.0, 0.0),
        "blur.jpg": (sig_055, 0.0),
        "inactive.jpg": (sig_off, 0.0),
    }
    seen_legacy = collect(legacy)
    b = seen_legacy["blur.jpg"]
    ok &= check("2-tuple blur image has no upper bound",
                b.size > 100 and b.max() > sig_080,
                f"n={b.size} max={b.max():.4f}")
    ok &= check("2-tuple blur image still respects sigma_min", b.min() > sig_055 * BUFFER,
                f"min={b.min():.4f}")
    # T=0.999 leaves ~0.1% of the sigma mass above the threshold, so "inactive"
    # means rare rather than absent.
    inactive_frac = seen_legacy["inactive.jpg"].size / sum(v.size for v in seen_legacy.values())
    ok &= check("inactive bucket gets <0.5% of draws", inactive_frac < 0.005,
                f"frac={inactive_frac:.5f}")

    # A banded annotation with an unreachable ceiling must match the 2-tuple exactly.
    same = collect({"clean.jpg": (0.0, 0.0), "blur.jpg": (sig_055, 0.0, 1e6),
                    "inactive.jpg": (sig_off, 0.0)})
    ok &= check("band_max=1e6 is identical to omitting the field",
                np.allclose(np.sort(same["blur.jpg"]), np.sort(b)),
                f"n={same['blur.jpg'].size} vs {b.size}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
