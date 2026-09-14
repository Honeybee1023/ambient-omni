"""Top-down withdrawal schedule: blurred images usable only below an upper noise cutoff.

Checks:
  1. the cutoff t_hi(progress) follows the tau table (level t withdrawn tau(t) kimg before the end);
  2. t_hi never increases over training;
  3. the sampler never yields a blurred image at or above the cutoff, and never at all once t_hi = 0;
  4. an unsorted / decreasing tau table is refused.

Run:  python tests/test_topdown_schedule.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torch_utils.misc import InfiniteSampler
# training_loop.py imports wandb/diffusers at module scope (absent on the review
# machine), so lift the two pure functions out of the real source with ast --
# this still tests the committed code, not a copy.
import ast
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "training_loop.py")
_tree = ast.parse(open(_SRC).read())
_fns = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in ("compute_topdown_band", "topdown_band_sigma")]
_ns = {"np": np}
exec(compile(ast.Module(body=_fns, type_ignores=[]), _SRC, "exec"), _ns)
compute_topdown_band, topdown_band_sigma = _ns["compute_topdown_band"], _ns["topdown_band_sigma"]

TAU = [[0.0, 100], [0.2, 100], [0.3, 300], [0.5, 400], [0.7, 500], [1.0, 600]]
SCHED = {"type": "topdown", "tau_points": TAU, "total_kimg": 2000}
S_MAX = 4.0
BUFFER = (1 + 1 / (S_MAX - 1)) ** 0.5


class FakeDataset:
    def __init__(self, fnames):
        self._image_fnames = list(fnames); self.annotations = {}
    def __len__(self):
        return len(self._image_fnames)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else "")); return ok


def main():
    ok = True
    exp = {0: 1.0, 1400: 1.0, 1450: 0.85, 1500: 0.7, 1600: 0.5, 1700: 0.3, 1850: 0.225, 1901: 0.0, 2000: 0.0}
    for k, want in exp.items():
        got = compute_topdown_band(SCHED, k / 2000)
        ok &= check(f"t_hi at {k} kimg", abs(got - want) < 2e-3, f"got {got:.4f} want {want}")
    ths = [compute_topdown_band(SCHED, p) for p in np.linspace(0, 1, 2001)]
    ok &= check("t_hi non-increasing", all(b <= a + 1e-12 for a, b in zip(ths, ths[1:])))
    ok &= check("inf cutoff before withdrawal", topdown_band_sigma(1.0) == float("inf"))
    ok &= check("zero cutoff after", topdown_band_sigma(0.0) == 0.0)

    for t_hi in (0.5, 0.0):
        np.random.seed(0)
        band = topdown_band_sigma(t_hi)
        ds = FakeDataset(["clean_0.png", "blur_0.png"])
        ds.annotations = {"clean_0.png": (0.0, 0.0), "blur_0.png": (0.0, 0.0, band)}
        s = InfiniteSampler(dataset=ds, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0, s_max=S_MAX)
        it = iter(s); blur = []; clean = []
        for _ in range(40000):
            f = ds._image_fnames[next(it)]
            (blur if f.startswith("blur") else clean).append(s.sampled_sigmas[f])
        blur = np.array(blur); clean = np.array(clean)
        if t_hi == 0.0:
            ok &= check("t_hi=0: blurred image never yielded", len(blur) == 0, f"{len(blur)} draws")
        else:
            ok &= check("t_hi=0.5: blurred only below cutoff", len(blur) > 0 and blur.max() < band * BUFFER,
                        f"n={len(blur)} max={blur.max():.3f} edge={band*BUFFER:.3f}")
            ok &= check("t_hi=0.5: clean still used above cutoff", (clean > band * BUFFER).any())
    try:
        compute_topdown_band({"type": "topdown", "tau_points": [[0, 500], [1, 100]]}, 0.9); ok &= check("decreasing tau refused", False)
    except ValueError:
        ok &= check("decreasing tau refused", True)
    print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
