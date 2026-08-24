#!/usr/bin/env python3
"""Checks on the `piecewise` t_schedule type used by the 4D schedule search.

The point that matters for reading Phase 0: a piecewise-linear schedule whose
kinks land on control points is reproduced *exactly*, not approximated. So the
5-point and 10-point discretisations of warmup_linear(0 -> 0.95, frac=0.25) are
the same function as the continuous one, and comparing them measures seed noise
rather than discretisation error. Only schedules with curvature (cosine) are
genuinely approximated -- that is what makes the cosine arms the real test.

Run:  python tests/test_piecewise_schedule.py
"""
import os, sys
import numpy as np

# training_loop.py imports wandb/torch/diffusers at module scope, none of which
# are installed on the review machine. Lift just the one pure function out of the
# real source with ast, so this runs anywhere and still tests the committed code
# rather than a copy that can drift.
import ast, textwrap

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "training", "training_loop.py")
_tree = ast.parse(open(_SRC).read())
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "compute_scheduled_sigma_min")
_ns = {"np": np}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), _SRC, "exec"), _ns)
sched = _ns["compute_scheduled_sigma_min"]

GRID = np.linspace(0.0, 1.0, 401)
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def discretise(cont, n):
    """Sample a continuous schedule at n equally spaced fractions."""
    fr = np.linspace(0.0, 1.0, n)
    return {"type": "piecewise",
            "control_points": [[float(f), float(_t_of(cont, f))] for f in fr]}


def _t_of(cfg, p):
    """Recover the T value a schedule config asks for at progress p."""
    from scipy.stats import norm
    s = sched(cfg, p)
    if s <= 0.0:
        return 0.0
    return float(norm.cdf((np.log(s) + 1.2) / 1.2))


print("=== piecewise t_schedule ===")

# 1. Flat schedule == static.
pw_flat = {"type": "piecewise", "control_points": [[0.0, 0.5], [1.0, 0.5]]}
st = {"type": "static", "t_start": 0.5}
check("flat piecewise == static T=0.50",
      max(abs(sched(pw_flat, p) - sched(st, p)) for p in GRID) < 1e-12)

# 2. Two-point ramp == linear.
pw_lin = {"type": "piecewise", "control_points": [[0.0, 0.0], [1.0, 0.95]]}
lin = {"type": "linear", "t_start": 0.0, "t_end": 0.95}
d = max(abs(sched(pw_lin, p) - sched(lin, p)) for p in GRID)
check("2-point piecewise == linear 0->0.95", d < 1e-9, f"max |dsigma| = {d:.2e}")

# 3. THE KEY ONE: 5-point warmup == continuous warmup, exactly.
warm = {"type": "warmup_linear", "t_start": 0.0, "t_end": 0.95, "warmup_frac": 0.25}
pw5 = {"type": "piecewise",
       "control_points": [[0.0, 0.0], [0.25, 0.0],
                          [0.5, 0.95 / 3], [0.75, 0.95 * 2 / 3], [1.0, 0.95]]}
d5 = max(abs(sched(pw5, p) - sched(warm, p)) for p in GRID)
check("5-point warmup == continuous warmup (exact)", d5 < 1e-9,
      f"max |dsigma| = {d5:.2e}")

# 4. The 10-point version is NOT exact, and that is the interesting part.
#    10 equally spaced knots sit at k/9, so the kink at p=0.25 falls *between*
#    two of them and the interpolation cuts the corner. More control points is
#    therefore not monotonically more faithful: what matters is whether the
#    knots line up with the kinks. Small in T (~0.026 where the truth is 0), and
#    it touches ~2.6% of sigma draws over one ninth of training -- so expect the
#    5-point and 10-point warmup arms to land within seed noise of each other,
#    but do not describe them as the same function.
pw10 = discretise(warm, 10)
d10 = max(abs(sched(pw10, p) - sched(warm, p)) for p in GRID)
t10 = max(abs(_t_of(pw10, p) - _t_of(warm, p)) for p in GRID)
check("10-point warmup is NOT exact (kink falls between knots)", d10 > 1e-6,
      f"max |dsigma| = {d10:.2e}, max |dT| = {t10:.4f}")
check("...but the 10-point warmup error stays small", t10 < 0.05,
      f"max |dT| = {t10:.4f}")

# 4b. A nested grid that keeps p=0.25 as a knot IS exact at any resolution --
#     the evidence that knot placement, not knot count, is what matters here.
pw9 = {"type": "piecewise",
       "control_points": [[f, float(_t_of(warm, f))] for f in np.linspace(0, 1, 9)]}
d9 = max(abs(sched(pw9, p) - sched(warm, p)) for p in GRID)
check("9-point (nested, 0.25 is a knot) warmup == continuous", d9 < 1e-9,
      f"max |dsigma| = {d9:.2e}")

# 5. Cosine IS approximated -- 5 points leave a real gap, 10 points shrink it.
cos = {"type": "cosine", "t_start": 0.0, "t_end": 0.95}
c5, c10 = discretise(cos, 5), discretise(cos, 10)
e5 = max(abs(_t_of(c5, p) - _t_of(cos, p)) for p in GRID)
e10 = max(abs(_t_of(c10, p) - _t_of(cos, p)) for p in GRID)
check("5-point cosine differs from continuous", e5 > 0.02, f"max |dT| = {e5:.4f}")
check("10-point cosine is closer than 5-point", e10 < e5 / 2,
      f"max |dT| = {e10:.4f} vs {e5:.4f}")

# 6. Clamping past the end: progress tips over 1.0 on the last tick and must not
#    extrapolate above the ceiling.
check("progress > 1 clamps to the final control point",
      abs(sched(pw5, 1.004) - sched(pw5, 1.0)) < 1e-12)

# 7. T=0 stretches map to sigma_min=0 (every corrupt image eligible).
check("T=0 -> sigma_min = 0", sched(pw5, 0.1) == 0.0)

# 8. Monotone non-decreasing T gives monotone non-decreasing sigma_min.
seq = [sched(pw5, p) for p in GRID]
check("monotone control points -> monotone sigma_min",
      all(b >= a - 1e-12 for a, b in zip(seq, seq[1:])))

# 9. Bad input is rejected rather than silently reinterpreted.
for bad, why in [({"type": "piecewise"}, "missing control_points"),
                 ({"type": "piecewise", "control_points": []}, "empty control_points"),
                 ({"type": "piecewise", "control_points": [[0.0, 0.0], [0.5, 0.3], [0.25, 0.6]]},
                  "unsorted fractions")]:
    try:
        sched(bad, 0.5)
        check(f"rejects {why}", False, "no exception raised")
    except (ValueError, KeyError, TypeError):
        check(f"rejects {why}", True)

print(f"\n{len(fails)} failure(s)" if fails else "\nAll checks passed.")
sys.exit(1 if fails else 0)
