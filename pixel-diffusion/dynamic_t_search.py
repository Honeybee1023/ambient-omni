#!/usr/bin/env python3
"""Generate the Phase 0 + Phase 1 schedule specs for the discrete dynamic-T search.

THE SEARCH SPACE
----------------
A schedule is 5 control points at training fractions [0, .25, .5, .75, 1] with
linear interpolation between them (`piecewise` t_schedule, see
training/training_loop.py). The first is pinned at T=0 -- every hand-crafted
winner so far starts by using all the data -- so a schedule is the 4-vector

    x = [T2, T3, T4, T5]        0 <= T2 <= T3 <= T4 <= T5 <= 1

WHAT THE DATASET IS
-------------------
All runs share ONE dataset, `celeba_dynamic_t_v2`: bucket b5 (26,014 blurred)
carries the sigma_min=999 sentinel that the schedule overwrites each iteration,
b0 (500 clean) is always eligible, every other bucket is parked at T=0.999. So
no per-schedule dataset is needed, and every number here is comparable to the
existing `celeba_v2b_b5_T*` static sweep and the `v2_*` hand-crafted runs.

READING PHASE 0 -- one thing to get right
-----------------------------------------
5 equally spaced knots land exactly on warmup_linear's kink at p=0.25, so the
5-point discretisation of warmup is that schedule EXACTLY, not an approximation
(tests/test_piecewise_schedule.py pins this). The 10-point grid sits at k/9 and
*misses* the kink, so it is slightly wrong -- max |dT| = 0.026, touching ~2.6%
of sigma draws over one ninth of training. Consequences:

  * warmup_pw5 vs warmup_cont measures SEED NOISE and validates the new code
    path. It does not measure discretisation error, and must not be read as
    "5 points are enough".
  * cosine has real curvature, so cosine_pw5 vs the known continuous cosine is
    the arm that actually tests whether 5 control points resolve a schedule.
    max |dT| = 0.033 at 5 points, 0.007 at 10.

Usage:  python dynamic_t_search.py            # write the manifest
        python dynamic_t_search.py --show     # print the table only
"""

import argparse, json, os

import numpy as np
from scipy.stats import norm, qmc

FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
T_FIRST = 0.0          # pinned: always start by using every image
SOBOL_SEED = 20260823  # fixed so the manifest is reproducible on any machine

# Seed-averaged MIND of the hand-crafted runs already on lysine, for reference
# only. n is the number of seeds -- never compare across different n (a 1-seed
# point carries ~sqrt(3)x the variance of a 3-seed one and wins an argmin by
# chance). See generated/mind_v2_*.json.
KNOWN = {
    "static_T050":        (0.035229, 3),
    "warmup25_0to095":    (0.029622, 3),
    "linear_0to095":      (0.031211, 3),
    "cosine_0to095":      (0.032169, 3),
    "twophase_0_050_095": (0.032425, 3),
    "warmup15_0to095":    (0.029527, 1),
    "warmup40_0to095":    (0.029433, 1),
    "clean_only":         (0.044789, 3),
}


# ----------------------------------------------------------- schedule maths --

def warmup_T(p, t_end=0.95, frac=0.25):
    return 0.0 if p < frac else t_end * (p - frac) / (1.0 - frac)


def cosine_T(p, t_end=0.95):
    return t_end * (1 - np.cos(np.pi * p)) / 2


def linear_T(p, t_end=0.95):
    return t_end * p


def twophase_T(p, t_mid=0.5, t_end=0.95):
    return t_mid * (p / 0.5) if p < 0.5 else t_mid + (t_end - t_mid) * ((p - 0.5) / 0.5)


def sample_at(fn, n):
    """Sample a continuous schedule at n equally spaced fractions in [0,1]."""
    return [[round(float(p), 6), round(float(fn(p)), 6)] for p in np.linspace(0, 1, n)]


def piecewise(control_points):
    return {"type": "piecewise", "control_points": control_points}


def from_x(x):
    """4-vector -> full 5-point control-point list with T pinned at 0 up front."""
    return piecewise([[FRACS[0], T_FIRST]] +
                     [[FRACS[i + 1], round(float(v), 6)] for i, v in enumerate(x)])


# --------------------------------------------------------------- phase zero --

def phase0():
    """Six validation runs. The first four are the batch the plan asked for; the
    two cosine arms are added because they are the only ones that actually test
    discretisation fidelity (see the module docstring)."""
    warm_cont = {"type": "warmup_linear", "t_start": 0.0, "t_end": 0.95, "warmup_frac": 0.25}
    return [
        # Control: reproduces static T=0.50, whose 3-seed mean (0.035229, sd
        # 0.00037) is the tightest baseline we have. Runs through the NEW
        # piecewise code path, so it is a validation of that path too.
        ("p0_static_T050", piecewise([[f, 0.5] for f in FRACS]),
         "static T=0.50 via piecewise; expect ~0.0352 (n=3)"),
        # Control: the incumbent best, on the OLD code path, on this hardware.
        ("p0_warmup_cont", warm_cont,
         "continuous warmup 0->0.95; expect ~0.0296 (n=3)"),
        # Same function as p0_warmup_cont, exactly. Difference = seed noise.
        ("p0_warmup_pw5", piecewise(
            [[0.0, 0.0], [0.25, 0.0], [0.5, round(0.95 / 3, 6)],
             [0.75, round(0.95 * 2 / 3, 6)], [1.0, 0.95]]),
         "5-point warmup; EXACTLY equals p0_warmup_cont -- measures seed noise"),
        # 10 knots at k/9 miss the kink at 0.25; max |dT| = 0.026.
        ("p0_warmup_pw10", piecewise(sample_at(warmup_T, 10)),
         "10-point warmup; NOT exact (knots miss the p=0.25 kink), max |dT|=0.026"),
        # The real discretisation test: cosine has curvature.
        ("p0_cosine_pw5", piecewise(sample_at(cosine_T, 5)),
         "5-point cosine vs known continuous 0.032169 (n=3); max |dT|=0.033"),
        ("p0_cosine_pw10", piecewise(sample_at(cosine_T, 10)),
         "10-point cosine; max |dT|=0.007 -- brackets the 5-point error"),
    ]


# ---------------------------------------------------------------- phase one --

def anchors():
    """Hand-crafted schedules re-expressed as 4D points. Re-measured in THIS
    batch rather than borrowed from the old lysine JSONs: the GP needs a
    training set collected on one code path and one machine, and mixing eras is
    exactly the cross-batch comparison that has burned this project before."""
    out = []
    for name, fn, exact in [
        ("a_linear_0to095",   linear_T,   True),   # no kinks -> exact at 5 points
        ("a_twophase_050",    twophase_T, True),   # kink at p=0.5, which is a knot
        ("a_warmup15_0to095", lambda p: warmup_T(p, frac=0.15), False),
        ("a_warmup40_0to095", lambda p: warmup_T(p, frac=0.40), False),
    ]:
        x = [fn(p) for p in FRACS[1:]]
        out.append((name, x, ("exact 5-point form" if exact else "5-point approximation")
                    + " of a known hand-crafted schedule"))
    return out


def structured():
    """Deliberate coverage of regions that uniform-over-the-simplex sampling
    underweights. Sorting a uniform sample makes T2 ~ Beta(1,4) (mean 0.20) and
    T5 ~ Beta(4,1) (mean 0.80), so early ramps and low ceilings are rare -- and
    the plan explicitly asks for both."""
    return [
        ("s_early_steep",   [0.60, 0.70, 0.80, 0.90], "ramps early and hard"),
        ("s_early_mid",     [0.45, 0.60, 0.75, 0.95], "ramps early, high ceiling"),
        ("s_late_hard",     [0.00, 0.00, 0.40, 0.95], "holds T=0 for half, then jumps"),
        ("s_late_extreme",  [0.00, 0.05, 0.15, 1.00], "almost all data until the very end"),
        ("s_plateau_low",   [0.30, 0.50, 0.50, 0.50], "rise then hold at the static optimum"),
        ("s_plateau_mid",   [0.20, 0.60, 0.60, 0.70], "rise then hold above it"),
        ("s_ceiling_050",   [0.15, 0.30, 0.45, 0.50], "gentle ramp, low ceiling"),
        ("s_ceiling_100",   [0.25, 0.50, 0.75, 1.00], "linear ramp all the way to T=1"),
    ]


def sobol(n):
    """Uniform over the monotone simplex: draw in the unit cube, then sort. The
    sort is the constraint -- every sorted draw satisfies T2<=T3<=T4<=T5 by
    construction, so nothing is rejected and the coverage stays low-discrepancy."""
    # scipy warns that Sobol balance properties want n to be a power of 2. The
    # first n points of a scrambled Sobol sequence are still low-discrepancy --
    # we lose the exact equidistribution guarantee, not the coverage -- and the
    # GPU budget, not a power of two, is what sets n here.
    import warnings
    eng = qmc.Sobol(d=4, scramble=True, seed=SOBOL_SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*power of 2.*")
        pts = np.sort(eng.random(n), axis=1)
    return [(f"q_sobol{i:02d}", [round(float(v), 4) for v in row],
             "Sobol point, sorted to enforce monotonicity")
            for i, row in enumerate(pts)]


def phase1():
    runs, seen = [], set()
    for name, x, note in anchors() + structured() + sobol(12):
        x = [round(float(v), 6) for v in x]
        key = tuple(round(v, 3) for v in x)
        if key in seen:                      # never pay twice for the same point
            print(f"  (dropped duplicate {name}: {key})")
            continue
        seen.add(key)
        assert all(b >= a - 1e-9 for a, b in zip(x, x[1:])), f"{name} not monotone: {x}"
        assert all(0.0 <= v <= 1.0 for v in x), f"{name} out of range: {x}"
        runs.append((f"p1_{name}", from_x(x), note))
    return runs


# --------------------------------------------------------------------- main --

def as_x(spec):
    """The 4-vector a spec corresponds to, for the BO training set. Continuous
    schedules are projected onto the 5 control fractions; that projection is
    lossy for warmup_cont, which is why its 4D twin is run separately."""
    if spec["type"] != "piecewise":
        return None
    cp = spec["control_points"]
    fr = [p[0] for p in cp]
    tv = [p[1] for p in cp]
    return [round(float(np.interp(f, fr, tv)), 6) for f in FRACS[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print only, do not write")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.environ.get("AMBIENT_BASE") or next(
        (p for p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
         if os.path.isdir(p)), None)
    out = args.out or (os.path.join(base, "generated", "dyn_search_manifest.json")
                       if base else "dyn_search_manifest.json")

    print("=== Phase 0: validation ===")
    entries = []
    for name, spec, note in phase0():
        entries.append({"name": name, "phase": 0, "schedule": spec,
                        "x": as_x(spec), "note": note})
        print(f"  {name:<20} {note}")

    print("\n=== Phase 1: initial batch ===")
    for name, spec, note in phase1():
        x = as_x(spec)
        entries.append({"name": name, "phase": 1, "schedule": spec, "x": x, "note": note})
        print(f"  {name:<22} {str([f'{v:.3f}' for v in x]):<40} {note}")

    manifest = {
        "dataset": "celeba_dynamic_t_v2",
        "fractions": FRACS,
        "t_first_pinned": T_FIRST,
        "sobol_seed": SOBOL_SEED,
        "known_seed_averaged_reference": KNOWN,
        "runs": entries,
    }
    print(f"\nTotal: {len(entries)} runs "
          f"({sum(e['phase'] == 0 for e in entries)} validation, "
          f"{sum(e['phase'] == 1 for e in entries)} search)")
    if args.show:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
