"""Which part of training does the T schedule actually matter in?

Reproduces the analysis behind section 7 of PRINCIPLED_T_SEARCH.md from the
committed data, so the conclusions can be rechecked (or overturned) without any
GPU time.

The question this answers is not "which schedule is best" -- the discrete search
did that -- but "when does T matter". It turned out to be the second quarter of
training, and getting there required first being wrong about the first quarter:
`pr_predvar_hold25` held T=0 through [0, 0.25] and gained nothing, which is what
killed that hypothesis. Both windows are reported so the comparison is visible
rather than asserted.

Reads:
  results/principled_runs.json      MIND per run + how its T curve is defined
  results/probe_logs/<run>.jsonl.gz the probe log, for runs whose T was driven
                                    or smoothed and so is not a closed form

Usage:
    python analyze_schedule_effect.py
    python analyze_schedule_effect.py --windows      # full window scan
"""

import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "principled_runs.json")
LOGS = os.path.join(HERE, "results", "probe_logs")

# Run-to-run spread of MIND for one configuration, measured over 8 replicate
# pairs in the discrete study. Everything here is quoted against it.
NOISE_SD = 0.00089

GRID = 2001          # samples of the [0,1] training axis


def curve_from_knots(knots):
    p = np.linspace(0, 1, GRID)
    return np.interp(p, [k[0] for k in knots], [k[1] for k in knots])


def curve_from_log(name, total_kimg=2000):
    """Applied T over training, from a probe log.

    T is held constant between probes, and `T_applied` is null while a
    `hold_until` is in force -- during the hold the run trains at t_init, which
    is 0. Treating null as 0 is therefore correct, not a fill-in.
    """
    path = os.path.join(LOGS, f"{name}.jsonl.gz")
    if not os.path.exists(path):
        raise SystemExit(f"missing probe log: {path}")
    k, t = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # torn final line from a kill
            k.append(r["kimg"])
            t.append(r["T_applied"] if r.get("T_applied") is not None else 0.0)
    if not k:
        raise SystemExit(f"no usable records in {path}")
    return np.interp(np.linspace(0, total_kimg, GRID), k, t, left=t[0], right=t[-1])


def load():
    with open(RUNS) as f:
        spec = json.load(f)
    out = []
    for r in spec["runs"]:
        if r.get("mind") is None:
            continue
        if "knots" in r:
            c = curve_from_knots(r["knots"])
        elif "constant" in r:
            c = np.full(GRID, float(r["constant"]))
        else:
            c = curve_from_log(r["log"])
        out.append((r["name"], c, float(r["mind"]), r.get("kind", "")))
    return out, spec


def window_mean(c, lo, hi):
    return float(c[int(lo * (GRID - 1)):int(hi * (GRID - 1)) + 1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", action="store_true", help="full window scan")
    args = ap.parse_args()

    rows, spec = load()
    M = np.array([m for _, _, m, _ in rows])
    print("=" * 76)
    print(f"{len(rows)} schedules with a measured MIND   (noise sd {NOISE_SD})")
    print("=" * 76)

    print(f"\n{'schedule':<22} {'kind':<12} {'T[0,.25]':>9} {'T[.25,.5]':>10} "
          f"{'T[.75,1]':>9} {'MIND':>9}")
    for n, c, m, kind in sorted(rows, key=lambda z: z[2]):
        print(f"{n:<22} {kind:<12} {window_mean(c,0,.25):>9.3f} "
              f"{window_mean(c,.25,.5):>10.3f} {window_mean(c,.75,1):>9.3f} {m:>9.6f}")

    wins = [(0, .10), (0, .25), (0, .50), (.10, .40), (.25, .50),
            (.25, .75), (.50, .75), (.50, 1.0), (.75, 1.0), (0, 1.0)]
    if not args.windows:
        wins = [(0, .25), (.25, .50), (.75, 1.0)]
    print(f"\n{'window of training':<22} {'r with MIND':>12}")
    best = (0.0, None)
    for lo, hi in wins:
        x = np.array([window_mean(c, lo, hi) for _, c, _, _ in rows])
        r = float(np.corrcoef(x, M)[0, 1])
        print(f"  [{lo:.2f}, {hi:.2f}]{'':<10} {r:>+12.3f}")
        if abs(r) > best[0]:
            best = (abs(r), (lo, hi, r))
    lo, hi, r = best[1]
    print(f"\nstrongest: [{lo:.2f}, {hi:.2f}]  r = {r:+.3f}")

    mid = np.array([window_mean(c, .25, .50) for _, c, _, _ in rows])
    late = np.array([window_mean(c, .75, 1.0) for _, c, _, _ in rows])
    X = np.column_stack([np.ones(len(M)), mid, late])
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    pred = X @ beta
    r2 = 1 - ((M - pred) ** 2).sum() / ((M - M.mean()) ** 2).sum()
    r2_mid = float(np.corrcoef(mid, M)[0, 1]) ** 2
    print(f"\nMIND ~ 1 + T[.25,.50] + T[.75,1]")
    print(f"  T[.25,.50]  {beta[1]:+.5f}   {beta[1]*0.1/NOISE_SD:+.1f} sd per +0.1 of T")
    print(f"  T[.75,1]    {beta[2]:+.5f}   {beta[2]*0.1/NOISE_SD:+.1f} sd per +0.1 of T")
    print(f"  R^2 {r2:.3f}   (second quarter alone: {r2_mid:.3f} -- the late term adds nothing)")

    print(f"\n{'schedule':<22} {'MIND':>9} {'fit':>9} {'residual':>10}")
    for (n, _, m, _), p in sorted(zip(rows, pred), key=lambda z: z[0][2]):
        flag = "  <-- outlier" if abs(m - p) / NOISE_SD > 3 else ""
        print(f"{n:<22} {m:>9.6f} {p:>9.6f} {(m-p)/NOISE_SD:>+9.2f} sd{flag}")

    print("\nRead the fit as 'T in the second quarter dominates', not as a calibrated")
    print("predictor. The best schedule is the argmin of a 30-run search, so it is the")
    print("point most likely to be an optimistic draw; its residual is not evidence of")
    print("extra structure. See PRINCIPLED_T_SEARCH.md section 7.")


if __name__ == "__main__":
    main()
