"""Compare what each probe metric/threshold rule would choose, against the
schedule we already know is good.

Reads either
  * a run's probe_log.jsonl  (online, one record per probe), or
  * a probe_checkpoint.py JSON  (offline, one record per checkpoint)

and scores every metric/rule combination on the four things that decide whether
it is usable:

  reach       does it ever move? A rule pinned at T=0 or T=1 for the whole run
              is not a schedule, it is a constant -- and a constant that looks
              like a result until you plot it.
  shape       agreement with the known-good warmup 0->0.95 curve (RMSE in T, and
              rank correlation with training progress, which is what "warmup
              shape" actually means: T rising with progress).
  stability   mean |dT| between consecutive probes. A rule that swings 0.4 per
              probe is chasing noise whatever its average looks like.
  monotone    fraction of steps that do not go backwards. Reported, never
              enforced -- see principled_t_search.py for why forcing it would
              make the headline claim circular.

Usage:
    python analyze_probe.py --log $AMBIENT_BASE/train_outputs/dyn_search/dyn_gt_warmup_s0/probe_log.jsonl
    python analyze_probe.py --calib $AMBIENT_BASE/generated/probe_calib_sobol08.json
    python analyze_probe.py --calib ... --curves      # per-sigma tables for plots
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import json
import os

import numpy as np

# warmup_linear(0 -> 0.95, warmup_frac=0.25), the discrete search's reference.
REF_KNOTS = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.31667), (0.75, 0.63333), (1.0, 0.95)]


def reference_T(progress):
    xs = [k[0] for k in REF_KNOTS]
    ys = [k[1] for k in REF_KNOTS]
    return np.interp(progress, xs, ys)


def load_records(path):
    """Both log formats reduce to (progress, kimg, counterfactual_T, probe)."""
    recs = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a torn final line. Keep the
                    # rest rather than refusing to report anything.
                    print("  (skipped one unparseable line -- run killed mid-write?)")
    else:
        with open(path) as f:
            recs = json.load(f)["records"]
    return sorted(recs, key=lambda r: r.get("kimg", 0))


def spearman(a, b):
    """Rank correlation without pulling in scipy.stats for one call."""
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return order.astype(float)
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def score(progress, traj, ref):
    traj = np.asarray(traj, dtype=float)
    ok = np.isfinite(traj)
    if ok.sum() < 2:
        return None
    p, t, r = np.asarray(progress)[ok], traj[ok], np.asarray(ref)[ok]
    steps = np.diff(t)
    return {
        "n": int(ok.sum()),
        "T_min": float(t.min()), "T_max": float(t.max()),
        "T_final": float(t[-1]),
        "span": float(t.max() - t.min()),
        "rmse_vs_ref": float(np.sqrt(np.mean((t - r) ** 2))),
        "rho_with_progress": spearman(p, t),
        "mean_abs_step": float(np.mean(np.abs(steps))) if len(steps) else 0.0,
        "frac_nondecreasing": float(np.mean(steps >= -1e-12)) if len(steps) else 1.0,
        "degenerate": bool(t.max() - t.min() < 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="probe_log.jsonl from a run")
    ap.add_argument("--calib", default=None, help="probe_checkpoint.py output")
    ap.add_argument("--total_kimg", type=int, default=2000)
    ap.add_argument("--curves", action="store_true",
                    help="also dump per-sigma tables (for the paper's figures)")
    ap.add_argument("--out", default=None, help="write the scored table as JSON")
    args = ap.parse_args()

    path = args.log or args.calib
    if not path:
        raise SystemExit("give --log or --calib")
    recs = load_records(path)
    if not recs:
        raise SystemExit(f"no probe records in {path}")

    progress = [r.get("progress", r.get("kimg", 0) / args.total_kimg) for r in recs]
    ref = [reference_T(p) for p in progress]

    print("=" * 78)
    print(f"{os.path.basename(path)}   {len(recs)} probes, "
          f"kimg {recs[0].get('kimg', 0):.0f}..{recs[-1].get('kimg', 0):.0f}")
    print("=" * 78)

    # What the run actually followed, if it was a closed-loop run.
    if any("T_applied" in r for r in recs):
        applied = [r.get("T_applied") for r in recs]
        if any(a is not None for a in applied):
            ctrl = recs[-1].get("controller", {})
            print(f"\ndriven by {ctrl.get('metric')}/{ctrl.get('rule')}/"
                  f"{ctrl.get('threshold')}  alpha={ctrl.get('alpha')} "
                  f"monotone={ctrl.get('monotone')}")
            print("  kimg      T_raw   T_applied   reference")
            for r in recs:
                print(f"  {r.get('kimg', 0):>7.0f}   {r.get('T_raw', float('nan')):>6.3f}   "
                      f"{(r.get('T_applied') if r.get('T_applied') is not None else float('nan')):>8.3f}   "
                      f"{reference_T(r.get('progress', 0)):>8.3f}")
        else:
            print("\nlogging-only run: the probe rode along, it did not steer training.")

    keys = sorted({k for r in recs for k in r.get("counterfactual_T", {})
                   if not k.endswith("__error")})
    rows = {}
    for k in keys:
        traj = [r.get("counterfactual_T", {}).get(k, float("nan")) for r in recs]
        traj = [float("nan") if v is None else v for v in traj]
        s = score(progress, traj, ref)
        if s:
            rows[k] = dict(s, trajectory=traj)

    print(f"\n{'metric/rule/threshold':<32} {'span':>6} {'rmse':>6} {'rho':>6} "
          f"{'|dT|':>6} {'mono':>5} {'Tfin':>6}  verdict")
    print("-" * 96)
    # Best = tracks the reference and does not thrash. Degenerate rules are
    # listed last rather than hidden: "this metric never moves" is a result.
    ordered = sorted(rows.items(), key=lambda kv: (kv[1]["degenerate"], kv[1]["rmse_vs_ref"]))
    for k, s in ordered:
        if s["degenerate"]:
            verdict = f"DEGENERATE (constant T={s['T_final']:.2f})"
        elif s["span"] < 0.1:
            verdict = "barely moves"
        elif s["mean_abs_step"] > 0.25:
            verdict = "unstable"
        elif s["rho_with_progress"] > 0.7 and s["rmse_vs_ref"] < 0.2:
            verdict = "warmup-shaped, close to reference"
        elif s["rho_with_progress"] > 0.7:
            verdict = "warmup-shaped, different level"
        else:
            verdict = "moves, but not warmup-shaped"
        print(f"{k:<32} {s['span']:>6.2f} {s['rmse_vs_ref']:>6.3f} "
              f"{s['rho_with_progress']:>6.2f} {s['mean_abs_step']:>6.3f} "
              f"{s['frac_nondecreasing']:>5.2f} {s['T_final']:>6.2f}  {verdict}")

    print("\nreference (warmup 0->0.95) at these probes:")
    print("  " + "  ".join(f"{r:.2f}" for r in ref))

    if args.curves:
        # Per-sigma divergence at the first, middle and last probe. This is the
        # figure: does the boundary visibly move right as training proceeds?
        picks = [0, len(recs) // 2, len(recs) - 1]
        for i in picks:
            ps = recs[i].get("probe", {}).get("per_sigma", [])
            if not ps:
                continue
            print(f"\n--- per-sigma at kimg {recs[i].get('kimg', 0):.0f} ---")
            print(f"  {'T':>6} {'sigma':>8} {'skill_cln':>10} {'skill_cor':>10} "
                  f"{'skill_r':>8} {'loss_r':>8} {'pv_r':>8}")
            for r in ps:
                print(f"  {r['t']:>6.3f} {r['sigma']:>8.4f} "
                      f"{r.get('skill_clean', float('nan')):>10.4f} "
                      f"{r.get('skill_corrupt', float('nan')):>10.4f} "
                      f"{r.get('skill_ratio', float('nan')):>8.4f} "
                      f"{r.get('loss_ratio', float('nan')):>8.4f} "
                      f"{r.get('predvar_ratio', float('nan')):>8.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"source": path, "progress": progress, "reference": ref,
                       "rules": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
