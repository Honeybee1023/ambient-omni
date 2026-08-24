#!/usr/bin/env python3
"""Bayesian optimisation over 4D dynamic-T schedules -- proposes the next batch.

The design point x = [T2, T3, T4, T5] gives T at training fractions
[.25, .5, .75, 1]; T at fraction 0 is pinned to 0. Objective is MIND (lower
better). Monotonicity 0 <= T2 <= T3 <= T4 <= T5 <= 1 is enforced by construction
-- every candidate is generated already sorted, so nothing is ever rejected or
repaired after the fact.

GP: Matern-5/2 with ARD lengthscales, constant mean, Gaussian noise. Fitted by
maximising the log marginal likelihood (L-BFGS-B, multi-start) on standardised y.
Written against numpy/scipy rather than GPyTorch because GPyTorch is only
installed on CSAIL and the runs live on proline/lysine; at n ~ 30 in 4D an exact
GP is a few milliseconds either way.

THE NOISE FLOOR IS NOT OPTIONAL
-------------------------------
Repeat runs of one schedule differ by ~0.0009 MIND (pooled over the 5
hand-crafted schedules with 3 seeds each). Free-fitting the noise on ~30 points
routinely drives it toward zero, and a GP that believes its observations are
exact will interpolate the noise and send the whole batch chasing a lucky run --
which is how a 1-seed argmin was twice mistaken for a real optimum earlier in
this project. So the fitted noise is clamped below at the measured replicate SD.

Usage:
    python bo_suggest_4d.py                 # propose 4 points
    python bo_suggest_4d.py --batch 8
    python bo_suggest_4d.py --emit-manifest  # append proposals to the manifest
"""

import argparse, json, os

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, qmc

AMBIENT_BASE = os.environ.get("AMBIENT_BASE") or next(
    (p for p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if os.path.isdir(p)), "/data/scratch/honjar")
GENERATED = f"{AMBIENT_BASE}/generated"
MANIFEST = f"{GENERATED}/dyn_search_manifest.json"
RESULTS = f"{GENERATED}/dyn_all_results.json"

# Pooled within-schedule SD across the 5 hand-crafted schedules that have 3
# seeds each (static_T050, warmup, linear, cosine, twophase). See
# collect_dyn_results.py, which recomputes this from the batch's own replicates.
NOISE_FLOOR = 0.00089
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ------------------------------------------------------------------ kernel --

def matern52(A, B, ls, sf2):
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) / ls) ** 2, 0).sum(-1) + 1e-12)
    s = np.sqrt(5.0) * d
    return sf2 * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def nll(theta, X, y, noise_floor_std):
    """Negative log marginal likelihood. theta = log([ls(4), sf, sn])."""
    ls = np.exp(theta[:4]); sf2 = np.exp(2 * theta[4])
    sn2 = np.exp(2 * theta[5]) + noise_floor_std ** 2      # clamp, see docstring
    K = matern52(X, X, ls, sf2) + sn2 * np.eye(len(X))
    try:
        c = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    a = cho_solve(c, y)
    return float(0.5 * y @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(X) * np.log(2 * np.pi))


def fit_gp(X, y, noise_floor_std, restarts=12, seed=0):
    rng = np.random.default_rng(seed)
    best, best_v = None, np.inf
    for i in range(restarts):
        x0 = np.concatenate([
            np.log(rng.uniform(0.15, 1.5, 4)),       # ARD lengthscales
            [np.log(rng.uniform(0.5, 1.5))],         # signal sd
            [np.log(rng.uniform(0.05, 0.5))],        # extra noise sd above floor
        ])
        try:
            # Lengthscales capped at 2 on a unit-cube domain. Anything longer
            # makes the GP effectively linear, and a linear surrogate puts its
            # EI maximum on a corner and predicts an improvement that is pure
            # extrapolation -- on a synthetic bowl an unbounded fit proposed
            # points with predicted MIND *below* the true global minimum.
            r = minimize(nll, x0, args=(X, y, noise_floor_std), method="L-BFGS-B",
                         bounds=[(np.log(0.05), np.log(2.0))] * 4
                                + [(np.log(1e-2), np.log(3.0)), (np.log(1e-4), np.log(3))])
        except Exception:
            continue
        if r.fun < best_v:
            best, best_v = r.x, r.fun
    if best is None:
        raise RuntimeError("GP fit failed on every restart")
    return best, best_v


def posterior(theta, X, y, Xs, noise_floor_std):
    ls = np.exp(theta[:4]); sf2 = np.exp(2 * theta[4])
    sn2 = np.exp(2 * theta[5]) + noise_floor_std ** 2
    K = matern52(X, X, ls, sf2) + sn2 * np.eye(len(X))
    c = cho_factor(K, lower=True)
    Ks = matern52(X, Xs, ls, sf2)
    mu = Ks.T @ cho_solve(c, y)
    v = cho_solve(c, Ks)
    var = np.maximum(sf2 - np.einsum("ij,ij->j", Ks, v), 1e-12)
    return mu, np.sqrt(var)


def expected_improvement(mu, sd, best, xi=0.0):
    """EI for MINIMISATION. best = incumbent (lowest observed) objective."""
    imp = best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


# -------------------------------------------------------------- candidates --

def candidates(X_obs, n_sobol=8192, seed=7):
    """Monotone candidates, generated already sorted so none is ever rejected.

    Three sources, because pure Sobol-then-sort is uniform over the simplex and
    therefore rarely proposes an early ramp (sorting makes T2 ~ Beta(1,4)) --
    exactly the region the hand-crafted winners live in."""
    rng = np.random.default_rng(seed)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*power of 2.*")
        glob = np.sort(qmc.Sobol(d=4, scramble=True, seed=seed).random(n_sobol), axis=1)
    # Local jitter around every observed point: BO's exploitation needs
    # resolution near the incumbent that a global space-filling set will not have.
    local = np.sort(np.clip(
        np.repeat(X_obs, 60, axis=0) + rng.normal(0, 0.06, (len(X_obs) * 60, 4)),
        0.0, 1.0), axis=1)
    # Faces of the box: optima on a boundary (T5 = 1, or T2 = 0) are common here
    # and interior sampling reaches them slowly.
    edge = np.sort(rng.uniform(0, 1, (2000, 4)), axis=1)
    edge[:667, 0] = 0.0
    edge[667:1334, 3] = 1.0
    edge[1334:, 0] = 0.0
    edge[1334:, 3] = 1.0
    return np.unique(np.round(np.vstack([glob, local, edge]), 4), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--emit-manifest", action="store_true",
                    help="append the proposals to dyn_search_manifest.json as phase 2")
    # 0.05 in L-inf let two proposals differ only in a coordinate the GP had
    # already flagged as barely influential -- effectively buying the same
    # experiment twice at ~7 GPU-hours each.
    ap.add_argument("--min-sep", type=float, default=0.10,
                    help="minimum L-inf distance between proposals")
    args = ap.parse_args()

    res = json.load(open(args.results))
    rows = [r for r in res["runs"] if r.get("x")]
    if len(rows) < 8:
        raise SystemExit(f"Only {len(rows)} usable points; run Phase 0+1 first.")

    X = np.array([r["x"] for r in rows], float)
    y_raw = np.array([r["mind_mean"] for r in rows], float)
    ns = np.array([r["n"] for r in rows], float)

    # Standardise y; the GP is fitted in those units and the noise floor with it.
    mu_y, sd_y = y_raw.mean(), y_raw.std()
    y = (y_raw - mu_y) / sd_y
    # A run averaged over n seeds has standard error sigma/sqrt(n); using the
    # per-observation floor for all of them would overstate the noise on the
    # replicated points. Use the mean of the per-row standard errors.
    floor = float(np.mean(NOISE_FLOOR / np.sqrt(ns)) / sd_y)

    theta, v = fit_gp(X, y, floor)
    ls = np.exp(theta[:4])
    print(f"GP fitted on {len(X)} points (nll {v:.3f})")
    print(f"  ARD lengthscales [T2,T3,T4,T5]: {np.round(ls, 3)}")
    print(f"  signal sd {np.exp(theta[4]):.3f} | noise sd "
          f"{np.sqrt(np.exp(2*theta[5]) + floor**2):.3f} (floor {floor:.3f}) [standardised]")
    print(f"  -> in MIND units, noise sd = "
          f"{np.sqrt(np.exp(2*theta[5]) + floor**2) * sd_y:.6f}")
    short = np.argsort(ls)
    print(f"  most influential coordinate: T{short[0]+2} (shortest lengthscale); "
          f"least: T{short[-1]+2}")

    best_i = int(np.argmin(y))
    print(f"\nIncumbent: {rows[best_i]['name']}  x={np.round(X[best_i],3).tolist()}  "
          f"MIND={y_raw[best_i]:.6f} (n={int(ns[best_i])})")

    C = candidates(X)
    print(f"Scoring {len(C)} monotone candidates...")

    # Batch by kriging believer: take the best-EI point, insert it with its own
    # posterior mean as a fantasy observation, refit the posterior (hyperparams
    # held fixed) and repeat. Keeps the batch from collapsing onto one peak.
    Xa, ya = X.copy(), y.copy()
    picks = []
    for k in range(args.batch):
        mu, sd = posterior(theta, Xa, ya, C, floor)
        # Incumbent = best POSTERIOR MEAN at an already-evaluated point, not the
        # lowest observed value. With ~30 points at this noise level the running
        # minimum sits ~2 sd below the best true value purely by luck, and
        # anchoring EI there makes every improvement look impossible. This is the
        # same unequal-noise trap that produced two false optima in the
        # conditional sweeps, wearing an acquisition function as a disguise.
        mu_obs, _ = posterior(theta, Xa, ya, Xa, floor)
        ei = expected_improvement(mu, sd, mu_obs.min())
        if picks:  # keep proposals apart so the batch explores 4 places, not 1
            far = np.min(np.max(np.abs(C[:, None, :] - np.array(picks)[None, :, :]), axis=2), axis=1)
            ei = np.where(far >= args.min_sep, ei, -np.inf)
        j = int(np.argmax(ei))
        if not np.isfinite(ei[j]):
            print(f"  (stopped at {k} proposals: nothing left {args.min_sep} away)")
            break
        picks.append(C[j].copy())
        Xa = np.vstack([Xa, C[j]])
        ya = np.append(ya, mu[j])
        print(f"  {k+1}. x={np.round(C[j],3).tolist()}  EI={ei[j]*sd_y:.6f}  "
              f"predicted MIND={mu[j]*sd_y + mu_y:.6f} +/- {sd[j]*sd_y:.6f}")

    if args.emit_manifest and picks:
        man = json.load(open(MANIFEST))
        have = {e["name"] for e in man["runs"]}
        added = 0
        for i, x in enumerate(picks):
            name = f"p2_bo{len(have)+i:02d}"
            while name in have:
                name = f"{name}x"
            man["runs"].append({
                "name": name, "phase": 2,
                "schedule": {"type": "piecewise",
                             "control_points": [[FRACS[0], 0.0]]
                                               + [[FRACS[t+1], round(float(v), 6)]
                                                  for t, v in enumerate(x)]},
                "x": [round(float(v), 6) for v in x],
                "note": "BO proposal (EI, kriging-believer batch)"})
            have.add(name); added += 1
        json.dump(man, open(MANIFEST, "w"), indent=2)
        print(f"\nAppended {added} phase-2 runs to {MANIFEST}")
        print("Queue them with:  bash run_dyn_queue.sh init 2  (into a fresh state dir)")


if __name__ == "__main__":
    main()
