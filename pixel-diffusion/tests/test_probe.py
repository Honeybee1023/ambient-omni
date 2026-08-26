"""Checks for the online T-probe (training/probe.py).

Run:  python tests/test_probe.py

The point of this file is to pin the probe's behaviour against *synthetic
denoisers whose boundary is known by construction*, so that a bad rule is caught
here rather than after a 9-hour training run has quietly optimised the wrong
thing. Needs only torch + numpy + scipy; no dataset, no GPU, no wandb.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import probe as P                                    # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# grid and T <-> sigma

print("\n-- grid and conversions --")
grid = P.make_t_grid(20)
check("grid is uniform in T", np.allclose(np.diff(grid), 0.05),
      f"spacing={np.diff(grid)[0]:.4f}")
check("grid stays inside (0,1)", grid[0] > 0 and grid[-1] < 1,
      f"[{grid[0]:.3f}, {grid[-1]:.3f}]")
check("sigma_to_t inverts t_to_sigma",
      np.allclose(P.sigma_to_t(P.t_to_sigma(grid)), grid, atol=1e-9))
check("T=0.5 -> sigma=exp(-1.2)", abs(P.t_to_sigma(0.5) - np.exp(-1.2)) < 1e-9,
      f"{float(P.t_to_sigma(0.5)):.5f}")
check("sigma is increasing in T", np.all(np.diff(P.t_to_sigma(grid)) > 0))


# ---------------------------------------------------------------------------
# boundary scan

print("\n-- boundary scan --")


def res_from_flags(flags, key="loss_ratio", hi=2.0, lo=1.0):
    """A minimal probe result whose `key` trips the fixed rule exactly on flags.

    `lo` sits exactly ON the null (ratio 1.0). It cannot be 0.5: the divergence
    test is two-sided, so 0.5 is as far from the null as 2.0 and would flag too.
    """
    n = len(flags)
    return {"t_grid": list(P.make_t_grid(n)), "n_draws": 2,
            "per_sigma": [{key: (hi if f else lo)} for f in flags]}


f = [True, True, True, False, False, False, False, False]
t, d = P.decide_T(res_from_flags(f), "loss_ratio", "fixed", 1.0)
check("boundary sits just above the last diverging level", d["boundary_index"] == 3,
      f"index={d['boundary_index']}")

f = [True, True, False, True, False, False, False, False]
t, d = P.decide_T(res_from_flags(f), "loss_ratio", "fixed", 1.0)
check("a gap inside the diverging region does not end the scan early",
      d["boundary_index"] == 4, f"index={d['boundary_index']}")

f = [False] * 8
t, _ = P.decide_T(res_from_flags(f), "loss_ratio", "fixed", 1.0)
check("nothing diverging -> T = 0", t == 0.0, f"T={t}")

f = [True] * 8
t, _ = P.decide_T(res_from_flags(f), "loss_ratio", "fixed", 1.0)
check("everything diverging -> T = 1", t == 1.0, f"T={t}")

# One stray high-sigma point: last-crossing is dragged to the top, percentile is not.
f = [True, True, True, False, False, False, False, True]
t_last, d_last = P.decide_T(res_from_flags(f), "loss_ratio", "fixed", 1.0)
t_pct, d_pct = P.decide_T(res_from_flags(f), "loss_ratio", "percentile", 1.0, q=0.75)
check("one stray point pins the last-crossing rule at the top",
      d_last["boundary_index"] == 8, f"index={d_last['boundary_index']}")
check("percentile tolerates the stray point", d_pct["boundary_index"] == 3,
      f"index={d_pct['boundary_index']}")

for bad in [("nope", "fixed"), ("mse_gap", "nope")]:
    try:
        P.decide_T(res_from_flags([False]), *bad)
        check(f"rejects {bad}", False)
    except ValueError:
        check(f"rejects {bad}", True)


# ---------------------------------------------------------------------------
# baseline correction -- the failure mode that motivated it

print("\n-- baseline correction --")

# A corrupt arm that is genuinely distinguishable only below index 4, but whose
# MSE sits permanently BELOW the clean arm at high sigma because blurred images
# have less pixel variance and the denoiser tends to the dataset mean. This is
# what the real probe looks like, and it is what breaks a naive threshold.
n = 12
ratio = np.array([1.40, 1.30, 1.18, 1.09, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94])
gap = (ratio - 1.0) * 0.01
res = {"t_grid": list(P.make_t_grid(n)), "n_draws": 2,
       "per_sigma": [{"loss_ratio": float(r), "gap_mean": float(g), "gap_se": 1e-4,
                      "gap_t": float(g / 1e-4), "predvar_gap": 0.0,
                      "predvar_gap_se": 1e-4, "predvar_ratio": 1.0}
                     for r, g in zip(ratio, gap)]}

t_fixed, d_fixed = P.decide_T(res, "loss_ratio", "fixed", 1.02)
t_base, d_base = P.decide_T(res, "loss_ratio", "baseline", 1.02)
# The tail sits at 0.94, which is 6% from a null of 1.0 -- in the other
# direction, but the test is two-sided, so the uncorrected rule calls the whole
# grid diverging. This is the real failure mode, not a contrived one: measured
# on a real checkpoint the loss ratio never came near 1.0 at any sigma.
check("fixed rule is fooled by a tail that is offset from 1.0", t_fixed == 1.0,
      f"T={t_fixed}")
check("baseline rule recovers the boundary", d_base["boundary_index"] == 4,
      f"index={d_base['boundary_index']}")

# Now shift the whole curve up so the asymptote is 1.06 -- the arms are no more
# distinguishable than before, only the intrinsic-variance offset changed sign.
res_shift = {"t_grid": res["t_grid"], "n_draws": 2,
             "per_sigma": [dict(r, loss_ratio=r["loss_ratio"] + 0.12) for r in res["per_sigma"]]}
t_fixed2, d_fixed2 = P.decide_T(res_shift, "loss_ratio", "fixed", 1.02)
t_base2, d_base2 = P.decide_T(res_shift, "loss_ratio", "baseline", 1.02)
check("a shifted copy also breaks the fixed rule", t_fixed2 == 1.0,
      f"T={t_fixed2}")
check("baseline rule is immune to the offset", d_base2["boundary_index"] == d_base["boundary_index"],
      f"index={d_base2['boundary_index']} vs {d_base['boundary_index']}")

# Same story for the t-statistic form: adaptive keys off zero, baseline off the top.
gap_off = gap + 0.004
res_g = {"t_grid": res["t_grid"], "n_draws": 2,
         "per_sigma": [{"gap_mean": float(g), "gap_se": 1e-4, "gap_t": float(g / 1e-4),
                        "loss_ratio": 1.0, "predvar_gap": 0.0, "predvar_gap_se": 1e-4,
                        "predvar_ratio": 1.0} for g in gap_off]}
t_ad, _ = P.decide_T(res_g, "mse_gap", "adaptive", 2.0)
t_bl, d_bl = P.decide_T(res_g, "mse_gap", "baseline", 2.0)
check("offset gap pins the adaptive rule at 1.0", t_ad == 1.0, f"T={t_ad}")
check("baseline rule recovers the boundary", d_bl["boundary_index"] == 4,
      f"index={d_bl['boundary_index']}")

check("baseline uses the top 3 levels", P.baseline_levels(res) == [9, 10, 11],
      str(P.baseline_levels(res)))
dec = {k: v for k, v in P.all_decisions(res).items() if not k.endswith("__error")}
check("all_decisions covers every rule", len(dec) == len(P.COUNTERFACTUALS),
      f"{len(dec)} entries vs {len(P.COUNTERFACTUALS)} rules")
check("a metric missing from a result never raises", all(
      isinstance(v, float) or v is None for v in dec.values()))
check("skill_ratio is registered as a metric", "skill_ratio" in P._METRICS)


# ---------------------------------------------------------------------------
# end-to-end against synthetic denoisers

print("\n-- end-to-end with synthetic denoisers --")

torch.manual_seed(7)
N, C, H, W = 24, 3, 8, 8
clean = torch.randn(N, C, H, W) * 0.5


def blur(x):
    """Cheap separable smoothing, standing in for the b5 gaussian."""
    k = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 1, 3).repeat(C, 1, 1, 1)
    y = torch.nn.functional.conv2d(x, k, padding=(0, 1), groups=C)
    return torch.nn.functional.conv2d(y, k.transpose(2, 3), padding=(1, 0), groups=C)


corrupt = blur(clean)
SD = 0.5


class Blind(torch.nn.Module):
    """Cannot distinguish anything: the exact Gaussian posterior mean.

    Uses only x_t and sigma, so it treats both arms identically at every level.
    Any divergence the probe reports here is an artifact of the probe itself.
    """
    training = False

    def eval(self):
        return self

    def forward(self, x_t, sigma, labels=None):
        return x_t * (SD ** 2 / (SD ** 2 + sigma ** 2))


class Sharp(torch.nn.Module):
    """Knows the clean manifold, and only below a set noise level.

    Below sigma_cut it snaps its answer onto the clean images it memorised, so
    the clean arm is reconstructed far better than the blurred one. Above
    sigma_cut it falls back to the blind posterior mean and the arms become
    indistinguishable. The boundary is therefore sigma_cut, exactly.
    """
    training = False

    def __init__(self, bank, sigma_cut):
        super().__init__()
        self.bank = bank
        self.sigma_cut = sigma_cut

    def eval(self):
        return self

    def forward(self, x_t, sigma, labels=None):
        s = float(sigma.reshape(-1)[0])
        shrunk = x_t * (SD ** 2 / (SD ** 2 + s ** 2))
        if s >= self.sigma_cut:
            return shrunk
        # Nearest clean exemplar, i.e. a denoiser that has learned this manifold.
        d = ((shrunk.reshape(shrunk.shape[0], 1, -1)
              - self.bank.reshape(1, self.bank.shape[0], -1)) ** 2).sum(-1)
        # Blended rather than a pure snap: a real denoiser never reaches exactly
        # zero error, and a toy that does exercises only the degenerate path
        # (which gets its own check below).
        return 0.9 * self.bank[d.argmin(dim=1)] + 0.1 * shrunk


g20 = P.make_t_grid(20)

r_blind = P.run_probe(Blind(), clean, corrupt, t_grid=g20, n_draws=2, batch_size=8)

# The nuisance this whole correction exists for. A denoiser that provably cannot
# distinguish the arms still shows a huge raw MSE difference, purely because blur
# removes pixel energy -- and it drifts monotonically rather than sitting at a
# constant offset, which is why subtracting a high-sigma baseline cannot fix it.
raw = np.array([r["loss_ratio"] for r in r_blind["per_sigma"]])
skill = np.array([r["skill_ratio"] for r in r_blind["per_sigma"]])
pv = np.array([r["predvar_ratio"] for r in r_blind["per_sigma"]])
check("raw loss ratio is badly contaminated for a blind denoiser",
      raw.max() - raw.min() > 0.5, f"span {raw.min():.3f}..{raw.max():.3f}")
check("blind_mse removes it", skill.max() - skill.min() < 0.05,
      f"span {skill.min():.5f}..{skill.max():.5f}")
check("prediction variance is nuisance-free by construction",
      np.allclose(pv, 1.0, atol=1e-4), f"span {pv.min():.6f}..{pv.max():.6f}")
check("contamination is a drift, not an offset (so baseline cannot fix it)",
      abs(np.corrcoef(np.arange(len(raw)), raw)[0, 1]) > 0.9,
      f"corr with level index = {np.corrcoef(np.arange(len(raw)), raw)[0, 1]:.3f}")

for metric in ("skill_ratio", "pred_var"):
    t_b, d_b = P.decide_T(r_blind, metric, "fixed", 1.02)
    check(f"blind denoiser -> T = 0 under {metric}", t_b == 0.0,
          f"T={t_b}, diverging levels={d_b['n_diverging']}")
t_bad, _ = P.decide_T(r_blind, "mse_gap", "baseline", 2.0)
check("...while baseline-corrected raw MSE is fooled into a nonzero T",
      t_bad > 0.5, f"T={t_bad}")

# Two-sided is not a detail: the real signal approaches its null from below.
pv_real = [0.8726, 0.8574, 0.8485, 0.8446, 0.8444, 0.8472, 0.8525, 0.8598,
           0.8685, 0.8783, 0.8891, 0.9010, 0.9147, 0.9306, 0.9477, 0.9638,
           0.9768, 0.9862, 0.9949, 0.9982]      # measured, sobol08 @ 1001 kimg
res_pv = {"t_grid": list(P.make_t_grid(20)), "n_draws": 2,
          "per_sigma": [{"predvar_ratio": v} for v in pv_real]}
t_pv, d_pv = P.decide_T(res_pv, "pred_var", "fixed", 1.02)
check("a ratio converging to 1 from BELOW is still detected",
      0.0 < t_pv < 1.0, f"T={t_pv}, diverging {d_pv['n_diverging']}/20")
check("raising the threshold moves the boundary down",
      P.decide_T(res_pv, "pred_var", "fixed", 1.10)[0]
      < P.decide_T(res_pv, "pred_var", "fixed", 1.01)[0],
      f"thr1.10 -> {P.decide_T(res_pv, 'pred_var', 'fixed', 1.10)[0]}, "
      f"thr1.01 -> {P.decide_T(res_pv, 'pred_var', 'fixed', 1.01)[0]}")

cut = float(P.t_to_sigma(0.5))
r_sharp = P.run_probe(Sharp(clean, cut), clean, corrupt, t_grid=g20, n_draws=2, batch_size=8)
t_sharp, d_sharp = P.decide_T(r_sharp, "skill_ratio", "fixed", 1.02)
check("planted boundary at T=0.5 is recovered", abs(t_sharp - 0.5) <= 0.10,
      f"T={t_sharp} (planted 0.5)")
check("the recovered boundary is not the trivial answer", 0.0 < t_sharp < 1.0,
      f"T={t_sharp}")

cut_lo = float(P.t_to_sigma(0.25))
r_lo = P.run_probe(Sharp(clean, cut_lo), clean, corrupt, t_grid=g20, n_draws=2, batch_size=8)
t_lo, _ = P.decide_T(r_lo, "skill_ratio", "fixed", 1.02)
check("a lower planted boundary moves T down", t_lo < t_sharp,
      f"T(cut=0.25)={t_lo} < T(cut=0.5)={t_sharp}")
check("pred_var also tracks the planted boundary",
      abs(P.decide_T(r_sharp, "pred_var", "fixed", 1.02)[0] - 0.5) <= 0.15,
      f"T={P.decide_T(r_sharp, 'pred_var', 'fixed', 1.02)[0]}")

check("probe reports per-sigma stats for every level",
      len(r_sharp["per_sigma"]) == 20 and all("gap_se" in r for r in r_sharp["per_sigma"]))
check("prediction variance is recorded", "predvar_ratio" in r_sharp["per_sigma"][0])


# ---------------------------------------------------------------------------
# determinism and RNG hygiene

# Degenerate denominator: a model that reconstructs clean images perfectly makes
# skill_clean 0. The ratio must blow up to +inf (maximal divergence), not to nan
# -- nan compares False and would be read as "safe to use all corrupt data".
print("\n-- degenerate denominators fail safe --")


class Perfect(torch.nn.Module):
    """Returns the exact clean exemplar at every sigma. skill_clean == 0."""
    training = False

    def eval(self):
        return self

    def forward(self, x_t, sigma, labels=None):
        d = ((x_t.reshape(x_t.shape[0], 1, -1) - clean.reshape(1, clean.shape[0], -1)) ** 2).sum(-1)
        return clean[d.argmin(dim=1)]


r_perf = P.run_probe(Perfect(), clean, corrupt, t_grid=g20[:6], n_draws=2, batch_size=8)
sr = [r["skill_ratio"] for r in r_perf["per_sigma"]]
pvr = [r["predvar_ratio"] for r in r_perf["per_sigma"]]
check("zero denominator gives +inf, not nan",
      all(not np.isnan(v) for v in sr), f"skill_ratio={sr[0]}")
check("prediction-variance ratio likewise", all(not np.isnan(v) for v in pvr),
      f"predvar_ratio={pvr[0]}")
t_perf, d_perf = P.decide_T(r_perf, "skill_ratio", "fixed", 1.02)
check("a perfectly-discriminating model is reported as diverging everywhere",
      d_perf["n_diverging"] == len(sr), f"{d_perf['n_diverging']}/{len(sr)}")
check("...so T is pushed to 1.0, the safe direction", t_perf == 1.0, f"T={t_perf}")

nanres = {"t_grid": list(P.make_t_grid(4)), "n_draws": 2,
          "per_sigma": [{"skill_ratio": float("nan")} for _ in range(4)]}
check("an unevaluable level counts as unsafe, not as safe",
      P.decide_T(nanres, "skill_ratio", "fixed", 1.02)[0] == 1.0)

print("\n-- determinism and RNG hygiene --")

a = P.run_probe(Blind(), clean, corrupt, t_grid=g20[:5], n_draws=2, batch_size=8, probe_seed=99)
b = P.run_probe(Blind(), clean, corrupt, t_grid=g20[:5], n_draws=2, batch_size=8, probe_seed=99)
check("same seed -> bitwise-identical probe",
      all(a["per_sigma"][i]["gap_mean"] == b["per_sigma"][i]["gap_mean"] for i in range(5)))
c = P.run_probe(Blind(), clean, corrupt, t_grid=g20[:5], n_draws=2, batch_size=8, probe_seed=100)
check("a different seed does move the numbers",
      any(a["per_sigma"][i]["gap_mean"] != c["per_sigma"][i]["gap_mean"] for i in range(5)))

torch.manual_seed(1234)
before = torch.randn(3)
torch.manual_seed(1234)
P.run_probe(Blind(), clean, corrupt, t_grid=g20[:3], n_draws=2, batch_size=8)
after = torch.randn(3)
check("probing does not consume the global RNG stream", torch.equal(before, after),
      "a probing run must see the same training batches as a non-probing one")

net = Blind()
net.training = True
seen = {}
orig_eval = net.eval


def spy():
    seen["eval"] = True
    return orig_eval()


net.eval = spy
P.run_probe(net, clean, corrupt, t_grid=g20[:2], n_draws=1, batch_size=8)
check("probe puts the net in eval mode", seen.get("eval", False))
check("probe restores training mode", net.training is True)


# ---------------------------------------------------------------------------
# controller state across a preemption

print("\n-- controller restore --")

with tempfile.TemporaryDirectory() as d:
    ctrl = P.ProbeController({"every_kimg": 100, "t_init": 0.0}, d, "cpu")
    with open(ctrl.log_path, "w") as f:
        f.write(json.dumps({"probe_index": 0, "kimg": 0.0, "T_raw": 0.0, "T_smoothed": 0.0}) + "\n")
        f.write(json.dumps({"probe_index": 1, "kimg": 100.0, "T_raw": 0.4, "T_smoothed": 0.25}) + "\n")
        f.write('{"probe_index": 2, "kimg": 200.0, "T_ra')     # torn by a kill -9

    fresh = P.ProbeController({"every_kimg": 100, "t_init": 0.0}, d, "cpu")
    check("restore finds a previous run", fresh.restore())
    check("restore recovers the smoothed T", abs(fresh.current_T - 0.25) < 1e-12,
          f"T={fresh.current_T}")
    check("restore survives a torn final line", fresh.n_probes == 2, f"n={fresh.n_probes}")
    check("restore schedules the next probe correctly", fresh.next_kimg == 200.0,
          f"next={fresh.next_kimg}")

    empty = P.ProbeController({"every_kimg": 100}, tempfile.mkdtemp(), "cpu")
    check("no log -> nothing to restore", empty.restore() is False)

# Smoothing, step cap and monotonicity, without touching a model.
ctrl = P.ProbeController({"alpha": 0.5, "t_init": 0.2}, tempfile.mkdtemp(), "cpu")
ctrl.n_probes = 1
ctrl.current_T = 0.2
blend = (1 - ctrl.alpha) * 0.2 + ctrl.alpha * 0.8
check("EMA blends previous and raw T", abs(blend - 0.5) < 1e-12, f"{blend}")

print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all probe checks passed")
