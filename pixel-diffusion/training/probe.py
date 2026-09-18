"""Online probe: find the noise level at which the model stops distinguishing
corrupted data from clean data, and read T off that boundary.

Motivation
----------
Ambient-o's premise is that a corrupted image is safe to train on at noise
levels high enough that the corruption is buried. The discrete search found the
best T(p) curve by training ~30 models and comparing MIND. This module instead
asks the model directly, during training, and costs a couple of percent of one
run.

At noise level sigma, noise a clean holdout image and a blurred holdout image
and denoise both. At small sigma the model reconstructs the clean one much
better -- blurred inputs are off the manifold it has learned. As sigma grows the
two errors converge, because the noise swamps the blur and the posterior mean is
the same for both. The sigma where they merge IS the boundary, and

    T = Phi((ln sigma + 1.2) / 1.2)

is that boundary in the units the schedule is written in. As the model sharpens
during training it can tell them apart at higher and higher sigma, so the
boundary -- and T -- should rise on its own. That is the warmup shape the
discrete search found, and recovering it without being told is the whole point.

Note the direction. Divergence lives at LOW sigma and vanishes at HIGH sigma, so
the boundary is where divergence *stops*, scanning upward. The code below reads
it as the top of the diverging region rather than the first crossing, which
survives a single noisy grid point; a first-crossing rule does not.

Design decisions that matter
----------------------------
*Common random numbers.* One noise tensor is drawn per probe from a fixed seed
and reused for every sigma, both arms, and every probe in the run. The clean and
corrupt arms therefore differ only by the blur, and the T trajectory over
training moves because the model moved -- not because the probe rolled different
dice. Without this the per-sigma curves are too noisy to threshold.

*Paired and unpaired, both.* The probe set covers the same faces in both arms
(see dataset_creation/create_probe_holdout.py), so the primary comparison is
paired: same face, same noise, blur is the only difference. That kills the
between-face variance, which is much larger than the effect. Because "compare
disjoint samples" is the more conservative design, both index ranges are also
crossed to give an unpaired estimate, and both are logged.

*Every rule, every probe, always.* Choosing T needs a metric and a threshold
rule. Rather than commit, each probe records what all of them would have picked.
A closed-loop run driven by one rule still reports the counterfactual T of every
other rule at every probe, so the comparison does not cost extra GPU-hours. It
is only a counterfactual -- the training trajectory did follow one rule -- but it
is enough to rule out degenerate metrics without retraining.

*The probe never touches the training RNG.* It runs under no_grad on its own
torch.Generator and restores the module's training mode. A probing run and a
non-probing run at the same seed see the same stream of training batches.
"""

import json
import math
import os
import time

import numpy as np
import torch

P_MEAN, P_STD = -1.2, 1.2

# EDM preconditioning constant, as used by AmbientEDMLoss and the networks here.
# It is what makes the analytic "blind" reference below exact rather than a fit:
# EDM writes D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F(...), so a network
# whose F carries no information returns exactly c_skip * x.
SIGMA_DATA = 0.5

# Below this a ratio's denominator counts as zero rather than small.
_RATIO_EPS = 1e-9

# Any T at or above this is "corrupt data is never used"; matches the clipping
# in compute_scheduled_sigma_min so the two agree at the endpoints.
T_MAX = 0.999


def t_to_sigma(t):
    """sigma = exp(1.2 * Phi^-1(T) - 1.2). Accepts scalars or arrays."""
    from scipy.stats import norm
    t = np.clip(np.asarray(t, dtype=np.float64), 0.001, T_MAX)
    return np.exp(P_STD * norm.ppf(t) + P_MEAN)


def sigma_to_t(sigma):
    from scipy.stats import norm
    sigma = np.asarray(sigma, dtype=np.float64)
    return norm.cdf((np.log(sigma) - P_MEAN) / P_STD)


def make_t_grid(n=20, lo=0.025, hi=0.975):
    """Probe levels, uniform in T rather than in sigma.

    T is the unit the schedule is written in, so a uniform T grid gives uniform
    resolution in the thing being chosen: with n=20 the returned T is quantised
    to 0.05, which is the granularity the discrete search worked at. A uniform
    grid in log-sigma would instead crowd resolution into the tails.
    """
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# forward passes


_HP_KERNEL = None


def _hp_kernel(device, dtype, sigma_px=0.5, radius=3):
    """1-D Gaussian used to split an image into the band the bucket's blur keeps
    (B x) and the band it removes (x - B x). sigma_px matches the b5 bucket."""
    global _HP_KERNEL
    key = (str(device), str(dtype))
    if _HP_KERNEL is None or _HP_KERNEL[0] != key:
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * sigma_px ** 2)); k = k / k.sum()
        _HP_KERNEL = (key, k.to(device=device, dtype=dtype))
    return _HP_KERNEL[1]


def hp_energy(x, sigma_px=0.5):
    """Per-image mean-square energy of the fine band (x - B x). B is the
    Gaussian blur of the corrupted bucket, so this is exactly the content that
    blur removes; a denoiser that has learned to output blurred targets carries
    less of it than the truth ("softness" = hp_energy(pred) / hp_energy(x))."""
    import torch.nn.functional as F
    k = _hp_kernel(x.device, x.dtype, sigma_px)
    C = x.shape[1]; r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1); ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    b = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=C)
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode="reflect"), ky, groups=C)
    return ((x - b) ** 2).mean(dim=(1, 2, 3)).double()


@torch.no_grad()
def _arm(net, imgs, noise, sigma, batch_size):
    """Denoise `imgs` at one sigma under every noise draw in `noise`.

    imgs  : [N, C, H, W]        in [-1, 1]
    noise : [N, K, C, H, W]     shared with the other arm (paired)

    Returns (mse [N], predvar [N]):
      mse     -- per-image denoising MSE against its own uncorrupted-by-noise
                 target, averaged over the K draws. For the corrupt arm the
                 target is the blurred image itself: the question is whether the
                 model can recover what it was given, not whether it can deblur.
      predvar -- per-image variance of the model's x0 prediction across the K
                 draws, averaged over pixels. How much the answer wobbles when
                 only the noise realisation changes.
    """
    N, K = noise.shape[0], noise.shape[1]
    mse = torch.zeros(N, device=imgs.device, dtype=torch.float64)
    pvar = torch.zeros(N, device=imgs.device, dtype=torch.float64)
    hp = torch.zeros(N, device=imgs.device, dtype=torch.float64)

    for i0 in range(0, N, batch_size):
        x0 = imgs[i0:i0 + batch_size]
        b = x0.shape[0]
        s = torch.full((b, 1, 1, 1), float(sigma), device=x0.device, dtype=x0.dtype)

        acc_mse = torch.zeros(b, device=x0.device, dtype=torch.float64)
        psum = torch.zeros_like(x0, dtype=torch.float64)
        psq = torch.zeros_like(x0, dtype=torch.float64)

        for k in range(K):
            x_t = x0 + s * noise[i0:i0 + b, k]
            pred = net(x_t, s, None).to(torch.float32)
            acc_mse += ((pred - x0) ** 2).mean(dim=(1, 2, 3)).double()
            hp[i0:i0 + b] += hp_energy(pred) / K
            p = pred.double()
            psum += p
            psq += p * p

        mse[i0:i0 + b] = acc_mse / K
        if K > 1:
            # Unbiased across-draw variance, then averaged over pixels.
            v = (psq - psum * psum / K) / (K - 1)
            pvar[i0:i0 + b] = v.clamp_min(0).mean(dim=(1, 2, 3))

    return mse.cpu().numpy(), pvar.cpu().numpy(), hp.cpu().numpy()


def blind_mse(mean_square, sigma, sigma_data=SIGMA_DATA):
    """MSE a denoiser that has learned nothing would score, per image.

    This is the correction that makes the MSE metrics usable at all.

    A blurred image carries less pixel energy than a clean one. As sigma grows,
    any denoiser's best answer shrinks toward zero, so its MSE tends to the
    image's own mean square -- and the two arms therefore separate by an amount
    that reflects blur reducing energy, not the model distinguishing anything.
    The measured curves show this plainly: for a provably blind denoiser the
    clean/corrupt MSE ratio still slides from 0.996 to 0.148 across the grid.

    Worse, that contamination *drifts monotonically*, so subtracting a
    high-sigma baseline does not remove it -- it just re-centres a slope, and
    turns "no divergence anywhere" into "divergence everywhere".

    EDM's preconditioning gives the reference in closed form. With
    D(x; sigma) = c_skip * x and c_skip = sigma_data^2 / (sigma_data^2 + sigma^2),

        E||D(x + sigma*n) - x||^2 / d  =  (1 - c_skip)^2 * mean(x^2) + c_skip^2 * sigma^2

    Dividing the measured MSE by this gives a *skill* score: 1.0 means "no better
    than uninformative", below 1.0 means the model knows something. It needs no
    extra forward passes -- mean(x^2) comes from the probe images themselves --
    and it cancels the energy nuisance exactly, which is why `skill_ratio` is the
    metric to trust among the MSE-based ones.
    """
    c = sigma_data ** 2 / (sigma_data ** 2 + sigma ** 2)
    return (1.0 - c) ** 2 * mean_square + (c ** 2) * (sigma ** 2)


def _welch(mean_a, sd_a, n_a, mean_b, sd_b, n_b):
    """Unpaired difference b - a with its standard error."""
    diff = mean_b - mean_a
    se = math.sqrt(sd_a ** 2 / max(n_a, 1) + sd_b ** 2 / max(n_b, 1))
    return diff, se


def _skill_stats(mse_c, mse_x, msq_c, msq_x, sigma, sigma_data=SIGMA_DATA):
    """Energy-corrected MSE comparison. See blind_mse for why this exists."""
    sk_c = mse_c / blind_mse(msq_c, sigma, sigma_data)
    sk_x = mse_x / blind_mse(msq_x, sigma, sigma_data)
    gap = sk_x - sk_c
    n = len(gap)
    sd = float(np.std(gap, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    m_c, m_x = float(np.mean(sk_c)), float(np.mean(sk_x))
    # A model that reconstructs clean images almost perfectly drives the
    # denominator to zero. That is the point of *maximal* divergence, so the
    # ratio must go to +inf, never to nan -- see _as_flags for why nan here
    # would be read as "no divergence" and hand the model all the corrupt data.
    if m_c > _RATIO_EPS:
        ratio = m_x / m_c
    else:
        ratio = float("inf") if m_x > _RATIO_EPS else 1.0
    return {
        "skill_clean": m_c,
        "skill_corrupt": m_x,
        "skill_ratio": ratio,
        "skill_gap": float(np.mean(gap)),
        "skill_gap_se": se,
        "skill_t": float(np.mean(gap)) / se if se > 0 and np.isfinite(se) else 0.0,
    }


def _stats(paired_gap, clean, corrupt, clean_u, corrupt_u):
    """All three metrics at one sigma, paired and unpaired.

    paired_gap : per-image (corrupt - clean) over the same faces
    clean/corrupt        : per-image values over the full probe set
    clean_u/corrupt_u    : disjoint index ranges, for the unpaired design
    """
    n = len(paired_gap)
    g_mean = float(np.mean(paired_gap))
    g_sd = float(np.std(paired_gap, ddof=1)) if n > 1 else 0.0
    g_se = g_sd / math.sqrt(n) if n > 1 else float("inf")

    c_mean, x_mean = float(np.mean(clean)), float(np.mean(corrupt))
    u_diff, u_se = _welch(float(np.mean(clean_u)), float(np.std(clean_u, ddof=1)), len(clean_u),
                          float(np.mean(corrupt_u)), float(np.std(corrupt_u, ddof=1)), len(corrupt_u))

    ratio = x_mean / c_mean if c_mean > 0 else float("nan")
    # Delta method on the ratio of two means. The arms are paired, so this
    # ignores their covariance and is therefore conservative (an overestimate
    # of the SE) -- fine for a threshold, and honest about it.
    if c_mean > 0 and n > 1:
        rel_c = (np.std(clean, ddof=1) / math.sqrt(len(clean))) / c_mean
        rel_x = (np.std(corrupt, ddof=1) / math.sqrt(len(corrupt))) / x_mean if x_mean > 0 else 0.0
        ratio_se = abs(ratio) * math.sqrt(rel_c ** 2 + rel_x ** 2)
    else:
        ratio_se = float("inf")

    return {
        "n_paired": n,
        "mse_clean_mean": c_mean,
        "mse_clean_sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "mse_corrupt_mean": x_mean,
        "mse_corrupt_sd": float(np.std(corrupt, ddof=1)) if len(corrupt) > 1 else 0.0,
        # metric 1: MSE gap
        "gap_mean": g_mean,
        "gap_sd": g_sd,
        "gap_se": g_se,
        "gap_t": g_mean / g_se if g_se > 0 and np.isfinite(g_se) else 0.0,
        "gap_unpaired": u_diff,
        "gap_unpaired_se": u_se,
        "gap_unpaired_t": u_diff / u_se if u_se > 0 else 0.0,
        # metric 2: loss ratio
        "loss_ratio": ratio,
        "loss_ratio_se": ratio_se,
        # relative gap == loss_ratio - 1, kept explicitly because a threshold on
        # the raw gap cannot work: MSE spans orders of magnitude across sigma.
        "rel_gap": (ratio - 1.0) if np.isfinite(ratio) else float("nan"),
    }


@torch.no_grad()
def run_probe(net, clean_imgs, corrupt_imgs, t_grid=None, n_draws=2,
              batch_size=250, probe_seed=0, split=None, sigma_data=SIGMA_DATA,
              train_imgs=None):
    """One full probe. Returns a dict of per-sigma statistics for all metrics.

    clean_imgs / corrupt_imgs : [N, C, H, W] in [-1, 1], SAME faces in the same
        order (that is what makes the paired comparison valid).
    split : index at which to cut the set into the two disjoint halves used for
        the unpaired estimate. Defaults to N // 2.
    """
    device = clean_imgs.device
    N = clean_imgs.shape[0]
    if corrupt_imgs.shape[0] != N:
        raise ValueError(f"arms must be the same length, got {N} and {corrupt_imgs.shape[0]}")
    if t_grid is None:
        t_grid = make_t_grid()
    t_grid = np.asarray(t_grid, dtype=np.float64)
    sigmas = t_to_sigma(t_grid)
    split = N // 2 if split is None else int(split)

    was_training = getattr(net, "training", False)
    net.eval()

    # Common random numbers: one draw, reused for every sigma and both arms, and
    # identical at every probe of the run. See the module docstring.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(probe_seed))
    noise = torch.randn((N, n_draws) + tuple(clean_imgs.shape[1:]),
                        generator=gen, device=device, dtype=clean_imgs.dtype)

    # Per-image pixel energy, for the analytic blind reference. Free.
    msq_c = clean_imgs.pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()
    msq_x = corrupt_imgs.pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()
    # Fine-band energy of the clean truth: the denominator of "softness".
    hp_truth = float(hp_energy(clean_imgs).mean().item())
    # Optional third arm: the model's OWN training clean faces, with their own
    # (fixed) noise. mem_gap = (held-out MSE - train MSE) / held-out MSE per
    # level is the memorisation view used by the trigger controllers.
    noise_tr = None
    if train_imgs is not None:
        gen_tr = torch.Generator(device=device); gen_tr.manual_seed(int(probe_seed) + 1)
        noise_tr = torch.randn((train_imgs.shape[0], n_draws) + tuple(train_imgs.shape[1:]),
                               generator=gen_tr, device=device, dtype=train_imgs.dtype)

    t0 = time.time()
    per_sigma = []
    for t_val, sig in zip(t_grid, sigmas):
        mse_c, pv_c, hp_c = _arm(net, clean_imgs, noise, sig, batch_size)
        mse_x, pv_x, hp_x = _arm(net, corrupt_imgs, noise, sig, batch_size)

        rec = {"t": float(t_val), "sigma": float(sig),
               # softness: fine-band energy of the output over that of the clean
               # truth; ~0.38 at sigma 0.85 while blurred targets are eligible
               # there, ~0.53 once withdrawn (measured on final checkpoints).
               "soft_clean": float(np.mean(hp_c)) / hp_truth if hp_truth > 0 else float("nan"),
               "soft_corrupt": float(np.mean(hp_x)) / hp_truth if hp_truth > 0 else float("nan")}
        if noise_tr is not None:
            mse_tr, _, _ = _arm(net, train_imgs, noise_tr, sig, batch_size)
            c_m, tr_m = float(np.mean(mse_c)), float(np.mean(mse_tr))
            rec["mse_train_mean"] = tr_m
            rec["mem_gap"] = (c_m - tr_m) / c_m if c_m > 0 else float("nan")
        rec.update(_stats(mse_x - mse_c, mse_c, mse_x, mse_c[:split], mse_x[split:]))
        rec.update(_skill_stats(mse_c, mse_x, msq_c, msq_x, sig, sigma_data))

        # metric 3: prediction variance across noise realisations
        if n_draws > 1:
            pv_gap = pv_x - pv_c
            pvc, pvx = float(np.mean(pv_c)), float(np.mean(pv_x))
            pv_sd = float(np.std(pv_gap, ddof=1)) if N > 1 else 0.0
            pv_se = pv_sd / math.sqrt(N) if N > 1 else float("inf")
            if pvc > _RATIO_EPS:
                pv_ratio = pvx / pvc
            else:
                pv_ratio = float("inf") if pvx > _RATIO_EPS else 1.0
            rec.update({
                "predvar_clean": pvc,
                "predvar_corrupt": pvx,
                "predvar_ratio": pv_ratio,
                "predvar_gap": float(np.mean(pv_gap)),
                "predvar_gap_se": pv_se,
                "predvar_t": float(np.mean(pv_gap)) / pv_se if pv_se > 0 and np.isfinite(pv_se) else 0.0,
            })
        per_sigma.append(rec)

    if was_training:
        net.train()

    return {
        "t_grid": [float(x) for x in t_grid],
        "sigma_grid": [float(x) for x in sigmas],
        "n_images": int(N),
        "n_draws": int(n_draws),
        "probe_seed": int(probe_seed),
        "split": split,
        "sigma_data": float(sigma_data),
        "per_sigma": per_sigma,
        "probe_seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# turning a probe into a T
#
# Every rule below is a pure function of a probe result, so any of them can be
# re-evaluated offline on a logged probe without retraining. That is deliberate:
# the threshold question is the cheapest thing to get wrong and the most
# expensive to re-run.

# metric name -> (per-sigma key, key for its scale, "diverging when above")
_METRICS = {
    # Absolute MSE difference. Only meaningful relative to its own noise, since
    # MSE at sigma=0.05 and sigma=10 differ by orders of magnitude -- so the
    # natural threshold for this one is a t-statistic, not a fixed number.
    "mse_gap": {"stat": "gap_t", "raw": "gap_mean", "fixed_key": "gap_mean"},
    # Scale-free by construction, which is why a fixed threshold like 1.05 is
    # meaningful here and nowhere else.
    "loss_ratio": {"stat": "loss_ratio", "raw": "loss_ratio", "fixed_key": "loss_ratio"},
    # MSE with the pixel-energy nuisance divided out analytically (blind_mse).
    # The only MSE-based metric whose null is genuinely 1.0 at every sigma.
    "skill_ratio": {"stat": "skill_ratio", "raw": "skill_ratio", "fixed_key": "skill_ratio"},
    # Does the model's answer wobble more for corrupt inputs than clean ones?
    # Nuisance-free by construction: for a blind denoiser D = c_skip * x_t, the
    # across-noise variance is c_skip^2 * sigma^2 * I for BOTH arms, so the ratio
    # is exactly 1 whatever the images are. Verified in tests/test_probe.py.
    "pred_var": {"stat": "predvar_ratio", "raw": "predvar_ratio", "fixed_key": "predvar_ratio"},
}

# Defaults, chosen to be roughly comparable in strictness across metrics.
_FIXED_DEFAULT = {"mse_gap": 0.0, "loss_ratio": 1.05, "skill_ratio": 1.02, "pred_var": 1.05}
_ADAPTIVE_DEFAULT = 2.0     # t-statistic
_PERCENTILE_DEFAULT = 0.90
_BASELINE_DEFAULT = 2.0     # t-statistic, above the high-sigma asymptote
# How many of the top grid levels define the "cannot distinguish" asymptote.
N_BASELINE_LEVELS = 3


def baseline_levels(result, n_top=N_BASELINE_LEVELS):
    """Indices of the grid levels used as the indistinguishable null."""
    n = len(result["per_sigma"])
    return list(range(max(0, n - int(n_top)), n))


def _ratio_excess(values, null=1.0):
    """Two-sided deviation of a ratio from its null, in ratio units.

    Divergence means "the model treats the arms differently", and that shows up
    in EITHER direction. Measured on a real checkpoint, the prediction-variance
    ratio runs 0.873 -> 0.998 across the grid: it converges on its null of 1.0
    from BELOW, because the model's answer wobbles *less* on blurred inputs than
    on clean ones. A one-sided `ratio > 1.05` test never fires on that, reports
    "indistinguishable everywhere", and pins T at 0 for the whole run -- which is
    exactly what the first calibration produced for every metric.

    So the statistic is max(r/null, null/r), which is >= 1 and equals 1 only at
    the null. A threshold keeps its natural reading: 1.05 means "5% away from
    the null, either way".
    """
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(null) or null == 0:
        return np.full(v.shape, np.nan)
    r = v / null
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.maximum(np.abs(r), 1.0 / np.abs(r))
    return np.where(np.isfinite(v) & (v != 0), out, np.inf)


def _as_flags(values, threshold):
    """`values > threshold`, with non-finite values counted as DIVERGING.

    Direction matters. A nan silently compares False, which the boundary scan
    reads as "the arms are indistinguishable here" -- the permissive answer, and
    the one that quietly trains on corrupt data at every noise level. Any level
    the probe could not evaluate is therefore treated as unsafe instead.
    """
    v = np.asarray(values, dtype=np.float64)
    return np.where(np.isnan(v), True, v > threshold)


def _diverging_flags(result, metric, rule, threshold=None):
    """Per-sigma booleans: is the corrupt arm distinguishable from the clean one?

    The `baseline` rule is the one to trust, and the reason is worth stating.
    The naive null -- "no difference means gap 0, or ratio 1" -- is wrong. As
    sigma grows the denoiser's best answer tends to the dataset mean, so the MSE
    of each arm tends to that arm's own pixel variance. Blurred images have less
    pixel variance than clean ones, so the gap converges to a nonzero constant
    that has nothing to do with whether the model can tell the arms apart. A
    fixed threshold reads that residue as divergence at *every* sigma and pins T
    at 1.0 forever.

    So `baseline` measures each level against the mean of the top
    N_BASELINE_LEVELS instead of against zero. That is exactly the planner's
    "X standard deviations above the high-sigma baseline".

    It buys robustness with one assumption: that the arms genuinely are
    indistinguishable at the very top of the grid (T >~ 0.875 by default). Mild
    -- it is the premise of ambient-o -- but real, and it means a true boundary
    above 0.875 cannot be detected by this rule. The fixed and adaptive rules
    make no such assumption and are kept for exactly that comparison.
    """
    ps = result["per_sigma"]
    if rule == "adaptive":
        # "X standard errors above no-difference". Uses the paired t-statistic,
        # which is why the paired design is worth having.
        key = {"mse_gap": "gap_t", "loss_ratio": "gap_t", "skill_ratio": "skill_t",
               "pred_var": "predvar_t"}[metric]
        thr = _ADAPTIVE_DEFAULT if threshold is None else float(threshold)
        # |t|, not t: a corrupt arm that is reliably *better* than the clean one
        # is just as distinguishable as one that is reliably worse.
        return _as_flags([abs(float(r.get(key, 0.0))) for r in ps], thr)

    if rule == "baseline":
        thr = _BASELINE_DEFAULT if threshold is None else float(threshold)
        base_idx = baseline_levels(result)
        if metric in ("loss_ratio", "skill_ratio", "pred_var"):
            key = {"pred_var": "predvar_ratio"}.get(metric, metric)
            vals = np.array([float(r.get(key, np.nan)) for r in ps])
            base = float(np.mean(vals[base_idx]))
            return _as_flags(_ratio_excess(vals, base), thr)
        raw_key, se_key = {"mse_gap": ("gap_mean", "gap_se")}[metric]
        vals = np.array([float(r.get(raw_key, np.nan)) for r in ps])
        ses = np.array([float(r.get(se_key, np.inf)) for r in ps])
        base = float(np.mean(vals[base_idx]))
        with np.errstate(invalid="ignore", divide="ignore"):
            tstat = np.abs((vals - base) / ses)
        return _as_flags(tstat, thr)

    key = _METRICS[metric]["fixed_key"]
    thr = _FIXED_DEFAULT[metric] if threshold is None else float(threshold)
    vals = [float(r.get(key, float("nan"))) for r in ps]
    if metric == "mse_gap":
        # A raw difference, not a ratio; its null is 0 and the test is on size.
        return _as_flags(np.abs(vals), thr)
    # Ratio metrics: null is 1.0 for the uncorrected test.
    return _as_flags(_ratio_excess(vals, 1.0), thr)


def _boundary_index(flags, rule, q=None):
    """Lowest grid index from which the model can no longer tell the arms apart.

    Read as the top of the diverging region rather than the first crossing: a
    single noisy grid point low in the range would otherwise pin T there for the
    rest of the run. Returns len(flags) when every level still diverges, i.e.
    "no corrupt data is safe yet".
    """
    n = len(flags)
    if rule == "percentile":
        # Tolerant version of the same idea: the first k from which at least q of
        # the levels above are quiet. Survives a couple of stray points instead
        # of just one.
        qq = _PERCENTILE_DEFAULT if q is None else float(q)
        for k in range(n):
            tail = flags[k:]
            if len(tail) == 0 or (1.0 - tail.mean()) >= qq:
                return k
        return n
    idx = np.flatnonzero(flags)
    return int(idx[-1] + 1) if len(idx) else 0


def decide_T(result, metric="mse_gap", rule="adaptive", threshold=None, q=None):
    """Read a T off one probe. Returns (T, diagnostics)."""
    if metric not in _METRICS:
        raise ValueError(f"unknown probe metric {metric!r}; have {sorted(_METRICS)}")
    if rule not in ("fixed", "adaptive", "percentile", "baseline"):
        raise ValueError(f"unknown probe rule {rule!r}")

    # `percentile` is a tolerance setting on the boundary scan, not a different
    # divergence test; it reuses the fixed test and softens how the scan reads it.
    flag_rule = rule if rule in ("adaptive", "baseline") else "fixed"
    flags = _diverging_flags(result, metric, flag_rule, threshold)
    k = _boundary_index(flags, rule, q)
    grid = result["t_grid"]
    if k >= len(grid):
        t = 1.0                      # nothing is safe; corrupt data stays off
    elif k == 0:
        t = 0.0                      # everything is safe; use it all
    else:
        t = float(grid[k])
    return t, {"metric": metric, "rule": rule, "boundary_index": int(k),
               "n_diverging": int(flags.sum()), "flags": [bool(x) for x in flags]}


# The full grid of (metric, rule, threshold) combinations recorded at every
# probe. Cheap -- they are all arithmetic on numbers already computed.
COUNTERFACTUALS = [
    # energy-corrected MSE (blind_mse) -- the recommended family
    ("skill_ratio", "fixed", 1.01, None),
    ("skill_ratio", "fixed", 1.02, None),
    ("skill_ratio", "fixed", 1.05, None),
    ("skill_ratio", "baseline", 1.02, None),
    ("skill_ratio", "adaptive", 2.0, None),
    ("skill_ratio", "percentile", 1.02, 0.90),
    # baseline-corrected (see _diverging_flags)
    ("mse_gap", "baseline", 1.0, None),
    ("mse_gap", "baseline", 2.0, None),
    ("mse_gap", "baseline", 4.0, None),
    ("loss_ratio", "baseline", 1.02, None),
    ("loss_ratio", "baseline", 1.05, None),
    ("loss_ratio", "baseline", 1.10, None),
    ("pred_var", "baseline", 1.0, None),
    ("pred_var", "baseline", 2.0, None),
    # uncorrected, for the comparison that shows why correction is needed
    ("mse_gap", "adaptive", 2.0, None),
    ("mse_gap", "adaptive", 3.0, None),
    ("mse_gap", "fixed", 0.0, None),
    ("loss_ratio", "fixed", 1.02, None),
    ("loss_ratio", "fixed", 1.05, None),
    ("loss_ratio", "fixed", 1.10, None),
    ("loss_ratio", "percentile", 1.05, 0.90),
    ("pred_var", "fixed", 1.02, None),
    ("pred_var", "fixed", 1.05, None),
    ("pred_var", "adaptive", 2.0, None),
    ("pred_var", "percentile", 1.05, 0.90),
]


def all_decisions(result):
    """What every rule would have chosen from this probe. Key: metric/rule/thr."""
    out = {}
    for metric, rule, thr, q in COUNTERFACTUALS:
        if metric == "pred_var" and result.get("n_draws", 1) < 2:
            continue
        try:
            t, _ = decide_T(result, metric, rule, thr, q)
        except Exception as exc:                      # never let a rule kill a run
            out[f"{metric}/{rule}/{thr}"] = None
            out[f"{metric}/{rule}/{thr}__error"] = repr(exc)
            continue
        out[f"{metric}/{rule}/{thr}"] = t
    return out


def fit_recovery_tau(times, values, tau_grid=None):
    """Fit s(k) = s_inf - A * exp(-k / tau) to one level's softness after withdrawal.

    times: kimg since blur left this level (>= 0). values: soft_clean at those times.
    Returns (tau90, tau, s_inf, rss) with tau90 = 2.3026 * tau, the time to close 90% of
    the gap, or (None, ...) if the series cannot support a fit.

    Why a fit and not "the slope went flat": a flatness test on a probe every 100 kimg can
    only ever return a multiple of 100 kimg, so the number it reports is the probe cadence
    (auto_backplan read 100/200/300 for exactly this reason). Given tau, s_inf and A enter
    linearly, so the fit is a 1-D search with least squares inside it.
    """
    t = np.asarray(times, dtype=float); y = np.asarray(values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if len(t) < 4 or t.max() <= 0 or np.ptp(y) <= 0:
        return None, None, None, None
    if tau_grid is None:
        tau_grid = np.exp(np.linspace(np.log(5.0), np.log(800.0), 160))
    best = None
    for tau in tau_grid:
        X = np.column_stack([np.ones_like(t), -np.exp(-t / tau)])   # [s_inf, A]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((X @ coef - y) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(tau), float(coef[0]), float(coef[1]))
    rss, tau, s_inf, A = best
    if A <= 0:                      # not a rise: nothing recovered here
        return None, tau, s_inf, rss
    # A fit whose time constant sits at the edge of the grid is an extrapolation, not a
    # measurement: the series is either far too short or has not turned over yet.
    if tau <= tau_grid[0] * 1.01 or tau >= tau_grid[-1] * 0.99:
        return None, tau, s_inf, rss
    return 2.302585 * tau, tau, s_inf, rss


# ---------------------------------------------------------------------------
# closed-loop controller


class ProbeController:
    """Decides when to probe, converts probes into T, and logs everything.

    Smoothing exists because a raw per-probe T is quantised to the grid and
    jumps around; `alpha` is an EMA over probes and `max_step` caps how far one
    probe can move T.

    `monotone` is off by default on purpose. We already know from the discrete
    search that the good schedules are non-decreasing, so forcing monotonicity
    would hand the method its answer and make the "it discovered warmup" claim
    circular. It stays available as an ablation.
    """

    def __init__(self, cfg, run_dir, device):
        cfg = dict(cfg or {})
        self.every_kimg = float(cfg.get("every_kimg", 100))
        # Probing densely only where it buys something: a recovery time can never be
        # measured finer than the probe spacing, so the cadence tightens once blur has
        # started leaving levels (or past `dense_from_progress`), and stays coarse before.
        _d = cfg.get("every_kimg_dense")
        self.every_kimg_dense = float(_d) if _d else None
        self.dense_from = float(cfg.get("dense_from_progress", 1.0))
        self.n_images = int(cfg.get("n_images", 400))
        self.n_draws = int(cfg.get("n_draws", 2))
        self.batch_size = int(cfg.get("batch_size", 200))
        self.n_levels = int(cfg.get("n_levels", 20))
        self.probe_seed = int(cfg.get("probe_seed", 12345))
        self.metric = cfg.get("metric", "mse_gap")
        self.rule = cfg.get("rule", "adaptive")
        self.threshold = cfg.get("threshold", None)
        self.q = cfg.get("q", None)
        self.alpha = float(cfg.get("alpha", 0.3))
        self.monotone = bool(cfg.get("monotone", False))
        # Hold t_init until this fraction of training has passed, then start
        # obeying the probe. Motivated by the measured result that MIND tracks T
        # over the FIRST quarter of training (r = +0.83) and is nearly blind to T
        # over the last (r = +0.06), while the probe's reading -- correctly --
        # rises fastest in exactly that first quarter. Holding lets the probe set
        # the ceiling without letting it restrict data during the only phase
        # where restriction is expensive.
        self.hold_until = float(cfg.get("hold_until", 0.0))
        self.max_step = cfg.get("max_step", None)
        self.image_dir = cfg.get("image_dir")
        # Which controller turns a probe into T. "boundary" is the original
        # distinguishability rule (decide_T). The others are the automatic-
        # schedule search's candidates; each is documented at its method.
        self.controller = cfg.get("controller", "boundary")
        self.ctl = dict(cfg.get("ctl", {}) or {})
        self.train_dir = cfg.get("train_dir")          # 500 training clean faces (b0_*)
        self.n_train = int(cfg.get("n_train", 256))
        self._train = None
        self.hist = []          # per-probe list of per-level dicts (soft_clean, mem_gap, ...)
        self.hist_kimg = []     # kimg of each entry in self.hist (recovery fits need the clock)
        self.band_t = 1.0       # topdown_live: blur eligible at levels BELOW this
        # Clean-exposure controller state. The training loop keeps these up to date every
        # tick: they are what the controller steers on, and unlike a distinguishability
        # reading they MOVE when the schedule moves, which is the whole point.
        self.n_clean = int(cfg.get("n_clean", 0)) or None   # clean images in this dataset
        self.observed_clean_kimg = 0.0                      # integral of the clean batch share
        self.observed_clean_frac = None                     # most recent tick's clean share
        self.clean_frac_at_zero = None                      # that share while T = 0 (calibration)
        self.withdraw_kimg = None                           # planned moment blur leaves
        self.tau_fits = {}      # level index -> fitted tau90 (kimg to close 90% of the gap)
        self._recent_full = []  # last few probes' per-level held-out / train MSE (starve rule)
        self.withdrawn_at = {}  # level index -> kimg at which blur left it (backplan)
        self.tau_seen = {}      # level index -> observed recovery time in kimg
        self.trigger_progress = None

        self.t_grid = make_t_grid(self.n_levels)
        self.device = device
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "probe_log.jsonl")
        self.current_T = float(cfg.get("t_init", 0.0))
        self.raw_T = self.current_T
        self.next_kimg = 0.0
        self.n_probes = 0
        self.total_seconds = 0.0
        self._clean = None
        self._corrupt = None

    # -- probe set ---------------------------------------------------------

    def _resolve_image_dir(self):
        if self.image_dir:
            return self.image_dir
        base = os.environ.get("AMBIENT_BASE") or next(
            (p for p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
             if os.path.isdir(p)), None)
        if base is None:
            raise RuntimeError("cannot locate AMBIENT_BASE for the probe set")
        return os.path.join(base, "probe_holdout_64")

    def load_images(self):
        """Load both arms once, in matching order. Cached for the run."""
        if self._clean is not None:
            return
        from PIL import Image
        root = self._resolve_image_dir()
        meta_path = os.path.join(root, "probe_set.json")
        if not os.path.exists(meta_path):
            raise RuntimeError(
                f"probe set missing at {root}. Build it with\n"
                "    python dataset_creation/create_probe_holdout.py")
        with open(meta_path) as f:
            files = json.load(f)["files"]
        if len(files) < self.n_images:
            raise RuntimeError(f"probe set has {len(files)} images, asked for {self.n_images}")
        files = files[:self.n_images]

        if self.train_dir and self._train is None:
            import glob
            fs = sorted(glob.glob(os.path.join(self.train_dir, "b0_*")))[:self.n_train]
            if not fs:
                raise RuntimeError(f"no b0_* training faces under {self.train_dir}")
            tr = [(np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
                  for f in fs]
            self._train = torch.tensor(np.stack(tr), device=self.device)
        arms = []
        for arm in ("clean", "blur05"):
            imgs = []
            for fname in files:
                a = np.array(Image.open(os.path.join(root, arm, fname)).convert("RGB"),
                             dtype=np.float32)
                imgs.append((a / 127.5 - 1.0).transpose(2, 0, 1))
            arms.append(torch.tensor(np.stack(imgs), device=self.device))
        self._clean, self._corrupt = arms

    # -- scheduling --------------------------------------------------------

    def due(self, cur_nimg):
        return (cur_nimg / 1000.0) >= self.next_kimg

    def restore(self):   # noqa: C901
        """Recover state after a preemption.

        CSAIL jobs get requeued mid-run. Without this a resumed principled run
        would restart from t_init and silently retrain under a schedule nobody
        chose -- the failure would only show up as a bad MIND at the end.
        """
        if not os.path.exists(self.log_path):
            return False
        last = None
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a torn final line from a killed process
        if last is None:
            return False
        self.current_T = float(last.get("T_current",
                               last.get("T_smoothed", last.get("T_raw", self.current_T))))
        self.trigger_progress = last.get("trigger_progress", None)
        self.band_t = float(last.get("band_t", self.band_t))
        self.observed_clean_kimg = float(last.get("observed_clean_kimg", 0.0))
        _wk = last.get("withdraw_kimg")
        self.withdraw_kimg = float(_wk) if _wk is not None else None
        self.tau_fits = {int(k): float(v) for k, v in (last.get("decision", {}).get("tau_fits") or {}).items()}
        self.withdrawn_at = {int(k): v for k, v in (last.get("withdrawn_at") or {}).items()}
        self.tau_seen = {int(k): v for k, v in (last.get("decision", {}).get("tau_seen") or {}).items()}
        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.hist.append([{k: v for k, v in lvl.items() if k in ("t", "soft_clean", "soft_corrupt", "mem_gap")}
                                      for lvl in r.get("probe", {}).get("per_sigma", [])])
                    # the recovery fits need each probe's clock, not just its order
                    self.hist_kimg.append(float(r.get("kimg", 0.0)))
                    self._recent_full.append([{k: v for k, v in lvl.items() if k in ("mse_clean_mean", "mse_train_mean")}
                                              for lvl in r.get("probe", {}).get("per_sigma", [])])
            self._recent_full = self._recent_full[-8:]
        except OSError:
            pass
        self.raw_T = float(last.get("T_raw", self.current_T))
        self.n_probes = int(last.get("probe_index", 0)) + 1
        cadence = self.every_kimg
        if self.every_kimg_dense and (self.withdrawn_at
                                      or float(last.get("progress", 0.0)) >= self.dense_from):
            cadence = self.every_kimg_dense
        self.next_kimg = float(last.get("kimg", 0.0)) + cadence
        return True

    # -- automatic-schedule candidates ---------------------------------------
    #
    # Each returns (T_raw, diagnostics). They share one design fact learned from
    # the ledger: the good schedules keep blurred data at every noise level for
    # a long stretch, then withdraw it smoothly with a few hundred kimg left to
    # recover. A rule that reads a saturating skill gives a concave curve and
    # fails; these instead read either a *bottleneck* (soft_slope), a *trigger*
    # plus a fixed withdrawal shape (trigger_ramp), or a *target with the
    # remaining budget in the loop* (soft_target).

    def _levels(self, key, n_last):
        """[n_last, n_levels] array of a per-level quantity over the last probes."""
        h = self.hist[-n_last:]
        return np.array([[float(l.get(key, np.nan)) for l in rec] for rec in h])

    def _soft_slope(self, result):
        """Bottleneck detector. While blurred targets are eligible at a level,
        the output's fine-band energy there keeps creeping up as the model learns;
        when it stops creeping, learning at that level is limited by the blurred
        targets, not by time, so blur is withdrawn from that level. Withdrawal
        proceeds from the lowest eligible level upward and never reverses.
        ctl: window (probes, default 3), eps (relative slope per probe, 0.005)."""
        win = int(self.ctl.get("window", 3)); eps = float(self.ctl.get("eps", 0.005))
        grid = np.asarray(result["t_grid"])
        if len(self.hist) < win:
            return self.current_T, {"reason": "warming up", "flags": []}
        S = self._levels("soft_clean", win)                   # [win, L]
        x = np.arange(win)
        slope = np.array([np.polyfit(x, S[:, j], 1)[0] for j in range(S.shape[1])])
        rel = slope / np.maximum(S[-1], 1e-9)
        eligible = grid >= self.current_T - 1e-9              # blur still used here
        # Flat means no longer moving in EITHER direction: softness first FALLS
        # (the model learns to strip noise) and only later creeps up, and a
        # falling level is still learning, not bottlenecked.
        flat = np.abs(rel) < eps
        k = int(np.searchsorted(grid, self.current_T - 1e-9))  # first eligible level
        while k < len(grid) and flat[k]:
            k += 1
        t = 1.0 if k >= len(grid) else float(grid[k])
        t = max(t, self.current_T)
        return t, {"rel_slope": [float(v) for v in rel], "flat": [bool(v) for v in flat],
                   "eligible": [bool(v) for v in eligible], "boundary_index": k}

    def _trigger_ramp(self, result, progress):
        """Two-stage: keep T=0 until a memorisation trigger fires, then follow a
        fixed concave withdrawal over the remaining budget. Trigger = mean
        memorisation gap over levels with t <= gap_t_max exceeds gap_thr.
        ctl: gap_thr (0.08), gap_t_max (0.35), shape ('concave' -> 0.75 at the
        midpoint, 'linear'), t_end (0.95), force_at (progress, default 0.6:
        safety net, logged if used)."""
        thr = float(self.ctl.get("gap_thr", 0.08)); tmax = float(self.ctl.get("gap_t_max", 0.35))
        t_end = float(self.ctl.get("t_end", 0.95)); force_at = float(self.ctl.get("force_at", 0.6))
        ps = result["per_sigma"]
        gaps = [float(r.get("mem_gap", np.nan)) for r in ps if r["t"] <= tmax + 1e-9]
        g = float(np.nanmean(gaps)) if gaps else float("nan")
        fired_now = False
        if self.trigger_progress is None:
            if np.isfinite(g) and g > thr:
                self.trigger_progress = progress; fired_now = True
            elif progress >= force_at:
                self.trigger_progress = progress; fired_now = True
        t = self.ramp_T(progress, t_end)
        return t, {"mem_gap_low": g, "fired_now": fired_now, "forced": bool(fired_now and not (np.isfinite(g) and g > thr)),
                   "trigger_progress": self.trigger_progress}

    def ramp_T(self, progress, t_end=0.95):
        if self.trigger_progress is None:
            return 0.0
        u = (progress - self.trigger_progress) / max(1.0 - self.trigger_progress, 1e-9)
        u = float(np.clip(u, 0.0, 1.0))
        shape = self.ctl.get("shape", "concave")
        if shape == "linear":
            return t_end * u
        return float(np.interp(u, [0.0, 0.5, 1.0], [0.0, 0.75 * t_end / 0.95, t_end]))

    def _soft_target(self, result, progress):
        """Budget-in-the-loop. For each level, the softness deficit is how far the
        output's fine-band energy sits below the best seen at any level that has
        already been withdrawn (or, before any withdrawal, below the top level's
        own running maximum). Withdrawal at a level costs a recovery time tau
        (ctl, default 300 kimg, measured on the ledger's trajectories); the
        controller withdraws a level when the remaining budget falls to
        n_pending * tau / parallelism, i.e. it plans backwards from the end so
        every level is recovered exactly at the finish and blurred data is used
        as long as possible before that.
        ctl: tau_kimg (300), parallel (4: levels recovering concurrently),
        total_kimg (2000), t_end (0.95)."""
        tau = float(self.ctl.get("tau_kimg", 300)); par = float(self.ctl.get("parallel", 4))
        total = float(self.ctl.get("total_kimg", 2000)); t_end = float(self.ctl.get("t_end", 0.95))
        grid = np.asarray(result["t_grid"])
        remaining = (1.0 - progress) * total
        pending = grid[(grid >= self.current_T - 1e-9) & (grid <= t_end + 1e-9)]
        n_pending = len(pending)
        # how many levels must already be withdrawn to finish on time
        need = n_pending - int(np.floor(remaining / tau * par))
        t = self.current_T
        if need > 0:
            k = int(np.searchsorted(grid, self.current_T - 1e-9)) + need
            t = min(float(grid[k]) if k < len(grid) else 1.0, t_end)
        return max(t, self.current_T), {"remaining_kimg": remaining, "n_pending": n_pending, "need": int(need)}

    def _starve(self, result):
        """Data-starvation rule, present-step (control). A level is *starved*
        when the model keeps getting better on its own 500 clean training faces
        there while its held-out error has stopped improving: it is memorising,
        i.e. it has run out of clean data at that level, so blurred data is
        wanted there. Blur is used where starved and withdrawn elsewhere: T is
        the lowest level from which at least half of the levels above are
        starved (no starved level -> T = 1, withdraw everywhere).
        ctl: window (4), eps_train (rel slope/probe, -0.005), eps_hold (0.005)."""
        win = int(self.ctl.get("window", 4)); e_tr = float(self.ctl.get("eps_train", -0.005)); e_ho = float(self.ctl.get("eps_hold", 0.005))
        grid = np.asarray(result["t_grid"])
        if len(self.hist) < win:
            return self.current_T, {"reason": "warming up"}
        x = np.arange(win)
        def rel_slope(key):
            S = np.array([[float(l.get(key, np.nan)) for l in rec] for rec in [h for h in self.hist[-win:]]])
            return np.array([np.polyfit(x, S[:, j], 1)[0] for j in range(S.shape[1])]) / np.maximum(np.abs(S[-1]), 1e-12)
        # hist stores only soft/mem keys; held-out and train MSE come from the probe log tail
        S_ho = np.array([[float(l.get("mse_clean_mean", np.nan)) for l in rec] for rec in self._recent_full[-win:]])
        S_tr = np.array([[float(l.get("mse_train_mean", np.nan)) for l in rec] for rec in self._recent_full[-win:]])
        s_ho = np.array([np.polyfit(x, S_ho[:, j], 1)[0] for j in range(S_ho.shape[1])]) / np.maximum(S_ho[-1], 1e-12)
        s_tr = np.array([np.polyfit(x, S_tr[:, j], 1)[0] for j in range(S_tr.shape[1])]) / np.maximum(S_tr[-1], 1e-12)
        starved = (s_tr < e_tr) & (np.abs(s_ho) < e_ho)
        t = 1.0
        for k in range(len(grid)):
            if starved[k:].mean() >= 0.5:
                t = float(grid[k]); break
        return t, {"starved": [bool(v) for v in starved], "slope_train": [float(v) for v in s_tr],
                   "slope_holdout": [float(v) for v in s_ho]}

    def _backplan(self, result, progress, kimg):
        """Self-calibrating backward plan -- the controller the look-ahead result
        points to. It never asks "is quality better now?" (that question is
        confidently inverted at every affordable horizon). It asks only: how long
        does a noise level take to recover once blurred targets are withdrawn
        from it, and therefore how late can withdrawal start and still finish?

        Recovery time is measured on this run, not assumed. When a level is
        withdrawn, its fine-band energy (soft_clean) rises and then plateaus; the
        elapsed kimg from withdrawal to plateau IS tau for that level. The mean
        of the observed taus replaces the prior as soon as one level has
        plateaued, and withdrawal of the remaining levels is paced so the last
        one gets tau kimg before the end.

        ctl: tau_prior (300), parallel (levels recovering at once, 4),
        total_kimg (2000), t_end (0.95), window (3 probes for the slope),
        eps_flat (0.004 relative slope per probe = plateaued),
        tau_min/tau_max (100 / 900 kimg clip)."""
        c = self.ctl
        tau_prior = float(c.get("tau_prior", 300)); par = float(c.get("parallel", 4))
        total = float(c.get("total_kimg", 2000)); t_end = float(c.get("t_end", 0.95))
        win = int(c.get("window", 3)); eps = float(c.get("eps_flat", 0.004))
        tmin, tmax = float(c.get("tau_min", 100)), float(c.get("tau_max", 900))
        grid = np.asarray(result["t_grid"])

        # -- record which levels are withdrawn, and when
        for j, t in enumerate(grid):
            if t < self.current_T - 1e-9 and j not in self.withdrawn_at:
                self.withdrawn_at[j] = kimg
        # -- observe tau: a withdrawn level whose softness has stopped rising
        if len(self.hist) >= win:
            S = self._levels("soft_clean", win); x = np.arange(win)
            for j, k0 in list(self.withdrawn_at.items()):
                if j in self.tau_seen or kimg - k0 < 50:
                    continue
                col = S[:, j]
                if not np.all(np.isfinite(col)):
                    continue
                rel = np.polyfit(x, col, 1)[0] / max(abs(col[-1]), 1e-9)
                if abs(rel) < eps:
                    self.tau_seen[j] = kimg - k0
        tau = float(np.mean(list(self.tau_seen.values()))) if self.tau_seen else tau_prior
        tau = float(np.clip(tau, tmin, tmax))

        remaining = (1.0 - progress) * total
        pending = grid[(grid >= self.current_T - 1e-9) & (grid <= t_end + 1e-9)]
        n_pending = len(pending)
        # Levels that can still be started later and finish in time; the rest
        # must start now. Withdrawal is as late as possible, which is the point.
        can_wait = int(np.floor(max(remaining - tau, 0.0) / max(tau, 1e-9) * par))
        need = n_pending - can_wait
        t = self.current_T
        if need > 0:
            k = int(np.searchsorted(grid, self.current_T - 1e-9)) + need
            t = min(float(grid[k]) if k < len(grid) else 1.0, t_end)
        return max(t, self.current_T), {"tau": tau, "tau_seen": {str(k): v for k, v in self.tau_seen.items()},
                                        "remaining_kimg": remaining, "n_pending": n_pending,
                                        "can_wait": can_wait, "need": int(need)}

    def _exposure(self, result, progress, kimg):
        """Withdraw blur when the projected CLEAN EXPOSURE hits its target.

        The mechanism this steers on, measured on the existing table rather than assumed:
        clean exposure (the integral of the clean share of the batch) predicts the end-of-run
        memorisation gap at rank 0.97; above ~1800 passes over the clean set it correlates
        +0.73 with a worse score and below that -0.16; the best runs sit at 1000-1660 passes.
        Scaling that band by the clean-set size predicted which schedule wins at 250, 500 and
        1000 clean faces, out of sample, 3 for 3.

        So: project total clean exposure to the end of training as a function of when blur is
        withdrawn, and pick the withdrawal time that lands on the target. Withdrawing raises
        the clean share (the batch becomes mostly clean), so the projection is
            E_total(k_w) = E_now + a (k_w - k) + f_hi (K - k_w)
        with `a` the clean share observed while blur is in use and `f_hi` the share after
        withdrawal. Both are measured on this run, not assumed. Solving for E_total = E* gives
        the withdrawal time, recomputed every probe, so an error in the projection corrects
        itself as the run goes.

        Two constants, both with stated origins and neither tuned on this run:
          target_epochs  1259, the median of the base table's top eight (a calibrated scalar);
          tau_rec        300 kimg, the recovery time measured on the look-ahead branches.
        Where they conflict the recovery floor wins: too little recovery is catastrophic
        (0.032+ when a level gets under 200 kimg), overshooting the exposure band is graded.

        ctl: target_epochs, tau_rec, total_kimg, t_end (0.95), f_hi (fallback 0.96 if the
        run has not yet observed the post-withdrawal share), shape ('jump' or 'ramp').
        """
        c = self.ctl
        total = float(c.get("total_kimg", 2000)); t_end = float(c.get("t_end", 0.95))
        tau_rec = float(c.get("tau_rec", 300)); target_epochs = float(c.get("target_epochs", 1259))
        n_clean = self.n_clean or int(c.get("n_clean", 500))
        a = self.clean_frac_at_zero if self.clean_frac_at_zero is not None else 0.019
        f_hi = float(c.get("f_hi", 0.96))
        if self.observed_clean_frac is not None and self.current_T >= t_end - 1e-9:
            # Only believe a post-withdrawal reading that actually looks post-withdrawal. The tick
            # straight after the drop still carries prefetched batches built under the old
            # threshold, so it reports the OLD clean share; taking it made f_hi = 0.046 against
            # a = 0.045 on exp_c1250, and the solve below divides by (a - f_hi).
            if float(self.observed_clean_frac) > a + 0.2:
                f_hi = float(self.observed_clean_frac)
        target_kimg = target_epochs * n_clean / 1000.0      # clean exposure we are aiming at
        E_now = float(self.observed_clean_kimg)

        # Once blur is gone the plan is history: re-solving from the post-drop exposure answers a
        # question nobody asked ("when should I withdraw, given I already have?") and its answer
        # wanders. Freeze it, so the logged field keeps meaning the decision the run made.
        if self.withdraw_kimg is not None and self.current_T >= t_end - 1e-9:
            proj_done = E_now + f_hi * max(total - kimg, 0.0)
            return max(self.current_T, t_end), {
                "clean_kimg_so_far": E_now, "clean_epochs_so_far": E_now * 1000.0 / n_clean,
                "target_clean_kimg": target_kimg, "target_epochs": target_epochs,
                "n_clean": n_clean, "a_clean_frac": a, "f_hi": f_hi,
                "withdraw_kimg": self.withdraw_kimg, "withdraw_kimg_unclipped": self.withdraw_kimg,
                "recovery_floor_binding": bool(self.withdraw_kimg >= total - tau_rec - 1e-9),
                "projected_total_clean_kimg": proj_done,
                "projected_total_epochs": proj_done * 1000.0 / n_clean,
                "plan_frozen": True, "tau_rec": tau_rec}

        denom = a - f_hi
        # A denominator this small means the two shares are indistinguishable, which happens only
        # when a reading is wrong -- dividing by it produces a plan of millions of kimg.
        if abs(denom) < 0.05:
            k_w = total - tau_rec
        else:
            k_w = (target_kimg - E_now + a * kimg - f_hi * total) / denom
        k_w_raw = k_w
        k_w = min(k_w, total - tau_rec)                     # recovery floor wins
        k_w = max(k_w, kimg if self.withdraw_kimg is None else min(self.withdraw_kimg, k_w))
        self.withdraw_kimg = float(k_w)
        t = t_end if kimg >= k_w - 1e-9 else 0.0
        if c.get("shape") == "ramp" and kimg >= k_w - 1e-9:
            span = max(total - k_w, 1e-9)
            t = t_end * float(np.clip((kimg - k_w) / span, 0.0, 1.0))
        t = max(t, self.current_T)                          # withdrawal never reverses
        proj = E_now + a * max(k_w - kimg, 0.0) + f_hi * max(total - k_w, 0.0)
        return t, {"clean_kimg_so_far": E_now, "clean_epochs_so_far": E_now * 1000.0 / n_clean,
                   "target_clean_kimg": target_kimg, "target_epochs": target_epochs,
                   "n_clean": n_clean, "a_clean_frac": a, "f_hi": f_hi,
                   "withdraw_kimg": self.withdraw_kimg, "withdraw_kimg_unclipped": k_w_raw,
                   "recovery_floor_binding": bool(k_w_raw > total - tau_rec),
                   "projected_total_clean_kimg": proj, "plan_frozen": False, "tau_rec": tau_rec,
                   "projected_total_epochs": proj * 1000.0 / n_clean}

    def _topdown_live(self, result, progress, kimg):
        """Top-down withdrawal with the recovery time measured on this run.

        Blur leaves each noise level tau90(t) kimg before the end, highest level first.
        That order is what makes the measurement honest: the slow high levels are withdrawn
        FIRST, so by the time the fast low levels have to be planned, their own neighbours
        above them have already produced fitted recovery times. (A threshold T withdraws in
        the opposite order, which is why auto_backplan could only ever measure the fast
        levels and then apply that number to the slow ones.)

        tau90 per level comes from fit_recovery_tau on the level's softness since its own
        withdrawal; with two or more fits, log tau90 is regressed on t and used for the
        levels still pending. Before any fit, `prior_points` are the recovery times measured
        offline on the look-ahead branches, not tuned on MIND.
        ctl: prior_points [[t, tau90], ...], total_kimg, tau_min/tau_max (kimg clip),
        min_points (probes needed before a level is fitted, 4).
        """
        c = self.ctl
        total = float(c.get("total_kimg", 2000))
        tmin, tmax = float(c.get("tau_min", 50)), float(c.get("tau_max", 800))
        min_pts = int(c.get("min_points", 4))
        prior = c.get("prior_points") or [[0.1, 100], [0.3, 100], [0.5, 200], [0.7, 300], [0.9, 300]]
        pt = np.array([float(p[0]) for p in prior]); pv = np.array([float(p[1]) for p in prior])
        grid = np.asarray(result["t_grid"])

        for j, t in enumerate(grid):                     # withdrawn = at or above the cutoff
            if t >= self.band_t - 1e-9 and j not in self.withdrawn_at:
                self.withdrawn_at[j] = kimg
        for j, k0 in self.withdrawn_at.items():
            idx = [i for i, k in enumerate(self.hist_kimg) if k >= k0 - 1e-9]
            if len(idx) < min_pts:
                continue
            ts = [self.hist_kimg[i] - k0 for i in idx]
            ys = [self.hist[i][j].get("soft_clean", np.nan) for i in idx]
            tau90, _, _, _ = fit_recovery_tau(ts, ys)
            if tau90 is not None:
                self.tau_fits[j] = float(np.clip(tau90, tmin, tmax))

        if len(self.tau_fits) >= 2:                      # log tau90 ~ a + b t, from this run
            xs = np.array([grid[j] for j in self.tau_fits]); ys = np.log(np.array(list(self.tau_fits.values())))
            b, a = np.polyfit(xs, ys, 1)
            tau_of = lambda t: float(np.clip(np.exp(a + b * np.asarray(t)), tmin, tmax))
            source = "fitted"
        else:
            tau_of = lambda t: float(np.clip(np.interp(t, pt, pv), tmin, tmax))
            source = "prior"

        remaining = max(1.0 - progress, 0.0) * total
        taus = np.array([tau_of(t) for t in grid])
        due = taus >= remaining                          # these levels must be withdrawn by now
        band = float(grid[np.argmax(due)]) if due.any() else 1.0
        if due.all():
            band = 0.0
        band = min(band, self.band_t)                    # withdrawal never reverses
        return band, {"band_t": band, "remaining_kimg": remaining, "tau_source": source,
                      "tau_at": {str(t): round(tau_of(t), 1) for t in (0.1, 0.3, 0.5, 0.7, 0.9)},
                      "tau_fits": {str(k): round(v, 1) for k, v in self.tau_fits.items()},
                      "withdrawn_at": {str(k): v for k, v in self.withdrawn_at.items()}}

    def tick(self, progress):
        """Between probes: controllers whose T is a function of progress keep
        moving (the trigger ramp), so the applied T is smooth rather than a
        staircase at the probe cadence."""
        if self.controller == "exposure" and self.withdraw_kimg is not None:
            # The withdrawal moment is a kimg, not a probe index: applying it at the tick it
            # falls on keeps the plan from quantising to the probe cadence (the failure that
            # made auto_backplan's "measurements" multiples of its probe spacing).
            total = float(self.ctl.get("total_kimg", 2000)); t_end = float(self.ctl.get("t_end", 0.95))
            kimg = progress * total
            if kimg >= self.withdraw_kimg - 1e-9:
                if self.ctl.get("shape") == "ramp":
                    span = max(total - self.withdraw_kimg, 1e-9)
                    self.current_T = max(self.current_T,
                                         t_end * float(np.clip((kimg - self.withdraw_kimg) / span, 0.0, 1.0)))
                else:
                    self.current_T = max(self.current_T, t_end)
        if self.controller == "trigger_ramp" and self.trigger_progress is not None:
            self.current_T = float(np.clip(self.ramp_T(progress, float(self.ctl.get("t_end", 0.95))), 0.0, 1.0))

    # -- the step ----------------------------------------------------------

    def step(self, net, cur_nimg, total_kimg, apply_to_schedule):
        """Run one probe, log it, and update T. Returns the log record."""
        self.load_images()
        kimg = cur_nimg / 1000.0
        result = run_probe(net, self._clean, self._corrupt, t_grid=self.t_grid,
                           n_draws=self.n_draws, batch_size=self.batch_size,
                           probe_seed=self.probe_seed, train_imgs=self._train)
        self.hist.append([{k: v for k, v in lvl.items() if k in ("t", "soft_clean", "soft_corrupt", "mem_gap")}
                          for lvl in result["per_sigma"]])
        self.hist_kimg.append(kimg)
        self._recent_full.append([{k: v for k, v in lvl.items() if k in ("mse_clean_mean", "mse_train_mean")}
                                  for lvl in result["per_sigma"]])
        self._recent_full = self._recent_full[-8:]
        progress = kimg / total_kimg if total_kimg else 0.0

        if self.controller == "boundary":
            t_raw, diag = decide_T(result, self.metric, self.rule, self.threshold, self.q)
        elif self.controller == "soft_slope":
            t_raw, diag = self._soft_slope(result)
        elif self.controller == "trigger_ramp":
            t_raw, diag = self._trigger_ramp(result, progress)
        elif self.controller == "soft_target":
            t_raw, diag = self._soft_target(result, progress)
        elif self.controller == "starve":
            t_raw, diag = self._starve(result)
        elif self.controller == "backplan":
            t_raw, diag = self._backplan(result, progress, kimg)
        elif self.controller == "topdown_live":
            t_raw, diag = self._topdown_live(result, progress, kimg)
        elif self.controller == "exposure":
            t_raw, diag = self._exposure(result, progress, kimg)
        else:
            raise ValueError(f"unknown controller {self.controller!r}")
        prev = self.current_T
        t_new = (1.0 - self.alpha) * prev + self.alpha * t_raw if self.n_probes > 0 else t_raw
        if self.max_step is not None:
            step = float(self.max_step)
            t_new = min(max(t_new, prev - step), prev + step)
        if self.monotone:
            t_new = max(t_new, prev)
        t_new = float(np.clip(t_new, 0.0, 1.0))

        if self.controller == "topdown_live":
            # t_raw is the top of the blur window, not a threshold: no EMA, no max_step
            # (both would blur a cutoff that is already a planned quantity), and T stays 0.
            t_new = float(np.clip(t_raw, 0.0, 1.0))
            if apply_to_schedule:
                self.band_t = t_new
            self.current_T = 0.0
        self.raw_T = t_raw
        if apply_to_schedule and self.controller != "topdown_live":
            self.current_T = t_new

        rec = {
            "probe_index": self.n_probes,
            "kimg": kimg,
            "progress": kimg / total_kimg if total_kimg else 0.0,
            "T_raw": t_raw,
            "T_smoothed": t_new,
            "T_previous": prev,
            "T_applied": self.current_T if apply_to_schedule else None,
            "applied": bool(apply_to_schedule),
            # The controller's actual state, which is NOT T_smoothed while the
            # hold is in force. restore() must read this one: keying off
            # T_smoothed would resume a preempted run at a T it never trained at.
            "T_current": self.current_T,
            "band_t": self.band_t,
            "observed_clean_kimg": self.observed_clean_kimg,
            "withdraw_kimg": self.withdraw_kimg,
            "trigger_progress": self.trigger_progress,
            "withdrawn_at": {str(k): v for k, v in self.withdrawn_at.items()},
            "controller": {"name": self.controller, "ctl": self.ctl,
                           "metric": self.metric, "rule": self.rule,
                           "threshold": self.threshold, "q": self.q,
                           "alpha": self.alpha, "monotone": self.monotone,
                           "max_step": self.max_step},
            "decision": diag,
            "counterfactual_T": all_decisions(result),
            "probe": result,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())      # preemption can arrive at any moment

        self.n_probes += 1
        self.total_seconds += result["probe_seconds"]
        cadence = self.every_kimg
        if self.every_kimg_dense and (self.withdrawn_at or progress >= self.dense_from):
            cadence = self.every_kimg_dense
        self.next_kimg = kimg + cadence
        return rec
