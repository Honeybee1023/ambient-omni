"""The direct test: does letting blurry data in at a noise band actually help?

Everything else we have measured is a proxy — distinguishability, memorisation,
gradient agreement. This measures the thing itself. From one saved checkpoint,
train two copies for K steps under identical conditions (same clean data, same
noise draws, same seeds) except that in arm A blurry images are eligible for
denoising tasks whose noise level falls in [band_lo, band_hi), and in arm B
they are never eligible. Then measure held-out clean error at every noise
level. delta = A - B, per level. Negative in the opened band means blurry data
helped there; positive means it hurt. Spillover to other bands is reported too.

This is what the first-order cosine is an approximation of, so the two can be
compared at the same checkpoint and band. It also captures many-step effects
the cosine cannot see -- in particular whether blurry data suppresses
memorisation of the 500 clean faces -- which is why the memorisation gap is
reported alongside.

The loss and eligibility mechanics are the training loop's own: an eligible
blurry image at noise level sigma_t enters with sigma_tn = sigma(band_lo) and
the ambient weighting, exactly as under a threshold T = band_lo.

Usage:
    python analysis/probe_counterfactual_checkpoint.py \\
        --ckpt $AMBIENT_BASE/probe_ckpts/dyn_gt_warmup_s0/network-snapshot-000250.pkl \\
        --band 0.20 0.35 --steps 200 --out $AMBIENT_BASE/generated/cf_warmup250_b020.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import copy
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                                      # noqa: F401
from torch_utils import persistence                # noqa: F401
from training import probe as P
from training.loss import AmbientEDMLoss

P_MEAN, P_STD = -1.2, 1.2


def load_paths(paths, device):
    out = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        out.append((a / 127.5 - 1.0).transpose(2, 0, 1))
    return torch.tensor(np.stack(out), device=device)


def per_level_mse(net, imgs, noise, sigmas, batch):
    return np.array([P._arm(net, imgs, noise, s, batch)[0].mean() for s in sigmas])


# Training's sampler is built with s_max=4 (run_dyn_job.sh --s_max=4), which
# keeps an eligible image's noise level at least buffer_factor * sigma_min:
#   buffer_factor = sqrt(1 + 1/(s_max-1)) = 1.155
# so the ambient weight sigma_t^4 / (sigma_t^2 - sigma_tn^2)^2 never exceeds
# ~16. Without this, a sample drawn just above the threshold gets a weight in
# the hundreds and dominates the batch -- the first version of this test did
# that, and its deltas were global and smooth across all noise levels, the
# signature of a destabilised arm rather than of blur helping in a band.
S_MAX = 4.0
BUFFER = (1 + 1 / (S_MAX - 1)) ** 0.5


def train_arm(net0, clean, blurry, band, steps, batch, lr, seed, device, open_blur):
    """K Adam steps from net0. Blurry images eligible only if open_blur and sigma_t in band."""
    net = copy.deepcopy(net0).train().requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = AmbientEDMLoss()
    g = torch.Generator(device=device); g.manual_seed(seed)
    sig_lo = float(P.t_to_sigma(band[0])); sig_hi = float(P.t_to_sigma(band[1]))
    n_c, n_b = clean.shape[0], blurry.shape[0]
    used_blur = 0
    for _ in range(steps):
        # Training noise distribution, one sigma per slot in the batch.
        sigma_t = (torch.randn(batch, generator=g, device=device) * P_STD + P_MEAN).exp()
        # Eligible = in band AND safely above the threshold, as the sampler enforces.
        in_band = (sigma_t >= sig_lo * BUFFER) & (sigma_t < sig_hi)
        take_blur = in_band if open_blur else torch.zeros_like(in_band)
        # Slot draws: an in-band slot draws from the whole pool (500 clean +
        # 26k blurry, so ~98% blurry), an out-of-band slot from clean only.
        # This is what the training sampler does under threshold T = band_lo.
        idx_pool = torch.randint(0, n_c + n_b, (batch,), generator=g, device=device)
        idx_clean = torch.randint(0, n_c, (batch,), generator=g, device=device)
        is_blur = take_blur & (idx_pool >= n_c)
        x0 = torch.where(is_blur[:, None, None, None],
                         blurry[(idx_pool - n_c).clamp(min=0)], clean[idx_clean])
        # Eligible blurry images enter pre-noised to sigma(band_lo), as the
        # training loop does with sigma_min; clean images enter as-is.
        sigma_tn = torch.where(is_blur, torch.full_like(sigma_t, sig_lo), torch.zeros_like(sigma_t))
        sigma_tn = torch.where(sigma_tn > sigma_t, torch.zeros_like(sigma_tn), sigma_tn)
        x_tn = x0 + torch.randn(x0.shape, generator=g, device=device) * sigma_tn[:, None, None, None]
        loss, *_ = loss_fn(net=net, x_tn=x_tn, sigma_tn=sigma_tn, sigma_t=sigma_t, labels=None)
        opt.zero_grad(set_to_none=True)
        loss.mean().backward()
        opt.step()
        used_blur += int(is_blur.sum())
    return net.eval().requires_grad_(False), used_blur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--band", type=float, nargs=2, required=True, help="T range in which blur is eligible in arm A")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="nominal training lr (train.py default 10e-4)")
    ap.add_argument("--lr_rampup_kimg", type=float, default=10000,
                    help="training_loop default; effective lr = lr * min(kimg/rampup, 1)")
    ap.add_argument("--train_dir", default=None, help="dir with b0_* and b5_* (default: the dyn dataset)")
    ap.add_argument("--probe_dir", default=f"{AMBIENT_BASE}/probe_holdout_64")
    ap.add_argument("--n_eval", type=int, default=160)
    ap.add_argument("--n_levels", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--null", action="store_true",
                    help="noise floor: neither arm sees blur, arms differ only by seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    train_dir = args.train_dir or next(
        d for d in (f"{AMBIENT_BASE}/annotated_datasets/celeba_dynamic_t_v2_b0b5",
                    f"{AMBIENT_BASE}/annotated_datasets/celeba_dynamic_t_v2") if os.path.isdir(d))
    clean_paths = sorted(glob.glob(os.path.join(train_dir, "b0_*.jpg")))
    blur_paths = sorted(glob.glob(os.path.join(train_dir, "b5_*.jpg")))
    t0 = time.time()
    clean = load_paths(clean_paths, device)
    blurry = load_paths(blur_paths, device)
    with open(os.path.join(args.probe_dir, "probe_set.json")) as f:
        files = json.load(f)["files"][:args.n_eval]
    hold = load_paths([os.path.join(args.probe_dir, "clean", f) for f in files], device)
    train_eval = clean[:args.n_eval]                  # memorisation arm
    print(f"loaded {len(clean_paths)} clean + {len(blur_paths)} blurry, {args.n_eval} held-out  ({time.time()-t0:.0f}s)")

    with open(args.ckpt, "rb") as f:
        net0 = pickle.load(f)["ema"].to(device).eval()

    # Training ramps the learning rate linearly over lr_rampup_kimg (default
    # 10,000 kimg -- longer than the whole 2,000-kimg run), so the step size at
    # a given checkpoint is lr * kimg / 10000. At kimg 250 that is 40x below
    # nominal. Match it, or the K steps here are nothing like K training steps.
    import re
    m = re.search(r"network-snapshot-(\d+)\.pkl$", os.path.basename(args.ckpt))
    ckpt_kimg = int(m.group(1)) if m else args.lr_rampup_kimg
    eff_lr = args.lr * min(ckpt_kimg / max(args.lr_rampup_kimg, 1e-8), 1.0)
    print(f"checkpoint kimg {ckpt_kimg}: effective lr {eff_lr:.2e} (nominal {args.lr:.0e}, ramp {args.lr_rampup_kimg} kimg)")
    args.lr = eff_lr

    t_grid = P.make_t_grid(args.n_levels); sigmas = P.t_to_sigma(t_grid)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    ev_noise = torch.randn((args.n_eval, 2) + tuple(hold.shape[1:]), generator=gen, device=device)

    def evaluate(net):
        return {"holdout": per_level_mse(net, hold, ev_noise, sigmas, 80),
                "train": per_level_mse(net, train_eval, ev_noise, sigmas, 80)}

    e0 = evaluate(net0)
    res = {}
    arms = (("A_blur_in_band", True, args.seed), ("B_no_blur", False, args.seed))
    if args.null:
        # Noise floor: two arms that differ ONLY by seed, neither sees blur.
        arms = (("A_blur_in_band", False, args.seed + 1), ("B_no_blur", False, args.seed))
    for arm, open_blur, seed in arms:
        t1 = time.time()
        net, nb = train_arm(net0, clean, blurry, args.band, args.steps, args.batch, args.lr,
                            seed, device, open_blur)
        res[arm] = dict(evaluate(net), blurry_images_used=nb, seconds=time.time() - t1)
        del net; torch.cuda.empty_cache()

    dA, dB = res["A_blur_in_band"]["holdout"], res["B_no_blur"]["holdout"]
    delta = dA - dB
    rel = delta / dB
    gapA = res["A_blur_in_band"]["holdout"] - res["A_blur_in_band"]["train"]
    gapB = res["B_no_blur"]["holdout"] - res["B_no_blur"]["train"]
    in_band = (t_grid >= args.band[0]) & (t_grid < args.band[1])

    print(f"\n{os.path.basename(args.ckpt)}  blur opened in T [{args.band[0]:.2f},{args.band[1]:.2f})  "
          f"{args.steps} steps x {args.batch}   (arm A used {res['A_blur_in_band']['blurry_images_used']} blurry images)")
    print(f"  {'T':>6} {'holdout B':>10} {'holdout A':>10} {'rel delta':>10} {'memgap B':>9} {'memgap A':>9}")
    for i, t in enumerate(t_grid):
        mark = " <- opened" if in_band[i] else ""
        print(f"  {t:>6.3f} {dB[i]:>10.5f} {dA[i]:>10.5f} {rel[i]:>+10.3f} {gapB[i]/dB[i]:>+9.3f} {gapA[i]/dA[i]:>+9.3f}{mark}")
    print(f"\n  in-band mean rel delta: {rel[in_band].mean():+.4f}   "
          f"(negative = blurry data HELPED held-out clean error in the band)")
    print(f"  out-of-band mean rel delta: {rel[~in_band].mean():+.4f}")

    out = {"config": vars(args), "t_grid": [float(x) for x in t_grid],
           "before": {k: v.tolist() for k, v in e0.items()},
           "A_blur_in_band": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res["A_blur_in_band"].items()},
           "B_no_blur": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res["B_no_blur"].items()},
           "rel_delta": rel.tolist(), "in_band": in_band.tolist(),
           "in_band_mean_rel_delta": float(rel[in_band].mean())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
