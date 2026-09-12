"""Offline gradient-agreement probe on saved checkpoints.

Does a batch of blurry faces push the network in the same direction a batch
of clean faces would, at a given noise level, at a given point in training?

Why this is the right quantity: a gradient step on blurry data changes the
loss on CLEAN data by, to first order, -(step size) * <g_clean, g_blur>. So the
sign of that inner product answers "does blurry data help the clean objective
right now" directly, not by proxy. Normalised to a cosine it is dimensionless
and comparable across noise levels, training progress and datasets.

Prediction from the coarse-to-fine account: cosine high at high noise always;
high at low noise EARLY (she is learning coarse structure, which blur keeps);
falling at low noise LATE (she is learning fine detail, which blur destroys).

Measurement details that matter:
- g_clean comes from HELD-OUT clean faces. Her 500 training faces are
  memorised late in training and their gradient goes to ~0.
- g_blur comes from the held-out faces' blurred twins (same faces, same noise:
  the paired design), so the only difference between the arms is the blur.
- Noise levels are grouped into bands in T; sigma is drawn uniformly in T
  within a band. The loss is the training loss (AmbientEDMLoss with
  sigma_tn = 0), so these are the gradients training would actually take.
- Gradients are accumulated over several noise draws BEFORE the cosine, since
  cosine of two noisy vectors is biased toward zero.

Usage:
    python analysis/probe_gradcos_checkpoint.py \\
        --ckpt_dir $AMBIENT_BASE/probe_ckpts/dyn_gt_warmup_s0 \\
        --out $AMBIENT_BASE/generated/gradcos_gt_warmup.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                                      # noqa: F401
from torch_utils import persistence                # noqa: F401
from training import probe as P
from training.loss import AmbientEDMLoss

BANDS = [(0.05, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.95)]


def load_dir(paths, device):
    imgs = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        imgs.append((a / 127.5 - 1.0).transpose(2, 0, 1))
    return torch.tensor(np.stack(imgs), device=device)


def kimg_of(path):
    m = re.search(r"network-snapshot-(\d+)\.pkl$", path)
    return int(m.group(1)) if m else -1


def flat_grad(net):
    return torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None])


def band_gradient(net, loss_fn, x0, band, n_draws, batch, gen, device):
    """Accumulated training gradient over `n_draws` noise draws, sigma ~ U(T in band)."""
    net.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(n_draws):
        for i0 in range(0, x0.shape[0], batch):
            xb = x0[i0:i0 + batch]
            t = torch.rand(xb.shape[0], generator=gen, device=device) * (band[1] - band[0]) + band[0]
            sigma_t = torch.tensor(P.t_to_sigma(t.cpu().numpy()), device=device, dtype=xb.dtype)
            sigma_tn = torch.zeros_like(sigma_t)
            loss, *_ = loss_fn(net=net, x_tn=xb, sigma_tn=sigma_tn, sigma_t=sigma_t, labels=None)
            (loss.mean() / n_draws).backward()
            total += float(loss.mean()) / n_draws
    return flat_grad(net), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--probe_dir", default=f"{AMBIENT_BASE}/probe_holdout_64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=128)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.probe_dir, "probe_set.json")) as f:
        files = json.load(f)["files"][:args.n_images]
    clean = load_dir([os.path.join(args.probe_dir, "clean", f) for f in files], device)
    blur = load_dir([os.path.join(args.probe_dir, "blur05", f) for f in files], device)
    loss_fn = AmbientEDMLoss()

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "network-snapshot-*.pkl")), key=kimg_of)
    print(f"{len(ckpts)} checkpoints, {args.n_images} faces/arm, {args.n_draws} draws, bands in T: "
          + " ".join(f"[{a:.2f},{b:.2f})" for a, b in BANDS))
    print(f"{'kimg':>6} | " + " ".join(f"{(a+b)/2:>5.2f}" for a, b in BANDS)
          + "   <- cos(g_clean, g_blur) per band; then cos(clean, clean) ceiling")

    records = []
    for path in ckpts:
        with open(path, "rb") as f:
            net = pickle.load(f)["ema"].to(device)
        net.eval().requires_grad_(True)              # eval: no dropout; grads: yes
        per = []
        half = clean.shape[0] // 2
        for band in BANDS:
            # Three gradients per band. A/B are disjoint halves of the faces.
            #   g_cA vs g_bA : clean vs blurred, SAME faces, SAME noise -> the signal
            #   g_cA vs g_cB : clean vs clean, different faces          -> noise ceiling
            # Noise pulls any cosine toward 0 (a perpendicular component), so
            # the signal must be read relative to the ceiling, not to 1.0.
            def grad(x):
                gen = torch.Generator(device=device); gen.manual_seed(args.seed)
                return band_gradient(net, loss_fn, x, band, args.n_draws, args.batch_size, gen, device)
            g_cA, l_cA = grad(clean[:half])
            g_bA, l_bA = grad(blur[:half])
            g_cB, _ = grad(clean[half:])
            cosine = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
            per.append({"band": list(band), "t_mid": (band[0] + band[1]) / 2,
                        "cos": cosine(g_cA, g_bA),
                        "cos_ceiling": cosine(g_cA, g_cB),
                        "norm_ratio_blur_over_clean": float(g_bA.norm() / (g_cA.norm() + 1e-12)),
                        "loss_clean": l_cA, "loss_blur": l_bA})
        k = kimg_of(path)
        records.append({"checkpoint": os.path.basename(path), "kimg": k, "per_band": per})
        print(f"{k:>6} | " + " ".join(f"{r['cos']:>5.2f}" for r in per)
              + "   ceiling: " + " ".join(f"{r['cos_ceiling']:>5.2f}" for r in per))
        net.requires_grad_(False)
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "bands": BANDS, "records": records}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
