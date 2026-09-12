"""Offline check of the data-starvation readout on saved checkpoints.

Starvation at a noise level = she has memorised her 500 clean training faces
there rather than learned the level. Readout: per noise level,

    gap(sigma) = error on held-out clean faces  -  error on training clean faces

Both arms are clean, so the pixel-energy nuisance that broke the blurry-vs-sharp
error comparison does not arise. Wobble is reported for both arms too, from the
same forward passes, as a cross-check.

The question this answers BEFORE any training run is spent: how does the gap
behave over training progress? The good schedule is permissive early and
restrictive late. If the gap is ~0 early (nothing memorised yet) and large
late, a rule of the form "open blurry data where the gap is large" would give
the *inverted* schedule. This script exists to find that out on rescued
checkpoints in minutes rather than in a 9-hour run.

Usage:
    python analysis/probe_gap_checkpoint.py \\
        --ckpt_dir $AMBIENT_BASE/probe_ckpts/dyn_gt_warmup_s0 \\
        --train_dir $AMBIENT_BASE/annotated_datasets/celeba_dynamic_t_v2 \\
        --out $AMBIENT_BASE/generated/gap_probe_gt_warmup.json
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


def load_dir(paths, device):
    imgs = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        imgs.append((a / 127.5 - 1.0).transpose(2, 0, 1))
    return torch.tensor(np.stack(imgs), device=device)


def kimg_of(path):
    m = re.search(r"network-snapshot-(\d+)\.pkl$", path)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--train_dir", required=True, help="dir holding the b0_* training clean faces")
    ap.add_argument("--holdout_dir", default=f"{AMBIENT_BASE}/probe_holdout_64/clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=160)
    ap.add_argument("--n_draws", type=int, default=2)
    ap.add_argument("--n_levels", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=80)
    ap.add_argument("--probe_seed", type=int, default=12345)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_paths = sorted(glob.glob(os.path.join(args.train_dir, "b0_*.jpg")))[:args.n_images]
    hold_paths = sorted(glob.glob(os.path.join(args.holdout_dir, "*.jpg")))[:args.n_images]
    if len(train_paths) < args.n_images or len(hold_paths) < args.n_images:
        raise SystemExit(f"need {args.n_images} per arm, have train={len(train_paths)} holdout={len(hold_paths)}")
    train = load_dir(train_paths, device)
    hold = load_dir(hold_paths, device)
    t_grid = P.make_t_grid(args.n_levels)
    sigmas = P.t_to_sigma(t_grid)

    # Common random numbers, as in the online probe.
    gen = torch.Generator(device=device); gen.manual_seed(args.probe_seed)
    noise = torch.randn((args.n_images, args.n_draws) + tuple(train.shape[1:]),
                        generator=gen, device=device, dtype=train.dtype)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "network-snapshot-*.pkl")), key=kimg_of)
    records = []
    print(f"{len(ckpts)} checkpoints, {args.n_images} faces/arm, {args.n_levels} levels")
    print(f"{'kimg':>6} | " + " ".join(f"{t:>5.2f}" for t in t_grid[::3]) + "   <- relative gap (held-out - train)/held-out")
    for path in ckpts:
        with open(path, "rb") as f:
            net = pickle.load(f)["ema"].to(device).eval()
        per = []
        for t_val, sig in zip(t_grid, sigmas):
            mse_tr, pv_tr = P._arm(net, train, noise, sig, args.batch_size)
            mse_ho, pv_ho = P._arm(net, hold, noise, sig, args.batch_size)
            gap = mse_ho - mse_tr
            per.append({
                "t": float(t_val), "sigma": float(sig),
                "mse_train": float(mse_tr.mean()), "mse_holdout": float(mse_ho.mean()),
                "gap": float(gap.mean()),
                "gap_se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
                "rel_gap": float(gap.mean() / mse_ho.mean()) if mse_ho.mean() > 0 else float("nan"),
                "predvar_train": float(pv_tr.mean()), "predvar_holdout": float(pv_ho.mean()),
                "predvar_ratio_holdout_over_train": float(pv_ho.mean() / pv_tr.mean()) if pv_tr.mean() > 0 else float("nan"),
            })
        k = kimg_of(path)
        records.append({"checkpoint": os.path.basename(path), "kimg": k, "per_sigma": per})
        print(f"{k:>6} | " + " ".join(f"{r['rel_gap']:>5.2f}" for r in per[::3]))
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "t_grid": [float(x) for x in t_grid], "records": records}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
