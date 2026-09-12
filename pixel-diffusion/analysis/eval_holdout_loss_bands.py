"""Held-out denoising loss of a final checkpoint, overall and by noise band.

Same quantity as eval_val_loss.py (EDM-weighted MSE of the EMA denoiser on
clean held-out faces at sigmas drawn from the training distribution), plus a
breakdown by noise band so a schedule that only changes the low-noise
behaviour is visible. Fixed seed, fixed images, several sigma draws per image,
so every checkpoint is scored on identical (image, sigma, noise) triples.

Usage:
    python analysis/eval_holdout_loss_bands.py --checkpoint X.pkl --holdout_dir D --out out.json
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse, glob, json, os, pickle, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                       # noqa: F401
from torch_utils import persistence # noqa: F401

P_MEAN, P_STD, SIGMA_DATA = -1.2, 1.2, 0.5
# T bands in the same convention as the schedules: T = Phi((log sigma + 1.2)/1.2)
BANDS = {"low_T<0.33": (0.0, 1/3), "mid_T.33-.67": (1/3, 2/3), "high_T>0.67": (2/3, 1.0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout_dir", default=f"{AMBIENT_BASE}/probe_holdout_64/clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=4, help="sigma/noise draws per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.checkpoint, "rb") as f:
        net = pickle.load(f)["ema"].to(dev).eval()
    files = sorted(glob.glob(os.path.join(a.holdout_dir, "*.jpg")) + glob.glob(os.path.join(a.holdout_dir, "*.png")))
    imgs = np.stack([np.array(Image.open(f).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for f in files])
    x0_all = torch.from_numpy(imgs).to(dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    N = len(files)
    from scipy.stats import norm
    w_all, mse_all, T_all = [], [], []
    with torch.no_grad():
        for d in range(a.draws):
            logs = torch.randn(N, device=dev, generator=g) * P_STD + P_MEAN
            sig = logs.exp()
            noise = torch.randn(x0_all.shape, device=dev, generator=g)
            for i in range(0, N, a.batch_size):
                x0 = x0_all[i:i+a.batch_size]; s = sig[i:i+a.batch_size]
                xt = x0 + s[:, None, None, None] * noise[i:i+a.batch_size]
                pred = net(xt, s[:, None, None, None], None)
                mse = ((pred - x0) ** 2).mean(dim=(1, 2, 3))
                w = (SIGMA_DATA**2 + s**2) / (s**2 * SIGMA_DATA**2)
                w_all.append((w * mse).cpu().numpy()); mse_all.append(mse.cpu().numpy())
                T_all.append(norm.cdf(((s.log() + 1.2) / 1.2).cpu().numpy()))
    w_all, mse_all, T_all = map(np.concatenate, (w_all, mse_all, T_all))
    res = {"checkpoint": a.checkpoint, "n_images": N, "draws": a.draws, "seed": a.seed,
           "weighted_loss_mean": float(w_all.mean()),
           "weighted_loss_sem": float(w_all.std() / np.sqrt(len(w_all))),
           "unweighted_mse_mean": float(mse_all.mean()), "bands": {}}
    for k, (lo, hi) in BANDS.items():
        m = (T_all >= lo) & (T_all < hi)
        res["bands"][k] = {"n": int(m.sum()), "weighted_loss_mean": float(w_all[m].mean()),
                           "unweighted_mse_mean": float(mse_all[m].mean())}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
