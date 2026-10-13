"""Held-out denoising loss as a function of noise level T, for many checkpoints in one pass.

For each checkpoint (EMA denoiser) and each T on a fixed grid (T = Phi((log sigma + 1.2) / 1.2), the schedules'
convention), the MSE of the denoised estimate on clean held-out images, with the same (image, noise) pairs for
every checkpoint and every T. Reported raw (per-pixel MSE in [-1, 1] units) and EDM-weighted. Plotting it for
several training times shows WHERE in the noise range a schedule gains or loses against another, and when.

Usage:
    python analysis/eval_holdout_loss_curve.py --holdout_dir D --out out.json --ckpt run_a:a.pkl --ckpt run_b:b.pkl
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse, glob, json, os, pickle, re, sys
import numpy as np, torch
from PIL import Image
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                       # noqa: F401
from torch_utils import persistence # noqa: F401

P_MEAN, P_STD, SIGMA_DATA = -1.2, 1.2, 0.5
T_GRID = [round(t, 3) for t in np.linspace(0.02, 0.98, 25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="<run>:<path to network-snapshot pkl>")
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2, help="noise draws per (image, T)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=250)
    a = ap.parse_args()
    dev = "cuda"
    files = sorted(glob.glob(os.path.join(a.holdout_dir, "*.png")) + glob.glob(os.path.join(a.holdout_dir, "*.jpg")))
    x0 = torch.from_numpy(np.stack([np.array(Image.open(f).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1
                                    for f in files])).to(dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    noise = [torch.randn(x0.shape, device=dev, generator=g) for _ in range(a.draws)]   # shared by all checkpoints
    sig = [float(np.exp(P_MEAN + P_STD * norm.ppf(t))) for t in T_GRID]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"T": T_GRID, "sigma": sig, "n_images": len(files),
                                                                "holdout_dir": a.holdout_dir, "runs": {}}
    for spec in a.ckpt:
        run, path = spec.split(":", 1)
        kimg = int(re.search(r"snapshot-(\d+)", path).group(1))
        if str(kimg) in res["runs"].get(run, {}):
            continue
        with open(path, "rb") as f:
            net = pickle.load(f)["ema"].to(dev).eval()
        mse = []
        with torch.no_grad():
            for s in sig:
                tot = 0.0
                for n in noise:
                    for i in range(0, len(files), a.batch_size):
                        xb = x0[i:i + a.batch_size]
                        sb = torch.full((len(xb), 1, 1, 1), s, device=dev)
                        pred = net(xb + s * n[i:i + a.batch_size], sb, None)
                        tot += ((pred - xb) ** 2).mean(dim=(1, 2, 3)).sum().item()
                mse.append(tot / (len(files) * a.draws))
        w = [(SIGMA_DATA ** 2 + s ** 2) / (s ** 2 * SIGMA_DATA ** 2) for s in sig]
        res["runs"].setdefault(run, {})[str(kimg)] = {"mse": mse, "weighted": [m * wi for m, wi in zip(mse, w)]}
        json.dump(res, open(a.out, "w"))
        print(run, kimg, "mean weighted loss", round(float(np.mean(res["runs"][run][str(kimg)]["weighted"])), 4), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
