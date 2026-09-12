"""Split a checkpoint's held-out denoising error into the part blur removes and the part it keeps.

The blur bucket is Gaussian blur at sigma_b = 0.5 px. Let B be that blur. For a
clean target x and the model's denoised output y at noise level sigma:
    coarse error  = ||B y - B x||^2       (content blur keeps; dominates plain MSE)
    fine error    = ||(I-B) y - (I-B) x||^2   (content blur removes; low energy)
    softness      = ||(I-B) y||^2 / ||(I-B) x||^2   (< 1: the model's output is smoother
                                                     than the truth, i.e. it learned to blur)
Reported per noise band and overall, on held-out clean faces and (optionally) the
500 training clean faces, with the same (image, sigma, noise) draws for every
checkpoint. Also emits the plain EDM-weighted loss so the two can be compared.
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar") if _os.path.isdir(_p)), "/data/scratch/honjar")
import argparse, glob, json, os, pickle, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                       # noqa
from torch_utils import persistence # noqa
P_MEAN, P_STD, SIGMA_DATA = -1.2, 1.2, 0.5
T_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

def gauss_kernel(sig, r=3):
    x = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-x**2 / (2 * sig**2)); k /= k.sum()
    return k
def blur(x, k):
    C = x.shape[1]; k1 = k.to(x.device)
    kx = k1.view(1, 1, 1, -1).repeat(C, 1, 1, 1); ky = k1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.pad(x, (3, 3, 0, 0), mode="reflect"); x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, 3, 3), mode="reflect"); return F.conv2d(x, ky, groups=C)
def load_dir(d, n=None):
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))[:n]
    return torch.from_numpy(np.stack([np.array(Image.open(f).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for f in fs]))

def score(net, x0_all, k, draws, seed, dev, bs=128):
    g = torch.Generator(device=dev).manual_seed(seed); N = len(x0_all); out = {}
    sig_grid = [float(np.exp(P_STD * norm.ppf(t) + P_MEAN)) for t in T_LEVELS]
    with torch.no_grad():
        for T, s in zip(T_LEVELS, sig_grid):
            acc = dict(coarse=0., fine=0., hp_y=0., hp_x=0., mse=0., w=0., n=0)
            for d in range(draws):
                noise = torch.randn(x0_all.shape, device=dev, generator=g)
                for i in range(0, N, bs):
                    x0 = x0_all[i:i+bs]; xt = x0 + s * noise[i:i+bs]
                    y = net(xt, torch.full((len(x0), 1, 1, 1), s, device=dev), None)
                    bx, by = blur(x0, k), blur(y, k); hx, hy = x0 - bx, y - by
                    acc["coarse"] += ((by - bx)**2).mean(dim=(1,2,3)).sum().item()
                    acc["fine"] += ((hy - hx)**2).mean(dim=(1,2,3)).sum().item()
                    acc["hp_y"] += (hy**2).mean(dim=(1,2,3)).sum().item(); acc["hp_x"] += (hx**2).mean(dim=(1,2,3)).sum().item()
                    m = ((y - x0)**2).mean(dim=(1,2,3)); acc["mse"] += m.sum().item()
                    acc["w"] += (m * (SIGMA_DATA**2 + s**2) / (s**2 * SIGMA_DATA**2)).sum().item(); acc["n"] += len(x0)
            n = acc["n"]
            out[f"T{T}"] = dict(sigma=s, coarse=acc["coarse"]/n, fine=acc["fine"]/n, softness=acc["hp_y"]/acc["hp_x"],
                               mse=acc["mse"]/n, weighted=acc["w"]/n)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_dir", default=f"{AMBIENT_BASE}/probe_holdout_64/clean")
    ap.add_argument("--train_dir", default=None, help="the 500 training clean faces (b0_*), for a memorisation view")
    ap.add_argument("--draws", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=1024)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = pickle.load(open(a.checkpoint, "rb"))["ema"].to(dev).eval(); k = gauss_kernel(0.5)
    res = {"checkpoint": a.checkpoint, "T_levels": T_LEVELS, "blur_sigma_px": 0.5}
    res["holdout"] = score(net, load_dir(a.holdout_dir, a.max_images).to(dev), k, a.draws, a.seed, dev)
    if a.train_dir:
        fs = sorted(glob.glob(os.path.join(a.train_dir, "b0_*")))[:a.max_images]
        x = torch.from_numpy(np.stack([np.array(Image.open(f).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for f in fs])).to(dev)
        res["train_clean"] = score(net, x, k, a.draws, a.seed, dev)
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({t: {kk: round(v, 5) for kk, v in r.items()} for t, r in res["holdout"].items()}))

if __name__ == "__main__":
    main()
