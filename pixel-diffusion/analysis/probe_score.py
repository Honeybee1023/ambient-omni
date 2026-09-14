"""Score a checkpoint with the online probe's metrics (for look-ahead arms).

Prints one JSON line: {"soft_hi": mean soft_clean over levels t>=0.6,
"mem_gap_lo": mean mem_gap over levels t<=0.35, "score": soft_hi - w*mem_gap_lo}.
Higher score = more true fine detail at high noise and less memorisation at
low noise. Same probe set, seed and draws as the training-time probe.
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar") if _os.path.isdir(_p)), "/data/scratch/honjar")
import argparse, glob, json, os, pickle, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                       # noqa
from torch_utils import persistence # noqa
from training.probe import run_probe, make_t_grid

def load(files):
    return torch.tensor(np.stack([(np.array(Image.open(f).convert("RGB"), np.float32) / 127.5 - 1).transpose(2, 0, 1) for f in files]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--train_dir", required=True)
    ap.add_argument("--probe_dir", default=f"{AMBIENT_BASE}/probe_holdout_64")
    ap.add_argument("--n_images", type=int, default=160); ap.add_argument("--w", type=float, default=1.0)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = pickle.load(open(a.checkpoint, "rb"))["ema"].to(dev).eval()
    files = json.load(open(os.path.join(a.probe_dir, "probe_set.json")))["files"][:a.n_images]
    clean = load([os.path.join(a.probe_dir, "clean", f) for f in files]).to(dev)
    blur = load([os.path.join(a.probe_dir, "blur05", f) for f in files]).to(dev)
    train = load(sorted(glob.glob(os.path.join(a.train_dir, "b0_*")))[:a.n_images]).to(dev)
    r = run_probe(net, clean, blur, t_grid=make_t_grid(20), n_draws=2, batch_size=80, probe_seed=12345, train_imgs=train)
    ps = r["per_sigma"]
    soft_hi = float(np.mean([p["soft_clean"] for p in ps if p["t"] >= 0.6]))
    gap_lo = float(np.mean([p["mem_gap"] for p in ps if p["t"] <= 0.35]))
    print(json.dumps({"soft_hi": soft_hi, "mem_gap_lo": gap_lo, "score": soft_hi - a.w * gap_lo}))

if __name__ == "__main__":
    main()
