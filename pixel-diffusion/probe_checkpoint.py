"""Run the T-probe against saved network snapshots, offline.

Same code path as the online probe in training/probe.py, but driven from
checkpoints on disk instead of from inside a training loop. Two uses:

1. Calibration without spending a training run. A finished run's snapshots are a
   free sample of "the model at various points in training", so the per-sigma
   divergence curves, and what every threshold rule would do with them, can be
   had in minutes rather than the ~9 h a probe-only training run costs.
   run_dyn_job.sh deletes intermediate snapshots on completion to reclaim disk,
   so they have to be copied out of a live run first -- see
   $AMBIENT_BASE/probe_ckpts.

2. Re-probing a finished principled run at a different resolution than it logged.

Usage:
    python probe_checkpoint.py --ckpt_dir $AMBIENT_BASE/probe_ckpts/dyn_p1_q_sobol08_s0 \
        --out $AMBIENT_BASE/generated/probe_calib_sobol08.json
    python probe_checkpoint.py --ckpt a.pkl --ckpt b.pkl --out out.json
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
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnnlib                                        # noqa: F401  (unpickling needs it)
from torch_utils import persistence                  # noqa: F401
from training import probe as probe_mod


def kimg_of(path):
    m = re.search(r"network-snapshot-(\d+)\.pkl$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_arms(root, n_images, device):
    from PIL import Image
    meta = os.path.join(root, "probe_set.json")
    if not os.path.exists(meta):
        raise SystemExit(f"probe set missing at {root}\n"
                         "  build it: python dataset_creation/create_probe_holdout.py")
    with open(meta) as f:
        files = json.load(f)["files"][:n_images]
    out = []
    for arm in ("clean", "blur05"):
        imgs = []
        for fname in files:
            a = np.array(Image.open(os.path.join(root, arm, fname)).convert("RGB"),
                         dtype=np.float32)
            imgs.append((a / 127.5 - 1.0).transpose(2, 0, 1))
        out.append(torch.tensor(np.stack(imgs), device=device))
    return out[0], out[1], files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="repeatable")
    ap.add_argument("--ckpt_dir", default=None, help="probe every snapshot in this dir")
    ap.add_argument("--probe_dir", default=f"{AMBIENT_BASE}/probe_holdout_64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_images", type=int, default=400)
    ap.add_argument("--n_draws", type=int, default=2)
    ap.add_argument("--n_levels", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--probe_seed", type=int, default=12345)
    ap.add_argument("--total_kimg", type=int, default=2000)
    ap.add_argument("--model", choices=["ema", "net"], default="ema",
                    help="ema is what generation and MIND use, so it is the default")
    args = ap.parse_args()

    ckpts = list(args.ckpt)
    if args.ckpt_dir:
        ckpts += glob.glob(os.path.join(args.ckpt_dir, "network-snapshot-*.pkl"))
    ckpts = sorted(set(ckpts), key=kimg_of)
    if not ckpts:
        raise SystemExit("no checkpoints given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean, corrupt, files = load_arms(args.probe_dir, args.n_images, device)
    t_grid = probe_mod.make_t_grid(args.n_levels)
    print(f"device={device}  arms={tuple(clean.shape)}  levels={args.n_levels}  "
          f"draws={args.n_draws}")
    print(f"T grid: {t_grid[0]:.3f} .. {t_grid[-1]:.3f}   "
          f"sigma {probe_mod.t_to_sigma(t_grid)[0]:.4f} .. "
          f"{probe_mod.t_to_sigma(t_grid)[-1]:.2f}")

    records = []
    for path in ckpts:
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = data[args.model].to(device).eval()
        t0 = time.time()
        res = probe_mod.run_probe(net, clean, corrupt, t_grid=t_grid,
                                  n_draws=args.n_draws, batch_size=args.batch_size,
                                  probe_seed=args.probe_seed)
        kimg = kimg_of(path)
        rec = {"checkpoint": os.path.basename(path), "kimg": kimg,
               "progress": kimg / args.total_kimg if args.total_kimg else 0.0,
               "model": args.model,
               "counterfactual_T": probe_mod.all_decisions(res),
               "probe": res}
        records.append(rec)

        ps = res["per_sigma"]
        print(f"\n{os.path.basename(path)}  ({time.time() - t0:.1f}s)")
        print(f"  {'T':>6} {'sigma':>8} {'mse_clean':>10} {'mse_corr':>10} "
              f"{'ratio':>7} {'gap_t':>8} {'pv_ratio':>8}")
        for r in ps:
            print(f"  {r['t']:>6.3f} {r['sigma']:>8.4f} {r['mse_clean_mean']:>10.5f} "
                  f"{r['mse_corrupt_mean']:>10.5f} {r['loss_ratio']:>7.4f} "
                  f"{r['gap_t']:>8.2f} {r.get('predvar_ratio', float('nan')):>8.4f}")
        for key in sorted(rec["counterfactual_T"]):
            if key.endswith("__error"):
                continue
            print(f"    T[{key:<28}] = {rec['counterfactual_T'][key]}")
        del net, data
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "n_probe_files": len(files),
                   "records": records}, f, indent=2)
    print(f"\nwrote {args.out}  ({len(records)} checkpoints)")


if __name__ == "__main__":
    main()
