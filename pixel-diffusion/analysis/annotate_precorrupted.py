"""Ambient-o annotation for data that is ALREADY corrupted on disk.

`analysis/annotate.py` is Ambient-o's annotator: it corrupts clean images on
the fly through a noise config, then queries the time-conditional classifier
across a dense sigma grid and stores, per image, the classifier's probability
of "corrupted" at every sigma. `training/training_loop.py` later turns that
trajectory into a per-image sigma_min (first confusion, epsilon 0.05, EMA
window 32). This script is the same pipeline for our setting, where the blurred
images (b5_*) already exist as files and there is nothing to corrupt on the fly:

* same sigma grid (seed 42, sorted samples of exp(1.2 N(0,1) - 1.2)),
* same query (get_classifier_trajectory, num_trials fresh noise draws per sigma),
* same output format (`probabilities` per file + sigmas.txt), so the training
  loop consumes it unchanged and the per-image thresholds are computed by
  Ambient-o's own code, not a reimplementation.

The one deliberate difference is num_sigmas: annotate.py defaults to 2048, which
at 26k images and 4 trials is ~200M forward passes. 128 keeps the grid dense in
log-sigma (~0.05 apart) and makes it a few GPU-hours.

It also prints the summary the experiment is actually after: the distribution of
per-image thresholds converted to T, so the median can be handed straight to a
static run.

Usage:
    python analysis/annotate_precorrupted.py \\
        --checkpoint_path $AMBIENT_BASE/train_outputs/cls_b0b5/network-snapshot-000200.pkl \\
        --dataset_path    $AMBIENT_BASE/annotated_datasets/celeba_dynamic_t_v2_b0b5 \\
        --out             $AMBIENT_BASE/annotated_datasets/celeba_amb_perimage
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib                                      # noqa: F401  (unpickling)
from torch_utils import persistence                # noqa: F401
from ambient_utils.classifier import get_classifier_trajectory, analyze_classifier_trajectory
from training.training_loop import apply_ema, sigma_min_to_t

CLEAN_PREFIX, CORRUPT_PREFIX = "b0_", "b5_"


def load_image(path):
    a = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy((a / 127.5 - 1.0).transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True, help="edmcls network-snapshot pkl")
    ap.add_argument("--dataset_path", required=True, help="dir with b0_* and b5_* images")
    ap.add_argument("--out", required=True, help="annotated dataset dir to create")
    ap.add_argument("--num_sigmas", type=int, default=128)
    ap.add_argument("--num_trials_per_t", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64, help="vmap chunk over sigmas")
    ap.add_argument("--max_images", type=int, default=None, help="annotate a subset (for a quick median)")
    ap.add_argument("--cls_epsilon", type=float, default=0.05, help="train.py default")
    ap.add_argument("--cls_ema_window", type=int, default=32, help="train.py default")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.checkpoint_path, "rb") as f:
        net = pickle.load(f)["ema"].to(device).eval()

    # Same grid as annotate.py, including the seed -- it is the grid the
    # training loop will read back from sigmas.txt.
    rnd = torch.randn([args.num_sigmas, 1, 1, 1], device=device,
                      generator=torch.Generator(device=device).manual_seed(42))
    sigmas, _ = (rnd * 1.2 - 1.2).exp().sort(dim=0)
    sigmas = sigmas.squeeze()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sigmas.txt"), "w") as f:
        for s in sigmas:
            f.write(f"{s.item()}\n")

    files = sorted(f for f in os.listdir(args.dataset_path) if f.endswith(".jpg"))
    corrupt = [f for f in files if f.startswith(CORRUPT_PREFIX)]
    clean = [f for f in files if f.startswith(CLEAN_PREFIX)]
    if args.max_images:
        corrupt = corrupt[:args.max_images]
    print(f"annotating {len(corrupt)} blurred + {len(clean)} clean  "
          f"({args.num_sigmas} sigmas x {args.num_trials_per_t} trials)")

    ann_path = os.path.join(args.out, "annotations.jsonl")
    done = set()
    if os.path.exists(ann_path):                    # resumable
        with open(ann_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["filename"])
                except Exception:
                    pass

    def scheduler(x, s):
        return x + s * torch.randn_like(x)

    def model_fn(x, s):
        return net(x, s, None)["cls_logits"]

    def link(fname):
        p = os.path.join(args.out, fname)
        if not os.path.lexists(p):
            os.symlink(os.path.realpath(os.path.join(args.dataset_path, fname)), p)

    t0 = time.time()
    thresholds = []
    with open(ann_path, "a") as out:
        for fname in clean:                          # clean is clean at every sigma
            link(fname)
            if fname not in done:
                out.write(json.dumps({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0}) + "\n")
        for i, fname in enumerate(corrupt):
            link(fname)
            if fname in done:
                continue
            x = load_image(os.path.join(args.dataset_path, fname)).to(device)
            x = x.unsqueeze(0).repeat(args.num_trials_per_t, 1, 1, 1)
            probs = get_classifier_trajectory(input=x, scheduler=scheduler, model=model_fn,
                                              diffusion_times=sigmas, batch_size=args.batch_size,
                                              model_output_type="logits")     # [n_sigmas, trials]
            out.write(json.dumps({"filename": fname, "probabilities": probs.tolist()}) + "\n")
            # Ambient-o's own reduction, as training_loop applies it at load time.
            p = apply_ema(probs.numpy().mean(axis=-1), window=args.cls_ema_window)
            fc = analyze_classifier_trajectory(torch.tensor(p).to(device), sigmas,
                                               epsilon=args.cls_epsilon)["first_confusion"]
            thresholds.append(float(fc))
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(corrupt)}  {rate:.1f} img/s  eta {(len(corrupt) - i - 1) / rate / 60:.0f} min")
            out.flush()

    if thresholds:
        sig = np.array(thresholds)
        T = np.array([sigma_min_to_t(s) for s in sig])
        q = lambda a, p: float(np.quantile(a, p))
        summary = {
            "n": int(len(sig)),
            "sigma_min": {"median": q(sig, .5), "q10": q(sig, .1), "q90": q(sig, .9)},
            "T": {"median": q(T, .5), "q10": q(T, .1), "q25": q(T, .25),
                  "q75": q(T, .75), "q90": q(T, .9), "mean": float(T.mean())},
            "cls_epsilon": args.cls_epsilon, "cls_ema_window": args.cls_ema_window,
            "num_sigmas": args.num_sigmas, "num_trials_per_t": args.num_trials_per_t,
            "checkpoint": args.checkpoint_path,
        }
        with open(os.path.join(args.out, "threshold_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nper-image threshold the classifier assigns to the blur bucket, in T:")
        print(f"  median {summary['T']['median']:.3f}   q10 {summary['T']['q10']:.3f}   "
              f"q90 {summary['T']['q90']:.3f}   (n={len(sig)})")
        print(f"  MIND-optimal static T from the sweep: 0.50")
        print(f"wrote {os.path.join(args.out, 'threshold_summary.json')}")


if __name__ == "__main__":
    main()
