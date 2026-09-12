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

DEFAULT_CLEAN_PREFIX, DEFAULT_CORRUPT_PREFIX = "b0_", "b5_"


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
    ap.add_argument("--cls_ema_window", type=int, default=None,
                    help="EMA window over the sigma grid. train.py's default of 32 is for annotate.py's "
                         "2048-point grid; if unset it is scaled to the grid here (32 * num_sigmas / 2048, "
                         "min 1), otherwise a 64-point grid gets smoothed over half its length and a "
                         "confident classifier can never fall below the confusion threshold before the grid ends.")
    ap.add_argument("--clean_prefix", default=DEFAULT_CLEAN_PREFIX)
    ap.add_argument("--corrupt_prefix", default=DEFAULT_CORRUPT_PREFIX,
                    help="e.g. bX_ for a directory of held-out faces blurred at a control strength")
    args = ap.parse_args()
    CLEAN_PREFIX, CORRUPT_PREFIX = args.clean_prefix, args.corrupt_prefix
    if args.cls_ema_window is None:
        args.cls_ema_window = max(1, round(32 * args.num_sigmas / 2048))
        print(f"EMA window scaled to the {args.num_sigmas}-point grid: {args.cls_ema_window}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build the network from training_options.json and copy the EMA weights in,
    # exactly as annotate.py does. Unpickling the snapshot directly re-executes
    # the network's source through torch_utils.persistence, which fails outside
    # the training process for the classifier network; constructing by name and
    # copying parameters sidesteps that and is the path Ambient-o itself uses.
    from torch_utils.misc import copy_params_and_buffers
    # The snapshot embeds ambient_networks.py's source, which begins
    # `from .networks import ...`. persistence re-executes that source as a
    # standalone module, where a relative import has no parent package and
    # raises ImportError. persistence provides import_hook for precisely this:
    # rewrite the embedded source before it is exec'd.
    def _fix_relative_import(meta):
        meta.module_src = meta.module_src.replace("from .networks import",
                                                  "from training.networks import")
        return meta
    persistence.import_hook(_fix_relative_import)
    opts_path = os.path.join(os.path.dirname(args.checkpoint_path), "training_options.json")
    with open(opts_path) as f:
        options = json.load(f)
    interface_kwargs = dict(img_resolution=options["dataset_kwargs"]["resolution"],
                            img_channels=3, label_dim=0)
    net = dnnlib.util.construct_class_by_name(**options["network_kwargs"], **interface_kwargs)
    with open(args.checkpoint_path, "rb") as f:
        data = pickle.load(f)
    copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
    del data
    net = net.to(device).eval()

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

    files = sorted(f for f in os.listdir(args.dataset_path) if f.endswith((".jpg", ".png")))
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
