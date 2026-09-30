#!/usr/bin/env python3
"""Per-image features for the learned threshold policy, computed once before training.

  a  Ambient-o's verdict: the classifier's first-confusion time in T units (0..1), reduced from
     the stored probabilities exactly as training does (EMA window 1 on the 64-point grid).
  h  sharpness: log variance of the Laplacian of the grayscale image, standardised against the
     clean images (z-score), divided by 3 so its range is O(1). Needs no labels or classifier.

Written BESIDE the dataset folder (the loader scans the folder):
  annotated_datasets/<dataset>.policy_features.json   {filename: [a, h]}  (corrupted images only)

Usage (CSAIL):  python policy/make_image_features.py --dataset celeba_mix4_v1 --ambo celeba_mix4_ambo
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)
import argparse, json
import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import norm


def sharpness(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return float(np.log(laplace(g).var() + 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="training dataset (corrupt files carry the 999 sentinel)")
    ap.add_argument("--ambo", required=True, help="the classifier-annotated copy (probabilities per file)")
    args = ap.parse_args()
    D = f"{AMBIENT_BASE}/annotated_datasets"
    import torch
    from ambient_utils.classifier import analyze_classifier_trajectory
    sig = torch.tensor([float(l) for l in open(f"{D}/{args.ambo}/sigmas.txt")]).sort()[0]
    to_T = lambda s: 0.0 if s <= 0 else float(norm.cdf((np.log(s) + 1.2) / 1.2))
    a = {}
    for line in open(f"{D}/{args.ambo}/annotations.jsonl"):
        d = json.loads(line)
        if "probabilities" in d:
            p = torch.tensor(d["probabilities"], dtype=torch.float64).mean(-1)
            a[d["filename"]] = to_T(float(analyze_classifier_trajectory(p, sig, epsilon=0.05)["first_confusion"]))
    clean, corrupt = [], []
    for line in open(f"{D}/{args.dataset}/annotations.jsonl"):
        d = json.loads(line)
        (corrupt if d["sigma_min"] >= 900 else clean).append(d["filename"])
    hc = np.array([sharpness(f"{D}/{args.dataset}/{f}") for f in clean])
    mu, sd = hc.mean(), hc.std()
    out, miss = {}, 0
    for f in corrupt:
        if f not in a:
            miss += 1
            continue
        out[f] = [round(a[f], 5), round((sharpness(f"{D}/{args.dataset}/{f}") - mu) / sd / 3.0, 5)]
    if miss:
        raise SystemExit(f"{miss} corrupt files have no classifier verdict")
    path = f"{D}/{args.dataset}.policy_features.json"
    json.dump(out, open(path, "w"))
    by = {}
    for f, (av, hv) in out.items():
        by.setdefault(f.split("_")[0], []).append((av, hv))
    for g in sorted(by):
        x = np.array(by[g]); print(f"{g}: n={len(x)} a median {np.median(x[:, 0]):.3f}  h median {np.median(x[:, 1]):+.3f}")
    print(f"clean sharpness mean {mu:.3f} sd {sd:.3f}; wrote {path}")


if __name__ == "__main__":
    main()
