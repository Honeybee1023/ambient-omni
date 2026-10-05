#!/usr/bin/env python3
"""Label-free groups: 1-D k-means of Ambient-o's per-image thresholds -> a k-grouped copy of a mix4 set.

This is the procedure that built celeba_mix4_km (2026-10-03, run by hand at the time; written down
here so the AFHQ-dog copy is made the same way):

  * T per blurred image from the ambo annotations: mean over trials of the classifier's
    P(corrupted) at each sigma of sigmas.txt (EMA window 1 = no smoothing), first sigma where it
    drops below 0.5 + eps (eps 0.05) -- "first confusion"; if it never does, the last sigma.
    T = Phi((ln sigma + 1.2) / 1.2)   (training_loop.sigma_min_to_t).
  * 1-D k-means, k = 4, Lloyd, centers initialised at the quantiles (j + 0.5) / k, iterated to
    convergence; clusters renumbered by center so k1 is the mildest (lowest T).
  * The new dataset copies every file of the source set: clean b0_* unchanged, each blurred file
    renamed k{c}_<original name> (e.g. k2_g05_xxx.png). Annotations are the source's, renamed, so
    the blurred files keep the sigma_min = 999 sentinel and the per_group schedule sets thresholds.
  * km_assignment.json inside the new folder: what / centers_T / sizes / per-file {T, cluster}.
    (It is JSON, not an image, so the loader ignores it -- same place as in celeba_mix4_km.)

Also cross-checks T against threshold_summary.json written by analysis/annotate_precorrupted.py
(which computes it with Ambient-o's own analyze_classifier_trajectory): the medians must agree.

Usage:  python dataset_creation/create_km_dataset.py --src afhqdog_mix4_v1 --ambo afhqdog_mix4_ambo \
            --name afhqdog_mix4_km
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import argparse, json, os, shutil, sys

import numpy as np
from scipy.stats import norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="mix4 dataset (b0_* + g*_* files, sentinel annotations)")
    ap.add_argument("--ambo", required=True, help="annotate_precorrupted.py output for --src")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    D = f"{AMBIENT_BASE}/annotated_datasets"
    src, ambo, dst = f"{D}/{args.src}", f"{D}/{args.ambo}", f"{D}/{args.name}"
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists; refusing to overwrite")

    sig = np.loadtxt(f"{ambo}/sigmas.txt")
    T = {}
    for l in open(f"{ambo}/annotations.jsonl"):
        r = json.loads(l)
        if "probabilities" not in r:
            continue
        p = np.array(r["probabilities"]).mean(1)
        idx = np.nonzero(0.5 - p + args.eps > 0)[0]
        i = idx[0] if len(idx) else len(sig) - 1
        T[r["filename"]] = float(norm.cdf((np.log(sig[i]) + 1.2) / 1.2))
    src_ann = [json.loads(l) for l in open(f"{src}/annotations.jsonl")]
    blurred = sorted(a["filename"] for a in src_ann if not a["filename"].startswith("b0_"))
    if sorted(T) != blurred:
        sys.exit(f"ambo annotates {len(T)} files, source has {len(blurred)} blurred; refusing")
    summ = f"{ambo}/threshold_summary.json"
    if os.path.exists(summ):
        med = json.load(open(summ))["T"]["median"]
        x_med = float(np.quantile(list(T.values()), 0.5))
        print(f"median T here {x_med:.4f} vs annotate_precorrupted {med:.4f}")
        if abs(x_med - med) > 1e-3:
            sys.exit("T reconstruction disagrees with Ambient-o's own reduction; refusing")

    names = sorted(T)
    x = np.array([T[n] for n in names])
    k = args.k
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(1000):
        a = np.argmin(abs(x[:, None] - c[None]), 1)
        c2 = np.array([x[a == j].mean() for j in range(k)])
        if np.allclose(c, c2):
            break
        c = c2
    o = np.argsort(c)
    rank = np.empty(k, int)
    rank[o] = np.arange(k)
    lab = rank[a] + 1
    c = c[o]
    m = {n: int(j) for n, j in zip(names, lab)}

    os.makedirs(dst)
    out = []
    for r in src_ann:
        f = r["filename"]
        nf = f if f.startswith("b0_") else f"k{m[f]}_{f}"
        shutil.copy(f"{src}/{f}", f"{dst}/{nf}")
        out.append(dict(r, filename=nf))
    with open(f"{dst}/annotations.jsonl", "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in out))
    sizes = [int((lab == j).sum()) for j in range(1, k + 1)]
    json.dump({"what": f"1-D k-means (k={k}, Lloyd, init at quantiles) on Ambient-o per-image T from {args.ambo} "
                       f"(eps {args.eps}, ema window 1, {len(sig)}-sigma grid); clusters ordered by center, k1 = mildest",
               "centers_T": [round(float(v), 4) for v in c], "sizes": sizes,
               "assignment": {n: {"T": round(T[n], 4), "cluster": m[n]} for n in names}},
              open(f"{dst}/km_assignment.json", "w"))
    # Cross-tab of k-means cluster against the true blur level (what the label-free grouping recovers).
    levels = sorted({n.split("_", 1)[0] for n in names})
    print("centers", np.round(c, 3), "sizes", sizes)
    print("cluster x true level:", {f"k{j}": {g: sum(1 for n in names if m[n] == j and n.startswith(g + "_"))
                                              for g in levels} for j in range(1, k + 1)})
    n_files = len([f for f in os.listdir(dst) if not f.endswith((".jsonl", ".json"))])
    assert n_files == len(out), (n_files, len(out))
    print(f"wrote {dst}: {n_files} images, km_assignment.json")


if __name__ == "__main__":
    main()
