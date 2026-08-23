#!/usr/bin/env python3
"""Create the restricted-bucket ("single bottleneck") validation sweeps.

In the conditional sweeps a bucket with threshold T was eligible for every noise
level in [sigma(T), inf), so consecutive buckets overlapped almost completely and
the second one was free to be redundant. Here each bucket instead owns an
exclusive band: the fixed bucket is capped at the swept bucket's threshold, and
the swept bucket takes everything above it. Removing one now leaves a real hole.

    clean (b0)     sigma in [0, inf)          -- always eligible
    fixed  bucket  sigma in [sigma(T_fix), sigma(T_swp))   <- newly bounded above
    swept  bucket  sigma in [sigma(T_swp), inf)
    all others     inactive (T = 0.999)

The band is carried by a third annotation field, `sigma_band_max`, which
InfiniteSampler ANDs onto the existing gate. Datasets without the field are
untouched -- see tests/test_restricted_bucket_sampler.py.

Two points of every sweep are degenerate on purpose, and both reproduce a
configuration we have already measured:

    T_swept == T_fixed  ->  fixed band is empty  ->  swept bucket solo at T_fixed
    T_swept == 1.0      ->  cap moves to ~inf    ->  fixed bucket solo at T_fixed

They are the in-batch baselines. Every comparison should be made against them
rather than against the older JSON: B2-only@0.55 was recorded twice before, as
0.031144 and 0.032623, and that 0.0015 spread is the size of the whole effect.
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar"
)

import os, json, glob
import numpy as np
from scipy.stats import norm

PROCESSED_DIR = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
DATASET_DIR = f"{AMBIENT_BASE}/annotated_datasets"
TVEC_DIR = f"{AMBIENT_BASE}/generated"

P_MEAN, P_STD = -1.2, 1.2
ALL_BLUR_BUCKETS = [1, 2, 3, 4, 5, 6, 7]
INACTIVE_T = 0.999
# Stands in for +inf: JSON has no infinity literal, and the sampler only ever
# compares against sigmas drawn from exp(1.2*N(0,1) - 1.2).
UNBOUNDED = 1e6

# Thresholds the fixed bucket is pinned at, read off the solo sweeps in
# v2b_all_results.json (argmin over MIND):
#   bucket 1 -> T=0.50 (0.030599)    bucket 2 -> T=0.55 (0.031144)
B1_FIXED_T = 0.50
B2_FIXED_T = 0.55

# Same shape as the conditional sweeps so the curves can be compared point by
# point. Only T >= T_fixed is meaningful: below it the band would be inverted.
SWEEP_TS_FROM_055 = [0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 1.0]
SWEEP_TS_FROM_050 = [0.50, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 1.0]

# Label is the dataset-name stem. The first three predate the later additions and
# keep their bare names; anything new spells out both buckets ("b5g2" = sweep b5
# given b2 fixed), because the swept bucket alone is no longer unique -- b3 is
# swept against both b2 and b1.
SWEEPS = [
    # (label, swept bucket, fixed bucket, T of fixed bucket, swept Ts)
    ("b3",   3, 2, B2_FIXED_T, SWEEP_TS_FROM_055),
    ("b4",   4, 2, B2_FIXED_T, SWEEP_TS_FROM_055),
    ("b2",   2, 1, B1_FIXED_T, SWEEP_TS_FROM_050),
    ("b5g2", 5, 2, B2_FIXED_T, SWEEP_TS_FROM_055),
    ("b3g1", 3, 1, B1_FIXED_T, SWEEP_TS_FROM_050),
]

# T=1.00 collapses every sweep with the same fixed bucket onto one configuration
# (the swept bucket goes inactive, leaving only the fixed bucket banded to ~inf),
# so those points are already measured and must not be retrained:
#   b5g2 T=1.00 == restr_b3_T100   (b2 alone @0.55)
#   b3g1 T=1.00 == restr_b2_T100   (b1 alone @0.50)
ALREADY_MEASURED = {"celeba_v2b_restr_b5g2_T100": "celeba_v2b_restr_b3_T100",
                    "celeba_v2b_restr_b3g1_T100": "celeba_v2b_restr_b2_T100"}


def t_to_sigma(t):
    """sigma = exp(1.2 * Phi^-1(T) - 1.2), matching create_v2b_cond_*.py."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        t = INACTIVE_T
    return float(np.exp(P_STD * norm.ppf(t) + P_MEAN))


def t_to_suffix(t):
    m = round(t * 1000)
    return "%03d" % (m // 10) if m % 10 == 0 else "%04d" % m


def create_dataset(name, fixed_bucket, t_fixed, swept_bucket, t_swept):
    ds_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(os.path.join(ds_dir, "annotations.jsonl")):
        print(f"  SKIP {name} (already exists)")
        return False
    os.makedirs(ds_dir, exist_ok=True)

    sigma_fixed = t_to_sigma(t_fixed)
    sigma_swept = t_to_sigma(t_swept)
    sigma_off = t_to_sigma(INACTIVE_T)

    annotations = []

    # Clean images: eligible at every noise level, as in every other v2b dataset.
    for src in sorted(glob.glob(os.path.join(PROCESSED_DIR, "b0_*.jpg"))):
        fname = os.path.basename(src)
        if not os.path.lexists(os.path.join(ds_dir, fname)):
            os.symlink(src, os.path.join(ds_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0,
                            "sigma_max": 0.0, "sigma_band_max": UNBOUNDED})

    for bucket in ALL_BLUR_BUCKETS:
        if bucket == fixed_bucket:
            smin, band = sigma_fixed, sigma_swept   # the restriction
        elif bucket == swept_bucket:
            smin, band = sigma_swept, UNBOUNDED
        else:
            smin, band = sigma_off, UNBOUNDED
        for src in sorted(glob.glob(os.path.join(PROCESSED_DIR, f"b{bucket}_*.jpg"))):
            fname = os.path.basename(src)
            if not os.path.lexists(os.path.join(ds_dir, fname)):
                os.symlink(src, os.path.join(ds_dir, fname))
            annotations.append({"filename": fname, "sigma_min": smin,
                                "sigma_max": 0.0, "sigma_band_max": band})

    annotations.sort(key=lambda x: x["filename"])
    with open(os.path.join(ds_dir, "annotations.jsonl"), "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    degenerate = ("swept bucket solo" if t_swept <= t_fixed else
                  "fixed bucket solo" if t_swept >= 1.0 else None)
    meta = {
        "dataset": name,
        "design": "restricted",
        "fixed_bucket": fixed_bucket, "t_fixed": t_fixed,
        "swept_bucket": swept_bucket, "t_swept": t_swept,
        "band_fixed": [sigma_fixed, sigma_swept],
        "band_swept": [sigma_swept, None],
        "degenerate_as": degenerate,
        "n_images": len(annotations),
    }
    with open(os.path.join(TVEC_DIR, f"tvec_{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    note = f"  [{degenerate}]" if degenerate else ""
    print(f"  Created {name}: {len(annotations)} imgs | "
          f"b{fixed_bucket} in [{sigma_fixed:.4f}, {sigma_swept:.4f}) | "
          f"b{swept_bucket} in [{sigma_swept:.4f}, inf){note}")
    return True


def main():
    print("=== Restricted-bucket validation sweeps ===")
    print(f"    base={AMBIENT_BASE}")
    created = 0
    names = []
    for label, swept, fixed, t_fixed, sweep_ts in SWEEPS:
        print(f"\n-- sweep {label}: b{fixed} fixed at T={t_fixed}, sweeping b{swept} --")
        for t in sweep_ts:
            name = f"celeba_v2b_restr_{label}_T{t_to_suffix(t)}"
            names.append(name)
            if create_dataset(name, fixed, t_fixed, swept, t):
                created += 1
    queueable = [n for n in names if n not in ALREADY_MEASURED]
    with open(os.path.join(TVEC_DIR, "restr_sweep_manifest.json"), "w") as f:
        json.dump(queueable, f, indent=2)
    for dup, src in ALREADY_MEASURED.items():
        print(f"  NOT queued: {dup} -- same configuration as {src}")
    print(f"\n=== Done: {created} new datasets, {len(names)} total in the sweep ===")


if __name__ == "__main__":
    main()
