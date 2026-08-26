"""Build the held-out probe set used by the principled T-schedule search.

The probe asks: at noise level sigma, can the model still tell a corrupted image
from a clean one? Answering that honestly needs images the model has never
trained on. The 500 clean training images have been seen ~4000 times by the end
of a 2000-kimg run, so a probe built from them would measure memorisation as
much as corruption.

So both arms come from the 20,000-image holdout split, and -- this is the part
that has to be exact -- they are rebuilt *from the raw CelebA files* through the
same `process_image` that produced the training buckets:

    raw jpg -> LANCZOS resize to 64 -> (optional) gaussian_filter -> jpg (q75)

Blurring the already-saved holdout_64 jpgs instead would put one extra
decode/encode cycle on the corrupt arm only, so a slice of the clean-vs-corrupt
gap would be JPEG artifacts rather than blur. Going back to raw costs a few
minutes once and removes the confound entirely.

Two directories are emitted over the SAME images and in the same order:

    probe_holdout_64/clean/    b0-equivalent (sigma_blur = 0.0)
    probe_holdout_64/blur05/   b5-equivalent (sigma_blur = 0.5)

Having both arms cover the same faces is what lets the probe run a *paired*
comparison (same face, same noise, blur is the only difference), which is far
more sensitive than comparing two disjoint samples. Disjoint comparisons are
still available -- take the clean arm from one index range and the corrupt arm
from another; see training/probe.py.

Raw CelebA lives only on lysine. Build there, then copy the (few MB) result:
    rsync -a $AMBIENT_BASE/probe_holdout_64/ proline:/var/local/honjar/probe_holdout_64/

Usage:
    python dataset_creation/create_probe_holdout.py [--n 1024]
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

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

RAW_DIR = f"{AMBIENT_BASE}/celeba_raw/img_align_celeba"
SPLIT_JSON = f"{AMBIENT_BASE}/celeba_processed_v2b/celeba_split_v2b.json"
OUT_ROOT = f"{AMBIENT_BASE}/probe_holdout_64"

RESOLUTION = 64
# Bucket b5 of celeba_processed_v2b. Keep in step with prepare_celeba_v2b.py.
BLUR_SIGMA = 0.5
# Which holdout images to take. Fixed so every machine and every rerun probes
# the same faces -- the T trajectory should move because the model moved, not
# because the probe set changed underneath it.
PICK_SEED = 20260826


def process_image(src_path, sigma_blur):
    """Byte-for-byte the pipeline in prepare_celeba_v2b.py."""
    img = Image.open(src_path).convert("RGB")
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    if sigma_blur > 0:
        arr = np.array(img, dtype=np.float32)
        arr = gaussian_filter(arr, sigma=(sigma_blur, sigma_blur, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024,
                    help="probe images per arm (both arms cover the same faces)")
    ap.add_argument("--out", default=OUT_ROOT)
    args = ap.parse_args()

    if not os.path.isdir(RAW_DIR):
        raise SystemExit(
            f"raw CelebA not found at {RAW_DIR}.\n"
            "Only lysine carries it. Build there and rsync $AMBIENT_BASE/probe_holdout_64 across.")

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    holdout = sorted(split["holdout_files"])
    if len(holdout) < args.n:
        raise SystemExit(f"holdout has {len(holdout)} files, need {args.n}")
    assert abs(split["blur_sigmas"]["5"] - BLUR_SIGMA) < 1e-9, \
        f"bucket b5 is sigma={split['blur_sigmas']['5']}, this script assumes {BLUR_SIGMA}"

    rng = np.random.RandomState(PICK_SEED)
    picked = [holdout[i] for i in sorted(rng.choice(len(holdout), args.n, replace=False))]

    arms = {"clean": 0.0, "blur05": BLUR_SIGMA}
    for arm, sigma_blur in arms.items():
        d = os.path.join(args.out, arm)
        os.makedirs(d, exist_ok=True)
        made = 0
        for fname in picked:
            out_path = os.path.join(d, fname)
            if os.path.exists(out_path):
                continue
            process_image(os.path.join(RAW_DIR, fname), sigma_blur).save(out_path)
            made += 1
        print(f"  {arm:<8} sigma_blur={sigma_blur}: {len(picked)} images ({made} new) -> {d}")

    meta = {
        "n_per_arm": args.n,
        "blur_sigma": BLUR_SIGMA,
        "matches_bucket": "b5 of celeba_processed_v2b",
        "resolution": RESOLUTION,
        "pick_seed": PICK_SEED,
        "source": "raw CelebA, holdout split of celeba_split_v2b.json",
        "files": picked,
    }
    with open(os.path.join(args.out, "probe_set.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Sanity: the arms must actually differ, and only mildly. A blur of 0.5 on
    # 64x64 moves pixels by a few levels; if this prints ~0 the blur silently
    # did nothing and every probe downstream would read "no divergence".
    a = np.array(Image.open(os.path.join(args.out, "clean", picked[0])), dtype=np.float32)
    b = np.array(Image.open(os.path.join(args.out, "blur05", picked[0])), dtype=np.float32)
    print(f"\nfirst image: mean|clean - blur05| = {np.abs(a - b).mean():.3f} / 255"
          f"   (max {np.abs(a - b).max():.0f})")
    print(f"wrote {os.path.join(args.out, 'probe_set.json')}")


if __name__ == "__main__":
    main()
