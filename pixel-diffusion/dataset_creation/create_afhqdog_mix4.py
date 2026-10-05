#!/usr/bin/env python3
"""Build the AFHQ-dog mixed-blur dataset (afhqdog_mix4_v1), the AFHQ counterpart of celeba_mix4_v1.

Source: AFHQ v1 dogs, $AMBIENT_BASE/raw/afhq/afhq/{train,val}/dog (4,739 train, 500 val, 512px JPEG).

Preprocessing is the CelebA v2 recipe (dataset_creation/prepare_celeba_v2.py): PIL convert RGB,
resize to 64x64 with LANCZOS, then scipy gaussian_filter(sigma=(s, s, 0)) on float32 -- with ONE
deliberate fix:

  * The CelebA pipeline converted the blurred float array with np.clip(arr, 0, 255).astype(uint8),
    which TRUNCATES (a ~-0.5 grey-level bias on every blurred pixel), and saved with PIL's default
    JPEG quality 75 (a second, lossy corruption on top of the blur). Here blurred arrays are rounded
    (np.clip(np.rint(arr), 0, 255).astype(uint8)) and EVERY image, clean and blurred, is saved as
    lossless PNG. So the only difference between a clean and a blurred dog is the blur itself.

Layout (mirrors celeba_mix4_v1: files <group>_<id>, annotations.jsonl inside, meta BESIDE the folder
because the loader scans the folder):

  $AMBIENT_BASE/afhqdog_processed/train_clean_64/<id>.png   all 4,739 train dogs, clean (MIND/FID reference;
                                                            Ambient-o measures FID against the full clean set)
  $AMBIENT_BASE/afhqdog_processed/val_64/<id>.png           all 500 val dogs, clean, held out (never trained on)
  $AMBIENT_BASE/annotated_datasets/afhqdog_mix4_v1/
        b0_<id>.png    474 clean (10% of 4,739, rounded)           sigma_min 0
        g03_<id>.png   1,067 blurred sigma_B 0.3                   sigma_min 999 (sentinel)
        g05_<id>.png   1,066 blurred sigma_B 0.5                   sigma_min 999
        g10_<id>.png   1,066 blurred sigma_B 1.0                   sigma_min 999
        g20_<id>.png   1,066 blurred sigma_B 2.0                   sigma_min 999
        annotations.jsonl
  $AMBIENT_BASE/annotated_datasets/afhqdog_mix4_v1.meta.json   seed, counts, every id per group

Every train dog appears exactly once (disjoint groups); the split is a fixed-seed permutation.
<id> is the AFHQ file stem (e.g. flickr_dog_000002); group_of() splits on the FIRST underscore, so
the group key is still b0/g03/... .

Run as a CPU job (never on a login node):
    python dataset_creation/create_afhqdog_mix4.py [--dry]
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import argparse, json, os, shutil, sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

RAW = f"{AMBIENT_BASE}/raw/afhq/afhq"
PROC = f"{AMBIENT_BASE}/afhqdog_processed"
RESOLUTION = 64
SEED = 20261005
CLEAN_FRAC = 0.10
SENTINEL = 999.0
LEVELS = [("g03", 0.3), ("g05", 0.5), ("g10", 1.0), ("g20", 2.0)]
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def to64(path):
    """CelebA v2 recipe: convert RGB, LANCZOS resize to 64x64 (PIL returns uint8, already rounded)."""
    return Image.open(path).convert("RGB").resize((RESOLUTION, RESOLUTION), LANCZOS)


def blur(img, s):
    """CelebA v2 blur (gaussian_filter on float32, channel axis untouched), but ROUNDED, not truncated."""
    arr = gaussian_filter(np.array(img, dtype=np.float32), sigma=(s, s, 0))
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def stems(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [(os.path.splitext(f)[0], os.path.join(d, f)) for f in fs]


def split(train_ids):
    """Fixed-seed permutation -> 10% clean, rest split as evenly as possible over the 4 levels."""
    n = len(train_ids)
    perm = np.random.RandomState(SEED).permutation(n)
    n_clean = int(round(CLEAN_FRAC * n))
    clean = sorted(train_ids[i] for i in perm[:n_clean])
    parts = np.array_split(perm[n_clean:], len(LEVELS))       # first part takes the remainder
    groups = {g: sorted(train_ids[i] for i in p) for (g, _), p in zip(LEVELS, parts)}
    return clean, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="afhqdog_mix4_v1")
    ap.add_argument("--dry", action="store_true", help="select and verify the split, write nothing")
    args = ap.parse_args()
    out = f"{AMBIENT_BASE}/annotated_datasets/{args.name}"
    meta_path = f"{out}.meta.json"
    train_dir, val_dir = f"{PROC}/train_clean_64", f"{PROC}/val_64"

    train = stems(f"{RAW}/train/dog")
    val = stems(f"{RAW}/val/dog")
    assert len(train) == 4739 and len(val) == 500, (len(train), len(val))
    train_ids = [s for s, _ in train]
    val_ids = [s for s, _ in val]
    assert len(set(train_ids)) == len(train_ids) and not set(train_ids) & set(val_ids), "id clash"

    clean, groups = split(train_ids)
    # Disjointness and coverage: every train dog exactly once.
    seen = {}
    for g, ids in [("b0", clean)] + list(groups.items()):
        for i in ids:
            if i in seen:
                sys.exit(f"dog {i} would appear in both {seen[i]} and {g}; refusing")
            seen[i] = g
    assert set(seen) == set(train_ids), "split does not cover the train set exactly"
    print("selected: clean", len(clean), {g: len(v) for g, v in groups.items()})
    if args.dry:
        print("dry run: nothing written")
        return

    for p in (out, meta_path, train_dir, val_dir):
        if os.path.exists(p):
            sys.exit(f"{p} already exists; refusing to overwrite")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(out)

    imgs = {}
    for i, (s, path) in enumerate(train):
        im = to64(path)
        im.save(f"{train_dir}/{s}.png")
        imgs[s] = im
        if (i + 1) % 1000 == 0:
            print(f"  train clean {i + 1}/{len(train)}", flush=True)
    for s, path in val:
        to64(path).save(f"{val_dir}/{s}.png")
    print(f"wrote {len(train)} clean train -> {train_dir}, {len(val)} val -> {val_dir}", flush=True)

    lines = []
    for s in clean:
        shutil.copyfile(f"{train_dir}/{s}.png", f"{out}/b0_{s}.png")    # byte-identical to the reference copy
        lines.append({"filename": f"b0_{s}.png", "sigma_min": 0.0, "sigma_max": 0.0})
    sig = dict(LEVELS)
    for g, ids in groups.items():
        for s in ids:
            blur(imgs[s], sig[g]).save(f"{out}/{g}_{s}.png")
            lines.append({"filename": f"{g}_{s}.png", "sigma_min": SENTINEL, "sigma_max": 0.0})
    with open(f"{out}/annotations.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    meta = {
        "source": f"{RAW}/{{train,val}}/dog (AFHQ v1)",
        "seed": SEED, "resolution": RESOLUTION,
        "recipe": "PIL convert RGB -> resize 64x64 LANCZOS -> gaussian_filter(float32, sigma=(s,s,0)) "
                  "-> np.clip(np.rint(arr),0,255).astype(uint8); every image saved as PNG",
        "fix_vs_celeba": "CelebA mix4 truncated (astype without rint, ~-0.5 grey level) and saved JPEG q75",
        "n_clean": len(clean), "clean_frac": CLEAN_FRAC,
        "groups": {g: {"blur_sigma": sig[g], "n": len(groups[g])} for g in groups},
        "n_total": len(lines),
        "reference_train_clean_64": train_dir, "n_reference": len(train),
        "heldout_val_64": val_dir, "n_val": len(val),
        "ids": {"b0": clean, **groups, "val": val_ids},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)   # beside, not inside: the loader scans the folder

    # Verify what was written, not what was intended.
    on_disk = sorted(f for f in os.listdir(out) if f.endswith(".png"))
    assert on_disk == sorted(l["filename"] for l in lines), "files and annotations disagree"
    print(f"wrote {out}: {len(lines)} annotations, {len(on_disk)} PNGs; meta {meta_path}")


if __name__ == "__main__":
    main()
