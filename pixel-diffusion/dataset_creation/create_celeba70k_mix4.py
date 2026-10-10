#!/usr/bin/env python3
"""Build celeba70k_mix4: CelebA-64 on Ambient-o's FFHQ protocol (70k images, 10% clean, 90% blurred),
with our four blur levels. The setup for the training-curve and deployment-length phase (2026-10-10).

Source: CelebA aligned images (img_align_celeba, 178x218) with the OFFICIAL train/valid/test split,
from the Hugging Face mirror flwrlabs/celeba (config img_align+identity+attr), downloaded to
$AMBIENT_BASE/raw/celeba_hf. The official split is identity-disjoint.

Preprocessing:
  * Crop: the standard CelebA-64 crop used by NCSN/DDPM/DDIM -- a 128x128 box centred at (x=89, y=121)
    of the 178x218 aligned frame -- then LANCZOS resize to 64x64. (Our older CelebA sets resized the whole
    non-square frame, squashing the faces; this replaces that.)
  * Blur: the AFHQ (fixed) recipe -- gaussian_filter(float32, sigma=(s, s, 0)), ROUNDED, saved as PNG.
    Every image is lossless PNG, so a clean and a blurred face differ only by the blur.

Selection: 70,000 of the official train split by a fixed-seed permutation; 7,000 clean (b0), the other
63,000 split evenly over blur 0.3/0.5/1.0/2.0 (g03/g05/g10/g20).

Layout (same as afhqdog_mix4_v1):
  $AMBIENT_BASE/celeba70k_processed/train_clean_64/<id>.png  all 70,000 selected faces, clean
                                                            (FID/MIND reference = the full uncorrupted
                                                            training set, as EDM and Ambient-o; also the
                                                            memorization search set)
  $AMBIENT_BASE/celeba70k_processed/test_64/<id>.png         the whole official test split, clean (extra
                                                            held-out reference; never trained on)
  $AMBIENT_BASE/annotated_datasets/celeba70k_mix4/{b0,g03,g05,g10,g20}_<id>.png + annotations.jsonl
  $AMBIENT_BASE/annotated_datasets/celeba70k_mix4.meta.json

<id> = tr<index in official train split, 6 digits> or te<index in test split>.

Run as a CPU job (never on a login node). Needs pyarrow (installed to a side directory by the sbatch).
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import argparse, glob, io, json, os, shutil, sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import gaussian_filter

RAW = f"{AMBIENT_BASE}/raw/celeba_hf/img_align+identity+attr"
PROC = f"{AMBIENT_BASE}/celeba70k_processed"
RESOLUTION = 64
SEED = 20261010
N_TOTAL = 70000
CLEAN_FRAC = 0.10
SENTINEL = 999.0
LEVELS = [("g03", 0.3), ("g05", 0.5), ("g10", 1.0), ("g20", 2.0)]
CX, CY, HALF = 89, 121, 64          # NCSN/DDPM/DDIM CelebA crop: 128x128 centred at (89, 121)
OFFICIAL = {"train": 162770, "test": 19962}
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def read_split(split):
    files = sorted(glob.glob(f"{RAW}/{split}-*.parquet"))
    assert files, f"no parquet files for {split} in {RAW}"
    out = []
    for f in files:
        col = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        out.extend(c["bytes"] for c in col)
    return out


def to64(raw_bytes):
    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    assert im.size == (178, 218), im.size
    return im.crop((CX - HALF, CY - HALF, CX + HALF, CY + HALF)).resize((RESOLUTION, RESOLUTION), LANCZOS)


def blur(img, s):
    arr = gaussian_filter(np.array(img, dtype=np.float32), sigma=(s, s, 0))
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="celeba70k_mix4")
    ap.add_argument("--dry", action="store_true", help="read and verify the split, write nothing")
    args = ap.parse_args()
    out = f"{AMBIENT_BASE}/annotated_datasets/{args.name}"
    meta_path = f"{out}.meta.json"
    train_dir, test_dir = f"{PROC}/train_clean_64", f"{PROC}/test_64"

    train = read_split("train")
    test = read_split("test")
    print("official split sizes:", len(train), len(test), flush=True)
    for k, v in (("train", train), ("test", test)):
        if len(v) != OFFICIAL[k]:
            sys.exit(f"{k} has {len(v)} images, expected the official {OFFICIAL[k]}; refusing")

    perm = np.random.RandomState(SEED).permutation(len(train))[:N_TOTAL]
    n_clean = int(round(CLEAN_FRAC * N_TOTAL))
    clean = sorted(int(i) for i in perm[:n_clean])
    groups = {g: sorted(int(i) for i in p) for (g, _), p in zip(LEVELS, np.array_split(perm[n_clean:], len(LEVELS)))}
    chosen = sorted(int(i) for i in perm)
    assert len(set(chosen)) == N_TOTAL
    print("selected: clean", len(clean), {g: len(v) for g, v in groups.items()}, flush=True)
    if args.dry:
        to64(train[0]).save("/dev/null", format="PNG")
        print("dry run: crop/resize OK, nothing written")
        return

    for p in (out, meta_path, train_dir, test_dir):
        if os.path.exists(p):
            sys.exit(f"{p} already exists; refusing to overwrite")
    os.makedirs(train_dir); os.makedirs(test_dir); os.makedirs(out)

    tid = lambda i: f"tr{i:06d}"
    imgs = {}
    for n, i in enumerate(chosen):
        im = to64(train[i])
        im.save(f"{train_dir}/{tid(i)}.png")
        imgs[i] = im
        if (n + 1) % 10000 == 0:
            print(f"  train clean {n + 1}/{N_TOTAL}", flush=True)
    for i, b in enumerate(test):
        to64(b).save(f"{test_dir}/te{i:06d}.png")
    print(f"wrote {N_TOTAL} clean train -> {train_dir}, {len(test)} test -> {test_dir}", flush=True)

    lines = []
    for i in clean:
        shutil.copyfile(f"{train_dir}/{tid(i)}.png", f"{out}/b0_{tid(i)}.png")
        lines.append({"filename": f"b0_{tid(i)}.png", "sigma_min": 0.0, "sigma_max": 0.0})
    sig = dict(LEVELS)
    for g, ids in groups.items():
        for i in ids:
            blur(imgs[i], sig[g]).save(f"{out}/{g}_{tid(i)}.png")
            lines.append({"filename": f"{g}_{tid(i)}.png", "sigma_min": SENTINEL, "sigma_max": 0.0})
    with open(f"{out}/annotations.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    meta = {
        "source": "flwrlabs/celeba (img_align+identity+attr), official train/test split",
        "seed": SEED, "resolution": RESOLUTION,
        "crop": f"box ({CX - HALF},{CY - HALF},{CX + HALF},{CY + HALF}) of the 178x218 aligned frame (NCSN/DDPM/DDIM CelebA-64)",
        "recipe": "crop 128x128 -> resize 64x64 LANCZOS -> gaussian_filter(float32, sigma=(s,s,0)) "
                  "-> np.clip(np.rint(arr),0,255).astype(uint8); every image saved as PNG",
        "n_clean": len(clean), "clean_frac": CLEAN_FRAC,
        "groups": {g: {"blur_sigma": sig[g], "n": len(groups[g])} for g in groups},
        "n_total": len(lines),
        "reference_train_clean_64": train_dir, "n_reference": N_TOTAL,
        "heldout_test_64": test_dir, "n_test": len(test),
        "ids": {"b0": [tid(i) for i in clean], **{g: [tid(i) for i in v] for g, v in groups.items()}},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    on_disk = sorted(f for f in os.listdir(out) if f.endswith(".png"))
    assert on_disk == sorted(l["filename"] for l in lines), "files and annotations disagree"
    print(f"wrote {out}: {len(lines)} annotations, {len(on_disk)} PNGs; meta {meta_path}")


if __name__ == "__main__":
    main()
