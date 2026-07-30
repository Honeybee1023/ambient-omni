"""
Create the dataset for dynamic T scheduling experiments.
500 clean + 4500 blurred (sigma_blur=2.0) CelebA images at 64x64.
Annotations mark corrupt images with sigma_min=999 (placeholder; actual T set by schedule).
"""
import os, json, shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

CELEBA_SRC = "/var/local/honjar/celeba_processed_v2b/shared_buckets_64"
OUTPUT_DIR = "/var/local/honjar/annotated_datasets/celeba_dynamic_t"
SIGMA_BLUR = 2.0
N_CLEAN = 500
N_CORRUPT = 4500


def main():
    # Get clean images (b0_ prefix)
    clean_files = sorted([f for f in os.listdir(CELEBA_SRC) if f.startswith("b0_") and f.endswith(".jpg")])
    assert len(clean_files) >= N_CLEAN, f"Only {len(clean_files)} clean files"
    clean_files = clean_files[:N_CLEAN]

    # Get ALL other images for corruption source (use b0_ images beyond the 500 clean ones as source for blurring)
    # Actually let's use the remaining b0_ images blurred, to avoid confounding with pre-existing blur
    all_b0 = sorted([f for f in os.listdir(CELEBA_SRC) if f.startswith("b0_") and f.endswith(".jpg")])
    # We only have 500 b0_ images. Use b1_ (sigma_blur=0.5) as source and re-blur to sigma=2.0
    # Actually, let's just use images from other buckets as source and apply our own blur
    # Simplest: use images from ALL buckets (excluding the 500 clean) and blur them ourselves
    # But the bucket images are already blurred. Let's use b0_ for clean, and for corrupt let's
    # take more b0_ images... but there are only 500.
    
    # Alternative: take raw celeba images and blur them. The b0_ are clean originals.
    # We only have 500 b0_ images. So for corrupt, let's use images from bucket 3 (sigma_blur=2.0) directly.
    # These are already at our target blur level!
    b3_files = sorted([f for f in os.listdir(CELEBA_SRC) if f.startswith("b3_") and f.endswith(".jpg")])
    print(f"Available b3 (sigma_blur=2.0) images: {len(b3_files)}")
    assert len(b3_files) >= N_CORRUPT, f"Only {len(b3_files)} b3 files, need {N_CORRUPT}"
    corrupt_files = b3_files[:N_CORRUPT]

    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    annotations = []

    # Symlink clean images
    for fname in clean_files:
        src = os.path.join(CELEBA_SRC, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        os.symlink(src, dst)
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    # Symlink corrupt images (already blurred at sigma=2.0)
    for fname in corrupt_files:
        src = os.path.join(CELEBA_SRC, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        os.symlink(src, dst)
        # sigma_min=999 is a placeholder meaning "use schedule"
        annotations.append({"filename": fname, "sigma_min": 999.0, "sigma_max": 0.0})

    # Write annotations
    ann_path = os.path.join(OUTPUT_DIR, "annotations.jsonl")
    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    print(f"Created dataset at {OUTPUT_DIR}")
    print(f"  Clean: {N_CLEAN}, Corrupt: {N_CORRUPT} (sigma_blur={SIGMA_BLUR})")
    print(f"  Total: {N_CLEAN + N_CORRUPT} images")
    print(f"  Annotations: {ann_path}")


if __name__ == "__main__":
    main()
