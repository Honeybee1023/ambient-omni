"""
Prepare CelebA dataset for noise threshold experiments.

One-time setup:
1. Shuffle all 202,599 images with fixed seed
2. Split: 20K holdout + 8 equal training buckets (~22,825 each)
3. Resize all to 64x64
4. Apply Gaussian blur to buckets 1-7 at specified sigma levels
5. Save processed images + split metadata

Bucket 0: clean (target) — sigma_blur = 0
Bucket 1: sigma_blur = 0.5
Bucket 2: sigma_blur = 1.0
Bucket 3: sigma_blur = 2.0
Bucket 4: sigma_blur = 3.0
Bucket 5: sigma_blur = 4.0
Bucket 6: sigma_blur = 5.0
Bucket 7: sigma_blur = 8.0
"""
import os, json, time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# === Configuration ===
RAW_DIR = "/data/scratch/honjar/celeba_raw/images"
OUTPUT_ROOT = "/data/scratch/honjar/celeba_processed"
HOLDOUT_DIR = os.path.join(OUTPUT_ROOT, "holdout_64")
SHARED_DIR = os.path.join(OUTPUT_ROOT, "shared_buckets_64")

RESOLUTION = 64
MASTER_SEED = 2026
N_HOLDOUT = 20000
N_BUCKETS = 8

# Bucket index -> Gaussian blur sigma (in pixels, at 64x64)
BLUR_SIGMAS = {
    0: 0.0,   # clean target
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 3.0,
    5: 4.0,
    6: 5.0,
    7: 8.0,
}


def process_image(src_path, sigma_blur):
    """Load, resize to 64x64, optionally apply Gaussian blur."""
    img = Image.open(src_path).convert("RGB")
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    if sigma_blur > 0:
        arr = np.array(img, dtype=np.float32)
        # Blur spatial dims only, not color channels
        arr = gaussian_filter(arr, sigma=(sigma_blur, sigma_blur, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def main():
    t_start = time.time()
    print("=" * 60)
    print("CelebA Dataset Preparation")
    print("=" * 60)

    # Get all image files
    all_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.jpg') and not f.startswith('._')])
    print(f"Found {len(all_files)} images in {RAW_DIR}")

    # Shuffle with fixed seed
    rng = np.random.RandomState(MASTER_SEED)
    indices = rng.permutation(len(all_files))

    # Split: holdout + 8 equal training buckets
    holdout_indices = indices[:N_HOLDOUT]
    train_indices = indices[N_HOLDOUT:]

    bucket_size = len(train_indices) // N_BUCKETS
    bucket_assignments = {}  # image_index -> bucket_number
    for b in range(N_BUCKETS):
        start = b * bucket_size
        end = start + bucket_size if b < N_BUCKETS - 1 else len(train_indices)
        for idx in train_indices[start:end]:
            bucket_assignments[idx] = b

    print(f"\nHoldout: {len(holdout_indices)} images (clean, for FID)")
    for b in range(N_BUCKETS):
        count = sum(1 for v in bucket_assignments.values() if v == b)
        sigma = BLUR_SIGMAS[b]
        label = "clean (target)" if b == 0 else f"sigma_blur={sigma}"
        print(f"  Bucket {b}: {count:6d} images — {label}")
    print(f"  Total training: {len(bucket_assignments)} images")

    # Create output directories
    os.makedirs(HOLDOUT_DIR, exist_ok=True)
    os.makedirs(SHARED_DIR, exist_ok=True)

    # Check if already partially done (for resume)
    existing_holdout = len([f for f in os.listdir(HOLDOUT_DIR) if f.endswith('.jpg') and not f.startswith('._')])
    existing_train = len([f for f in os.listdir(SHARED_DIR) if f.endswith('.jpg') and not f.startswith('._')])
    if existing_holdout > 0 or existing_train > 0:
        print(f"\nWARNING: Output dirs not empty (holdout={existing_holdout}, train={existing_train})")
        print("Skipping existing files...")

    # Process holdout (clean, 64x64)
    print(f"\n--- Processing holdout images ---")
    skipped = 0
    for i in tqdm(holdout_indices, desc="Holdout"):
        fname = all_files[i]
        out_path = os.path.join(HOLDOUT_DIR, fname)
        if os.path.exists(out_path):
            skipped += 1
            continue
        img = process_image(os.path.join(RAW_DIR, fname), sigma_blur=0.0)
        img.save(out_path)
    if skipped > 0:
        print(f"  Skipped {skipped} existing files")

    # Process training buckets
    print(f"\n--- Processing training images ---")
    skipped = 0
    for idx in tqdm(sorted(bucket_assignments.keys()), desc="Training"):
        bucket = bucket_assignments[idx]
        sigma = BLUR_SIGMAS[bucket]
        fname = all_files[idx]
        out_name = f"b{bucket}_{fname}"
        out_path = os.path.join(SHARED_DIR, out_name)
        if os.path.exists(out_path):
            skipped += 1
            continue
        img = process_image(os.path.join(RAW_DIR, fname), sigma_blur=sigma)
        img.save(out_path)
    if skipped > 0:
        print(f"  Skipped {skipped} existing files")

    # Save metadata
    metadata = {
        "master_seed": MASTER_SEED,
        "resolution": RESOLUTION,
        "n_holdout": int(len(holdout_indices)),
        "n_buckets": N_BUCKETS,
        "blur_sigmas": {str(b): float(s) for b, s in BLUR_SIGMAS.items()},
        "bucket_counts": {
            str(b): sum(1 for v in bucket_assignments.values() if v == b)
            for b in range(N_BUCKETS)
        },
        "holdout_files": [all_files[i] for i in sorted(holdout_indices)],
    }
    meta_path = os.path.join(OUTPUT_ROOT, "celeba_split.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # Verify
    holdout_count = len([f for f in os.listdir(HOLDOUT_DIR) if f.endswith('.jpg') and not f.startswith('._')])
    train_count = len([f for f in os.listdir(SHARED_DIR) if f.endswith('.jpg') and not f.startswith('._')])
    elapsed = time.time() - t_start
    print(f"\n=== Verification ===")
    print(f"  Holdout: {holdout_count} images")
    print(f"  Training: {train_count} images")
    print(f"  Total: {holdout_count + train_count}")
    print(f"  Time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
