"""
Prepare CelebA dataset v2 for data-poor regime experiments.

Changes from v1:
- Only 500 clean training images (data-poor regime)
- 7 blur buckets with milder sigmas: 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0
- Remaining ~182K images split evenly across 7 blur buckets (~26K each)
- Reuses same holdout as v1 (same master seed, same 20K images)
- Holdout dir is symlinked to v1's holdout (identical images)

Bucket 0: clean (target) — 500 images, sigma_blur = 0
Bucket 1: ~26,014 images, sigma_blur = 0.1
Bucket 2: ~26,014 images, sigma_blur = 0.3
Bucket 3: ~26,014 images, sigma_blur = 0.5
Bucket 4: ~26,014 images, sigma_blur = 0.75
Bucket 5: ~26,014 images, sigma_blur = 1.0
Bucket 6: ~26,014 images, sigma_blur = 1.5
Bucket 7: ~26,014 images, sigma_blur = 2.0
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# === Configuration ===
RAW_DIR = f"{AMBIENT_BASE}/celeba_raw/images"
OUTPUT_ROOT = f"{AMBIENT_BASE}/celeba_processed_v2"
HOLDOUT_SRC = f"{AMBIENT_BASE}/celeba_processed/holdout_64"
HOLDOUT_DIR = os.path.join(OUTPUT_ROOT, "holdout_64")
SHARED_DIR = os.path.join(OUTPUT_ROOT, "shared_buckets_64")

RESOLUTION = 64
MASTER_SEED = 2026
N_HOLDOUT = 20000
N_CLEAN = 500
N_BUCKETS = 8

BLUR_SIGMAS = {
    0: 0.0,
    1: 0.1,
    2: 0.3,
    3: 0.5,
    4: 0.75,
    5: 1.0,
    6: 1.5,
    7: 2.0,
}


def process_image(src_path, sigma_blur):
    """Load, resize to 64x64, optionally apply Gaussian blur."""
    img = Image.open(src_path).convert("RGB")
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    if sigma_blur > 0:
        arr = np.array(img, dtype=np.float32)
        arr = gaussian_filter(arr, sigma=(sigma_blur, sigma_blur, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def main():
    t_start = time.time()
    print("=" * 60)
    print("CelebA Dataset v2 — Data-Poor Regime")
    print("=" * 60)

    all_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.jpg')])
    print(f"Found {len(all_files)} images in {RAW_DIR}")

    rng = np.random.RandomState(MASTER_SEED)
    indices = rng.permutation(len(all_files))

    holdout_indices = indices[:N_HOLDOUT]
    train_indices = indices[N_HOLDOUT:]

    clean_indices = train_indices[:N_CLEAN]
    blur_indices = train_indices[N_CLEAN:]

    n_blur_buckets = N_BUCKETS - 1
    blur_bucket_size = len(blur_indices) // n_blur_buckets

    bucket_assignments = {}

    for idx in clean_indices:
        bucket_assignments[idx] = 0

    for b in range(n_blur_buckets):
        start = b * blur_bucket_size
        end = start + blur_bucket_size if b < n_blur_buckets - 1 else len(blur_indices)
        for idx in blur_indices[start:end]:
            bucket_assignments[idx] = b + 1

    print(f"\nHoldout: {len(holdout_indices)} images (clean, for eval — reused from v1)")
    for b in range(N_BUCKETS):
        count = sum(1 for v in bucket_assignments.values() if v == b)
        sigma = BLUR_SIGMAS[b]
        label = f"clean (target), {count} images" if b == 0 else f"sigma_blur={sigma}, {count} images"
        print(f"  Bucket {b}: {label}")
    print(f"  Total training: {len(bucket_assignments)} images")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(SHARED_DIR, exist_ok=True)

    if not os.path.exists(HOLDOUT_DIR):
        if os.path.exists(HOLDOUT_SRC):
            os.symlink(HOLDOUT_SRC, HOLDOUT_DIR)
            print(f"\nHoldout symlinked: {HOLDOUT_DIR} -> {HOLDOUT_SRC}")
        else:
            print(f"\nWARNING: v1 holdout not found at {HOLDOUT_SRC}")
            print("You will need to create holdout images separately.")
    else:
        print(f"\nHoldout already exists at {HOLDOUT_DIR}")

    existing_train = len([f for f in os.listdir(SHARED_DIR) if f.endswith('.jpg')])
    if existing_train > 0:
        print(f"\nWARNING: Output dir not empty (train={existing_train})")
        print("Skipping existing files...")

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

    metadata = {
        "version": "v2_data_poor",
        "master_seed": MASTER_SEED,
        "resolution": RESOLUTION,
        "n_holdout": int(len(holdout_indices)),
        "n_clean": N_CLEAN,
        "n_buckets": N_BUCKETS,
        "blur_sigmas": {str(b): float(s) for b, s in BLUR_SIGMAS.items()},
        "bucket_counts": {
            str(b): sum(1 for v in bucket_assignments.values() if v == b)
            for b in range(N_BUCKETS)
        },
        "holdout_files": [all_files[i] for i in sorted(holdout_indices)],
        "clean_files": [all_files[i] for i in sorted(clean_indices)],
        "changes_from_v1": [
            "500 clean images instead of ~22,825",
            "7 blur buckets with sigmas [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]",
            "Old sigmas were [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0]",
            "Same holdout, same master seed, same shuffle order",
        ],
    }
    meta_path = os.path.join(OUTPUT_ROOT, "celeba_split_v2.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    holdout_ok = os.path.exists(HOLDOUT_DIR) and len(os.listdir(HOLDOUT_DIR)) > 0
    train_count = len([f for f in os.listdir(SHARED_DIR) if f.endswith('.jpg')])
    elapsed = time.time() - t_start
    print(f"\n=== Verification ===")
    print(f"  Holdout: {'OK (symlinked)' if holdout_ok else 'MISSING'}")
    print(f"  Training: {train_count} images")
    print(f"  Expected: {len(bucket_assignments)}")
    print(f"  Time: {elapsed/60:.1f} minutes")

    print(f"\n  Per-bucket counts:")
    for b in range(N_BUCKETS):
        prefix = f"b{b}_"
        count = len([f for f in os.listdir(SHARED_DIR) if f.startswith(prefix)])
        expected = sum(1 for v in bucket_assignments.values() if v == b)
        status = "OK" if count == expected else f"MISMATCH (expected {expected})"
        print(f"    Bucket {b} (sigma={BLUR_SIGMAS[b]}): {count} — {status}")


if __name__ == "__main__":
    main()
