"""Create clean-only CelebA datasets with reduced sample counts."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os
import json
import random
import sys

SHARED_DIR = f"{AMBIENT_BASE}/celeba_processed/shared_buckets_64"
OUT_BASE = f"{AMBIENT_BASE}/annotated_datasets"
SEED = 2026

def create_cleanonly(n_images, name):
    outdir = os.path.join(OUT_BASE, name)
    if os.path.exists(outdir):
        print(f"  {name} already exists, skipping")
        return
    os.makedirs(outdir, exist_ok=True)

    # Get all bucket-0 (clean) images
    all_clean = sorted([f for f in os.listdir(SHARED_DIR) if f.startswith("b0_")])
    print(f"  Total clean images available: {len(all_clean)}")

    # Subsample with fixed seed
    rng = random.Random(SEED)
    selected = rng.sample(all_clean, n_images)
    selected.sort()

    # Create symlinks + annotations
    annotations = []
    for fname in selected:
        src = os.path.join(SHARED_DIR, fname)
        dst = os.path.join(outdir, fname)
        os.symlink(src, dst)
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    with open(os.path.join(outdir, "annotations.jsonl"), "w") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")

    print(f"  Created {name}: {len(selected)} images")

if __name__ == "__main__":
    configs = [
        (2000, "celeba_cleanonly_2000"),
        (1000, "celeba_cleanonly_1000"),
        (500,  "celeba_cleanonly_500"),
    ]
    for n, name in configs:
        print(f"Creating {name}...")
        create_cleanonly(n, name)
    print("Done.")
