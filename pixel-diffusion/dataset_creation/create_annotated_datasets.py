"""
Create annotated datasets with random T vectors for the per-sample threshold experiment.

For each random T vector:
- Wolves (target): sigma_min = 0 (used at all noise levels)
- Dogs/cats (supplementary): sigma_min sampled via T ~ Uniform[0, 1] -> sigma_min = exp(1.2 * Phi_inv(T) - 1.2)

Outputs one annotated dataset folder per T vector, each containing:
- All images (wolves + dogs + cats)
- annotations.jsonl with per-image sigma_min values
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)


import os
import json
import shutil
import numpy as np
from scipy.stats import norm
from PIL import Image
from tqdm import tqdm
import argparse

# EDM noise schedule parameters
P_MEAN = -1.2
P_STD = 1.2

def t_to_sigma_min(t_values):
    """Convert T values in [0, 1] to sigma_min values using the EDM noise schedule.
    
    T represents the fraction of training noise levels where this image is EXCLUDED.
    T = 0 -> sigma_min ~ 0 (always used)
    T = 0.5 -> sigma_min = exp(P_MEAN) (used for half of training)
    T -> 1 -> sigma_min -> infinity (never used)
    """
    # Clip to avoid infinity at T=0 and T=1
    t_clipped = np.clip(t_values, 0.001, 0.999)
    # Inverse CDF of standard normal
    z = norm.ppf(t_clipped)
    # Convert to sigma
    sigma_min = np.exp(P_STD * z + P_MEAN)
    return sigma_min


def create_single_dataset(wolf_files, dog_files, cat_files, 
                          wolf_dir, dog_dir, cat_dir,
                          output_dir, t_vector_dogs, t_vector_cats, 
                          seed, resolution=64):
    """Create one annotated dataset with a specific T vector."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    annotations = []
    
    # Copy wolf images (target) - sigma_min = 0
    for fname in tqdm(wolf_files, desc="Wolves", leave=False):
        src = os.path.join(wolf_dir, fname)
        dst = os.path.join(output_dir, f"wolf_{fname}")
        
        # Resize to target resolution
        img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        img.save(dst)
        
        annotations.append({
            "filename": f"wolf_{fname}",
            "sigma_min": 0.0,
            "sigma_max": 0.0,
        })
    
    # Copy dog images (supplementary) - random sigma_min
    sigma_min_dogs = t_to_sigma_min(t_vector_dogs)
    for fname, sig_min in tqdm(zip(dog_files, sigma_min_dogs), desc="Dogs", 
                                total=len(dog_files), leave=False):
        src = os.path.join(dog_dir, fname)
        dst = os.path.join(output_dir, f"dog_{fname}")
        
        img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        img.save(dst)
        
        annotations.append({
            "filename": f"dog_{fname}",
            "sigma_min": float(sig_min),
            "sigma_max": 0.0,
        })
    
    # Copy cat images (supplementary) - random sigma_min
    sigma_min_cats = t_to_sigma_min(t_vector_cats)
    for fname, sig_min in tqdm(zip(cat_files, sigma_min_cats), desc="Cats",
                                total=len(cat_files), leave=False):
        src = os.path.join(cat_dir, fname)
        dst = os.path.join(output_dir, f"cat_{fname}")
        
        img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        img.save(dst)
        
        annotations.append({
            "filename": f"cat_{fname}",
            "sigma_min": float(sig_min),
            "sigma_max": 0.0,
        })
    
    # Write annotations
    annotations_path = os.path.join(output_dir, "annotations.jsonl")
    with open(annotations_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    
    # Save the T vector and sigma_min values for reference
    metadata = {
        "seed": seed,
        "num_wolves": len(wolf_files),
        "num_dogs": len(dog_files),
        "num_cats": len(cat_files),
        "t_vector_dogs": t_vector_dogs.tolist(),
        "t_vector_cats": t_vector_cats.tolist(),
        "sigma_min_dogs": sigma_min_dogs.tolist(),
        "sigma_min_cats": sigma_min_cats.tolist(),
        "resolution": resolution,
        "P_MEAN": P_MEAN,
        "P_STD": P_STD,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Created dataset: {len(annotations)} images, "
          f"dog sigma_min range: [{sigma_min_dogs.min():.3f}, {sigma_min_dogs.max():.3f}], "
          f"cat sigma_min range: [{sigma_min_cats.min():.3f}, {sigma_min_cats.max():.3f}]")


def create_baseline_dataset(wolf_files, dog_files, cat_files,
                           wolf_dir, dog_dir, cat_dir,
                           output_dir, mode="wolves_only", resolution=64):
    """Create baseline datasets for comparison.
    
    mode="wolves_only": only wolf images, all with sigma_min=0
    mode="naive_all": all images with sigma_min=0 (no gating)
    """
    os.makedirs(output_dir, exist_ok=True)
    annotations = []
    
    # Always include wolves
    for fname in tqdm(wolf_files, desc="Wolves", leave=False):
        src = os.path.join(wolf_dir, fname)
        dst = os.path.join(output_dir, f"wolf_{fname}")
        img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        img.save(dst)
        annotations.append({"filename": f"wolf_{fname}", "sigma_min": 0.0, "sigma_max": 0.0})
    
    if mode == "naive_all":
        for fname in tqdm(dog_files, desc="Dogs", leave=False):
            src = os.path.join(dog_dir, fname)
            dst = os.path.join(output_dir, f"dog_{fname}")
            img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
            img.save(dst)
            annotations.append({"filename": f"dog_{fname}", "sigma_min": 0.0, "sigma_max": 0.0})
        
        for fname in tqdm(cat_files, desc="Cats", leave=False):
            src = os.path.join(cat_dir, fname)
            dst = os.path.join(output_dir, f"cat_{fname}")
            img = Image.open(src).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
            img.save(dst)
            annotations.append({"filename": f"cat_{fname}", "sigma_min": 0.0, "sigma_max": 0.0})
    
    annotations_path = os.path.join(output_dir, "annotations.jsonl")
    with open(annotations_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    
    print(f"  Created {mode} baseline: {len(annotations)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=f"{AMBIENT_BASE}/afhq/afhq")
    parser.add_argument("--wolf_list", type=str, default=f"{AMBIENT_BASE}/afhq_classified/wolf_files.json")
    parser.add_argument("--output_root", type=str, default=f"{AMBIENT_BASE}/annotated_datasets")
    parser.add_argument("--num_random_vectors", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--master_seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load wolf file list
    with open(args.wolf_list) as f:
        wolf_files = json.load(f)
    print(f"Loaded {len(wolf_files)} wolf files")
    
    # Get dog and cat file lists
    dog_dir = os.path.join(args.data_root, "train/dog")
    cat_dir = os.path.join(args.data_root, "train/cat")
    wolf_dir = os.path.join(args.data_root, "train/wild")
    
    dog_files = sorted([f for f in os.listdir(dog_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    cat_files = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Dogs: {len(dog_files)}, Cats: {len(cat_files)}, Wolves: {len(wolf_files)}")
    
    # Create baseline datasets
    print("\n=== Creating baseline: wolves_only ===")
    create_baseline_dataset(
        wolf_files, dog_files, cat_files,
        wolf_dir, dog_dir, cat_dir,
        os.path.join(args.output_root, "baseline_wolves_only"),
        mode="wolves_only", resolution=args.resolution
    )
    
    print("\n=== Creating baseline: naive_all ===")
    create_baseline_dataset(
        wolf_files, dog_files, cat_files,
        wolf_dir, dog_dir, cat_dir,
        os.path.join(args.output_root, "baseline_naive_all"),
        mode="naive_all", resolution=args.resolution
    )
    
    # Create random T-vector datasets
    rng = np.random.RandomState(args.master_seed)
    
    for i in range(args.num_random_vectors):
        seed = rng.randint(0, 2**31)
        local_rng = np.random.RandomState(seed)
        
        # Sample T ~ Uniform[0, 1] for each supplementary image
        t_vector_dogs = local_rng.uniform(0, 1, size=len(dog_files))
        t_vector_cats = local_rng.uniform(0, 1, size=len(cat_files))
        
        dataset_name = f"random_t_vector_{i:03d}_seed{seed}"
        output_dir = os.path.join(args.output_root, dataset_name)
        
        print(f"\n=== Creating dataset {i+1}/{args.num_random_vectors}: {dataset_name} ===")
        create_single_dataset(
            wolf_files, dog_files, cat_files,
            wolf_dir, dog_dir, cat_dir,
            output_dir, t_vector_dogs, t_vector_cats,
            seed=seed, resolution=args.resolution
        )
    
    print(f"\n=== Done! Created {args.num_random_vectors + 2} datasets in {args.output_root} ===")


if __name__ == "__main__":
    main()