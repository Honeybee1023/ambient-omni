"""
Create annotated datasets for binary search over T-space.

Round 1: Each image (wolves, dogs, cats) randomly gets T=0 (included) or
T=100 (excluded). Creates num_models datasets with different random binary
assignments. All datasets share images via symlinks to baseline_naive_all.

Saves assignment matrix for attribution analysis.
"""

import os
import json
import numpy as np
from scipy.stats import norm
import argparse

P_MEAN = -1.2
P_STD = 1.2

def t_to_sigma_min(t_value):
    """Convert T in [0,1] to sigma_min using EDM noise schedule."""
    t_clipped = np.clip(t_value, 0.001, 0.999)
    z = norm.ppf(t_clipped)
    return float(np.exp(P_STD * z + P_MEAN))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str,
                        default="/data/scratch/honjar/annotated_datasets/baseline_naive_all",
                        help="Directory with pre-resized 64x64 images to symlink from")
    parser.add_argument("--output_root", type=str,
                        default="/data/scratch/honjar/annotated_datasets",
                        help="Root directory for output datasets")
    parser.add_argument("--round", type=int, default=1,
                        help="Binary search round number")
    parser.add_argument("--num_models", type=int, default=20,
                        help="Number of models (datasets) to create")
    parser.add_argument("--master_seed", type=int, default=2026,
                        help="Master seed for reproducibility")
    args = parser.parse_args()

    # Round 1: T=0 (included) vs T=100 (excluded)
    # T=100 maps to t=1.0, clipped to 0.999 -> sigma_min ~ 12.28
    SIGMA_MIN_EXCLUDED = t_to_sigma_min(0.999)
    print(f"T=0   -> sigma_min = 0.0 (fully included)")
    print(f"T=100 -> sigma_min = {SIGMA_MIN_EXCLUDED:.4f} (effectively excluded)")

    # List all image files from source directory
    all_files = sorted([f for f in os.listdir(args.source_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))])

    wolves = [f for f in all_files if f.startswith('wolf_')]
    dogs = [f for f in all_files if f.startswith('dog_')]
    cats = [f for f in all_files if f.startswith('cat_')]

    print(f"\nImages: {len(wolves)} wolves, {len(dogs)} dogs, {len(cats)} cats")
    print(f"Total: {len(all_files)} images")
    print(f"Creating {args.num_models} datasets for round {args.round}\n")

    # Generate assignment matrix: (num_models, num_images)
    # 0 = included (sigma_min=0), 1 = excluded (sigma_min~12.28)
    rng = np.random.RandomState(args.master_seed)
    assignments = rng.randint(0, 2, size=(args.num_models, len(all_files)))

    for model_idx in range(args.num_models):
        dataset_name = f"bsearch_r{args.round}_model_{model_idx:03d}"
        dataset_dir = os.path.join(args.output_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Symlink all images from source (no copying!)
        for fname in all_files:
            src = os.path.join(args.source_dir, fname)
            dst = os.path.join(dataset_dir, fname)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        # Write annotations.jsonl with binary T assignments
        model_assignments = assignments[model_idx]
        n_included = int(np.sum(model_assignments == 0))
        n_excluded = int(np.sum(model_assignments == 1))

        with open(os.path.join(dataset_dir, "annotations.jsonl"), "w") as f:
            for fname, assigned in zip(all_files, model_assignments):
                sigma_min = 0.0 if assigned == 0 else SIGMA_MIN_EXCLUDED
                f.write(json.dumps({
                    "filename": fname,
                    "sigma_min": sigma_min,
                    "sigma_max": 0.0
                }) + "\n")

        wolf_included = sum(1 for f, a in zip(all_files, model_assignments)
                           if f.startswith('wolf_') and a == 0)

        print(f"  {dataset_name}: {n_included} included / {n_excluded} excluded "
              f"(wolves: {wolf_included}/{len(wolves)} included)")

    # Save assignment matrix for attribution
    save_path = os.path.join(args.output_root, f"bsearch_r{args.round}_assignments.npz")
    np.savez(save_path,
             assignments=assignments,
             filenames=np.array(all_files),
             wolves=np.array(wolves),
             dogs=np.array(dogs),
             cats=np.array(cats),
             sigma_min_excluded=SIGMA_MIN_EXCLUDED,
             master_seed=args.master_seed,
             num_models=args.num_models,
             round_num=args.round)

    print(f"\nAssignment matrix saved: {save_path}")
    print(f"  Shape: {assignments.shape} (models x images)")
    print(f"\nDone! Created {args.num_models} datasets in {args.output_root}")

if __name__ == "__main__":
    main()
