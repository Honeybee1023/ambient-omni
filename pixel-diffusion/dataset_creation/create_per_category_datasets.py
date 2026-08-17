"""
Create annotated datasets with per-CATEGORY T values for the regression experiment.

Each model gets a random T value per supplementary category (uniform [0,1]).
Wolves (target) always get T=0 (fully included).
All images in a category share the same T value for a given model.

Categories from AFHQ:
  - Wild: wolf (target), tiger, lion, fox, leopard, cheetah
  - Domestic: dog, cat
  - Dropped: lynx, wild_cat, raccoon, other (too few images)

Step 1: Create shared 64x64 image directory (if not already done)
Step 2: For each model, sample T per category, create symlinks + annotations.jsonl
Step 3: Save assignment matrix as .npz for regression
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

# EDM noise schedule parameters (same as create_annotated_datasets.py)
P_MEAN = -1.2
P_STD = 1.2

# Paths
DATA_ROOT = f"{AMBIENT_BASE}/afhq/afhq"
CLASSIFIED_DIR = f"{AMBIENT_BASE}/afhq_classified"
OUTPUT_ROOT = f"{AMBIENT_BASE}/annotated_datasets"
SHARED_DIR = os.path.join(OUTPUT_ROOT, "shared_all_categories_64")

# Wild sub-categories to KEEP (from classifier)
WILD_CATEGORIES = ["wolf", "tiger", "lion", "fox", "leopard", "cheetah"]
# Dropped: lynx (22), wild_cat (35), raccoon (3)

# Domestic categories
DOMESTIC_CATEGORIES = ["dog", "cat"]

# All supplementary categories (everything except wolf)
SUPPLEMENTARY_CATEGORIES = ["tiger", "lion", "fox", "leopard", "cheetah", "dog", "cat"]

# Experiment parameters
NUM_MODELS = 20
MASTER_SEED = 2026
RESOLUTION = 64


def t_to_sigma_min(t_value):
    """Convert T value in [0, 1] to sigma_min using EDM noise schedule.
    Same function as create_annotated_datasets.py.
    """
    t_clipped = np.clip(t_value, 0.001, 0.999)
    z = norm.ppf(t_clipped)
    sigma_min = np.exp(P_STD * z + P_MEAN)
    return float(sigma_min)


def create_shared_directory():
    """Create shared directory with all images resized to 64x64.
    Each image is named {category}_{original_filename}.
    Only needs to run once — subsequent calls skip if directory exists.
    """
    if os.path.exists(SHARED_DIR):
        existing = [f for f in os.listdir(SHARED_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(existing) > 0:
            print(f"Shared directory already exists with {len(existing)} images, skipping creation.")
            return
    
    os.makedirs(SHARED_DIR, exist_ok=True)
    
    # Wild categories
    wild_dir = os.path.join(DATA_ROOT, "train/wild")
    for cat in WILD_CATEGORIES:
        list_path = os.path.join(CLASSIFIED_DIR, f"{cat}_files.json")
        with open(list_path) as f:
            files = json.load(f)
        
        print(f"  Resizing {len(files)} {cat} images...")
        for fname in tqdm(files, desc=f"  {cat}", leave=False):
            src = os.path.join(wild_dir, fname)
            dst = os.path.join(SHARED_DIR, f"{cat}_{fname}")
            img = Image.open(src).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            img.save(dst)
    
    # Domestic categories
    for cat in DOMESTIC_CATEGORIES:
        cat_dir = os.path.join(DATA_ROOT, f"train/{cat}")
        files = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"  Resizing {len(files)} {cat} images...")
        for fname in tqdm(files, desc=f"  {cat}", leave=False):
            src = os.path.join(cat_dir, fname)
            dst = os.path.join(SHARED_DIR, f"{cat}_{fname}")
            img = Image.open(src).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            img.save(dst)
    
    total = len([f for f in os.listdir(SHARED_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"  Shared directory created: {total} images at {RESOLUTION}x{RESOLUTION}")


def load_file_lists():
    """Load file lists for all kept categories. Returns dict of category -> list of filenames
    (filenames as they appear in the shared directory, i.e., {category}_{original}).
    """
    file_lists = {}
    
    # Wild categories
    for cat in WILD_CATEGORIES:
        list_path = os.path.join(CLASSIFIED_DIR, f"{cat}_files.json")
        with open(list_path) as f:
            original_files = json.load(f)
        file_lists[cat] = [f"{cat}_{fname}" for fname in original_files]
    
    # Domestic categories
    for cat in DOMESTIC_CATEGORIES:
        cat_dir = os.path.join(DATA_ROOT, f"train/{cat}")
        original_files = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        file_lists[cat] = [f"{cat}_{fname}" for fname in original_files]
    
    return file_lists


def create_model_dataset(model_idx, t_values, file_lists):
    """Create one model's dataset with symlinks + annotations.
    
    t_values: dict of category -> T value in [0, 1]. Wolves should be 0.
    file_lists: dict of category -> list of filenames in shared directory.
    """
    dataset_name = f"percat_r1_model_{model_idx:03d}"
    dataset_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    
    # Clean up if exists
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    
    annotations = []
    
    for cat in WILD_CATEGORIES + DOMESTIC_CATEGORIES:
        t_val = t_values[cat]
        sigma_min = 0.0 if t_val == 0 else t_to_sigma_min(t_val)
        
        for fname in file_lists[cat]:
            # Symlink to shared directory
            src = os.path.join(SHARED_DIR, fname)
            dst = os.path.join(dataset_dir, fname)
            os.symlink(src, dst)
            
            annotations.append({
                "filename": fname,
                "sigma_min": sigma_min,
                "sigma_max": 0.0,
            })
    
    # Write annotations
    with open(os.path.join(dataset_dir, "annotations.jsonl"), "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    
    return len(annotations)


def main():
    print("=" * 60)
    print("Per-Category Dataset Creation for Regression Experiment")
    print("=" * 60)
    
    # Step 1: Create shared 64x64 images
    print(f"\n=== Step 1: Creating shared 64x64 image directory ===")
    create_shared_directory()
    
    # Step 2: Load file lists
    print(f"\n=== Step 2: Loading file lists ===")
    file_lists = load_file_lists()
    for cat in WILD_CATEGORIES + DOMESTIC_CATEGORIES:
        print(f"  {cat:12s}: {len(file_lists[cat]):5d} images")
    total_images = sum(len(v) for v in file_lists.values())
    print(f"  {'TOTAL':12s}: {total_images:5d} images")
    
    # Step 3: Sample T assignments and create datasets
    print(f"\n=== Step 3: Creating {NUM_MODELS} model datasets ===")
    rng = np.random.RandomState(MASTER_SEED)
    
    # Assignment matrix: rows = models, columns = categories
    # Store T values for regression
    all_t_values = []
    
    for model_idx in range(NUM_MODELS):
        # Sample T for each supplementary category
        t_values = {}
        t_values["wolf"] = 0.0  # Target: always fully included
        
        for cat in SUPPLEMENTARY_CATEGORIES:
            t_values[cat] = float(rng.uniform(0, 1))
        
        all_t_values.append(t_values)
        
        n_images = create_model_dataset(model_idx, t_values, file_lists)
        
        # Print this model's assignment
        t_str = "  ".join([f"{cat[:3]}={t_values[cat]:.2f}" for cat in SUPPLEMENTARY_CATEGORIES])
        print(f"  Model {model_idx:03d}: {n_images} images  |  {t_str}")
    
    # Step 4: Save assignment matrix
    print(f"\n=== Step 4: Saving assignment matrix ===")
    
    # Save as structured npz
    categories = SUPPLEMENTARY_CATEGORIES
    t_matrix = np.array([[tv[cat] for cat in categories] for tv in all_t_values])
    # t_matrix shape: (NUM_MODELS, len(SUPPLEMENTARY_CATEGORIES))
    
    npz_path = os.path.join(OUTPUT_ROOT, "percat_r1_assignments.npz")
    np.savez(npz_path,
             t_matrix=t_matrix,
             categories=np.array(categories),
             num_models=NUM_MODELS,
             seed=MASTER_SEED)
    
    # Also save as human-readable JSON
    json_path = os.path.join(OUTPUT_ROOT, "percat_r1_assignments.json")
    with open(json_path, "w") as f:
        json.dump({
            "categories": categories,
            "supplementary_categories": SUPPLEMENTARY_CATEGORIES,
            "num_models": NUM_MODELS,
            "seed": MASTER_SEED,
            "assignments": all_t_values,
            "category_counts": {cat: len(file_lists[cat]) for cat in WILD_CATEGORIES + DOMESTIC_CATEGORIES},
        }, f, indent=2)
    
    print(f"  Saved: {npz_path}")
    print(f"  Saved: {json_path}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"  Models created: {NUM_MODELS}")
    print(f"  Categories: {len(WILD_CATEGORIES) + len(DOMESTIC_CATEGORIES)} "
          f"({len(SUPPLEMENTARY_CATEGORIES)} with variable T + wolf at T=0)")
    print(f"  Images per model: {total_images}")
    print(f"  T range: [0, 1] continuous (Uniform)")
    print(f"  Dataset prefix: percat_r1_model_XXX")
    print(f"  Assignment matrix shape: {t_matrix.shape}")
    print(f"\n  T matrix (rows=models, cols={categories}):")
    for i in range(NUM_MODELS):
        row = "  ".join([f"{t_matrix[i,j]:.3f}" for j in range(len(categories))])
        print(f"    Model {i:03d}: {row}")


if __name__ == "__main__":
    main()
