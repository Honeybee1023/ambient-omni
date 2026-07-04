"""
Classify AFHQ wild images into sub-categories using CLIP zero-shot classification.
Outputs per-category file lists for the per-category threshold experiment.

Based on classify_wolves.py — same CLIP model, same batch processing.
"""

import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Config
WILD_DIR = "/data/scratch/honjar/afhq/afhq/train/wild"
OUTPUT_DIR = "/data/scratch/honjar/afhq_classified"
BATCH_SIZE = 32

# Categories — these are the ones we want to separate into
# Using the same prompt format as classify_wolves.py
LABELS = [
    "a photo of a wolf",
    "a photo of a tiger",
    "a photo of a lion",
    "a photo of a fox",
    "a photo of a leopard",
    "a photo of a cheetah",
    "a photo of a lynx",
    "a photo of a wild cat",
    "a photo of a bear",
    "a photo of a raccoon",
    "a photo of another wild animal",
]

# Short names for output files (matches LABELS order)
SHORT_NAMES = [
    "wolf", "tiger", "lion", "fox", "leopard",
    "cheetah", "lynx", "wild_cat", "bear", "raccoon", "other",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CLIP
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Get all image files
    image_files = sorted([f for f in os.listdir(WILD_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {len(image_files)} images in wild category")

    # Classify in batches
    results = {}
    # Dict of category -> list of filenames
    category_files = {name: [] for name in SHORT_NAMES}

    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Classifying"):
        batch_files = image_files[i:i + BATCH_SIZE]
        images = []
        valid_files = []

        for f in batch_files:
            try:
                img = Image.open(os.path.join(WILD_DIR, f)).convert("RGB")
                images.append(img)
                valid_files.append(f)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not images:
            continue

        inputs = processor(text=LABELS, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        for j, fname in enumerate(valid_files):
            prob_dict = {LABELS[k]: float(probs[j][k]) for k in range(len(LABELS))}
            top_idx = torch.argmax(probs[j]).item()
            top_label = LABELS[top_idx]
            top_name = SHORT_NAMES[top_idx]
            top_prob = float(probs[j][top_idx])

            results[fname] = {
                "top_category": top_name,
                "top_probability": top_prob,
                "all_probs": prob_dict,
            }

            category_files[top_name].append(fname)

    # Save full results
    results_path = os.path.join(OUTPUT_DIR, "wild_classification_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save per-category file lists
    for name, files in category_files.items():
        if len(files) > 0:
            list_path = os.path.join(OUTPUT_DIR, f"{name}_files.json")
            with open(list_path, "w") as f:
                json.dump(sorted(files), f, indent=2)

    # Save a summary with all categories and counts
    summary = {
        name: {
            "count": len(files),
            "filenames": sorted(files),
        }
        for name, files in category_files.items()
    }
    summary_path = os.path.join(OUTPUT_DIR, "wild_category_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n=== Wild Animal Sub-Classification Summary ===")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully classified: {len(results)}")
    print(f"\nCategory distribution:")
    for name in SHORT_NAMES:
        count = len(category_files[name])
        if count > 0:
            pct = 100.0 * count / len(results)
            print(f"  {name:12s}: {count:5d}  ({pct:.1f}%)")

    # Flag categories with very few images (consider merging)
    print(f"\nCategories with < 20 images (consider merging into 'other'):")
    for name in SHORT_NAMES:
        count = len(category_files[name])
        if 0 < count < 20:
            print(f"  {name}: {count}")

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  wild_classification_results.json  -- per-image details")
    print(f"  wild_category_summary.json        -- counts + file lists")
    print(f"  <category>_files.json             -- per-category file lists")


if __name__ == "__main__":
    main()
