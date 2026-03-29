"""
Classify wolf images from AFHQ wild category using CLIP zero-shot classification.
Outputs a list of filenames classified as wolves and saves example images for verification.
"""

import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Config
WILD_DIR = os.path.expanduser("~/data/afhq/afhq/train/wild")
OUTPUT_DIR = os.path.expanduser("~/data/afhq_classified")
THRESHOLD = 0.5  # minimum probability to be classified as wolf
BATCH_SIZE = 32

# Categories to classify against
LABELS = [
    "a photo of a wolf",
    "a photo of a tiger", 
    "a photo of a lion",
    "a photo of a fox",
    "a photo of a bear",
    "a photo of a leopard",
    "a photo of a cheetah",
    "a photo of a lynx",
    "a photo of a wild cat",
    "a photo of a raccoon",
    "a photo of another wild animal",
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
    wolf_files = []

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
            wolf_prob = prob_dict["a photo of a wolf"]
            top_label = LABELS[torch.argmax(probs[j]).item()]
            
            results[fname] = {
                "wolf_probability": wolf_prob,
                "top_label": top_label,
                "all_probs": prob_dict,
            }

            if wolf_prob > THRESHOLD or top_label == "a photo of a wolf":
                wolf_files.append(fname)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "classification_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    wolf_list_path = os.path.join(OUTPUT_DIR, "wolf_files.json")
    with open(wolf_list_path, "w") as f:
        json.dump(wolf_files, f, indent=2)

    # Print summary
    print(f"\n=== Classification Summary ===")
    print(f"Total images: {len(image_files)}")
    print(f"Classified as wolf: {len(wolf_files)}")
    
    # Count top labels
    label_counts = {}
    for r in results.values():
        label = r["top_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nSpecies distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        short_label = label.replace("a photo of ", "")
        print(f"  {short_label}: {count}")

    print(f"\nResults saved to {results_path}")
    print(f"Wolf file list saved to {wolf_list_path}")


if __name__ == "__main__":
    main()