#!/usr/bin/env python3
"""
Evaluate generated images using three metrics:
  1. LAION Aesthetic Score (quality) - CLIP-L/14 embeddings + linear predictor
  2. PickScore (quality) - CLIP-H fine-tuned on human preferences
  3. Vendi Score (diversity) - eigenvalue entropy of similarity matrix

Usage:
  python eval_new_metrics.py --image_dir /path/to/images [--prompt "a photo of a wolf"] [--max_images 1000]
"""

import argparse
import os
import glob
import json
import numpy as np
import torch
from PIL import Image

# ============================================================
# 1. LAION AESTHETIC SCORE
# ============================================================
def load_aesthetic_model(device):
    """Load CLIP-L/14 + linear aesthetic predictor."""
    import open_clip
    import torch.nn as nn

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device
    )
    clip_model.eval()

    cache_dir = "/data/scratch/honjar/.cache/aesthetic_predictor"
    os.makedirs(cache_dir, exist_ok=True)
    weight_path = os.path.join(cache_dir, "sa_0_4_vit_l_14_linear.pth")

    if not os.path.exists(weight_path):
        from urllib.request import urlretrieve
        url = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        print(f"Downloading aesthetic predictor weights to {weight_path}...")
        urlretrieve(url, weight_path)

    linear = nn.Linear(768, 1)
    linear.load_state_dict(torch.load(weight_path, map_location=device))
    linear.eval().to(device)

    return clip_model, preprocess, linear


def compute_aesthetic_scores(image_paths, clip_model, preprocess, linear, device, batch_size=64):
    """Compute aesthetic score for each image. Returns list of floats."""
    scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        image_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            embs = clip_model.encode_image(image_tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            preds = linear(embs.float())

        scores.extend(preds.squeeze(-1).cpu().tolist())

        if (i // batch_size) % 5 == 0:
            print(f"  Aesthetic: {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

    return scores


# ============================================================
# 2. PICKSCORE
# ============================================================
def load_pickscore_model(device):
    """Load PickScore model and processor."""
    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    return model, processor


def compute_pickscore(image_paths, model, processor, prompt, device, batch_size=32):
    """Compute PickScore for each image given a text prompt. Returns list of floats."""
    scores = []

    text_inputs = processor(
        text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        image_inputs = processor(
            images=images, padding=True, truncation=True, max_length=77, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            batch_scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        scores.extend(batch_scores.cpu().tolist())

        if (i // batch_size) % 5 == 0:
            print(f"  PickScore: {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

    return scores


# ============================================================
# 3. VENDI SCORE
# ============================================================
def compute_vendi_score(image_paths, device, max_images=1000):
    """Compute Vendi diversity score for a set of images."""
    from vendi_score import image_utils

    images = [Image.open(p).convert("RGB") for p in image_paths[:max_images]]
    print(f"  Vendi: computing on {len(images)} images...")
    vs = image_utils.embedding_vendi_score(images, device=device, batch_size=64)
    return vs


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photograph of a wolf")
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--skip_pickscore", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image dir: {args.image_dir}")
    print(f"Prompt: '{args.prompt}'")

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not image_paths:
        print("ERROR: No images found!")
        return

    image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images\n")

    results = {"image_dir": args.image_dir, "num_images": len(image_paths), "prompt": args.prompt}

    # --- Aesthetic Score ---
    print("=" * 50)
    print("Computing LAION Aesthetic Scores...")
    print("=" * 50)
    try:
        clip_model, preprocess, linear = load_aesthetic_model(device)
        aesthetic_scores = compute_aesthetic_scores(image_paths, clip_model, preprocess, linear, device)
        del clip_model, preprocess, linear
        torch.cuda.empty_cache()

        results["aesthetic"] = {
            "mean": float(np.mean(aesthetic_scores)),
            "std": float(np.std(aesthetic_scores)),
            "median": float(np.median(aesthetic_scores)),
            "min": float(np.min(aesthetic_scores)),
            "max": float(np.max(aesthetic_scores)),
        }
        print(f"\n  Mean:   {results['aesthetic']['mean']:.4f}")
        print(f"  Std:    {results['aesthetic']['std']:.4f}")
        print(f"  Median: {results['aesthetic']['median']:.4f}")
        print(f"  Range:  [{results['aesthetic']['min']:.4f}, {results['aesthetic']['max']:.4f}]")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["aesthetic"] = {"error": str(e)}

    # --- PickScore ---
    if not args.skip_pickscore:
        print(f"\n{'=' * 50}")
        print("Computing PickScores...")
        print("=" * 50)
        try:
            ps_model, ps_processor = load_pickscore_model(device)
            pick_scores = compute_pickscore(image_paths, ps_model, ps_processor, args.prompt, device)
            del ps_model, ps_processor
            torch.cuda.empty_cache()

            results["pickscore"] = {
                "mean": float(np.mean(pick_scores)),
                "std": float(np.std(pick_scores)),
                "median": float(np.median(pick_scores)),
                "min": float(np.min(pick_scores)),
                "max": float(np.max(pick_scores)),
            }
            print(f"\n  Mean:   {results['pickscore']['mean']:.4f}")
            print(f"  Std:    {results['pickscore']['std']:.4f}")
            print(f"  Median: {results['pickscore']['median']:.4f}")
            print(f"  Range:  [{results['pickscore']['min']:.4f}, {results['pickscore']['max']:.4f}]")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results["pickscore"] = {"error": str(e)}

    # --- Vendi Score ---
    print(f"\n{'=' * 50}")
    print("Computing Vendi Diversity Score...")
    print("=" * 50)
    try:
        vs = compute_vendi_score(image_paths, device)
        results["vendi"] = {"score": float(vs)}
        print(f"\n  Vendi Score: {vs:.4f}")
        print(f"  (Interpretation: ~{vs:.0f} 'effective unique' images out of {len(image_paths)})")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["vendi"] = {"error": str(e)}

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print("=" * 50)
    print(f"  Images: {len(image_paths)} from {args.image_dir}")
    if "mean" in results.get("aesthetic", {}):
        print(f"  Aesthetic Score:  {results['aesthetic']['mean']:.4f} +/- {results['aesthetic']['std']:.4f}")
    if "mean" in results.get("pickscore", {}):
        print(f"  PickScore:        {results['pickscore']['mean']:.4f} +/- {results['pickscore']['std']:.4f}")
    if "score" in results.get("vendi", {}):
        print(f"  Vendi Diversity:  {results['vendi']['score']:.4f}")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

if __name__ == "__main__":
    main()
