"""Compute DINO-FD (Frechet Distance in DINOv2 feature space)."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os, glob, json, argparse
from scipy import linalg

DINOV2_LOCAL = f"{AMBIENT_BASE}/.cache/torch/hub/facebookresearch_dinov2_main"

def get_model(device):
    model = torch.hub.load(DINOV2_LOCAL, 'dinov2_vitb14', source='local')
    model.eval().to(device)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def extract_features(model, img_dir, transform, device, batch_size=64):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                   glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"  Extracting features from {len(files)} images in {img_dir}")
    all_feats = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = torch.stack([transform(Image.open(f).convert('RGB')) for f in batch_files]).to(device)
        feats = model(imgs)
        all_feats.append(feats.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"    {i+len(batch_files)}/{len(files)}")
    return np.concatenate(all_feats, axis=0)

def compute_fd(feats1, feats2):
    mu1, sigma1 = feats1.mean(0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', required=True)
    parser.add_argument('--ref_path', default=f'{AMBIENT_BASE}/celeba_processed/holdout_64')
    parser.add_argument('--ref_cache', default=f'{AMBIENT_BASE}/celeba_processed/dino_holdout_feats.npz')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model = get_model(args.device)
    transform = get_transform()

    if os.path.exists(args.ref_cache):
        print(f"Loading cached reference features from {args.ref_cache}")
        ref_feats = np.load(args.ref_cache)['features']
    else:
        print("Computing reference features (will cache)...")
        ref_feats = extract_features(model, args.ref_path, transform, args.device)
        np.savez(args.ref_cache, features=ref_feats)
        print(f"Cached to {args.ref_cache}")

    gen_feats = extract_features(model, args.gen_path, transform, args.device)
    fd = compute_fd(gen_feats, ref_feats)
    print(f"\nDINO-FD: {fd:.4f}")

    with open(args.out_path, 'w') as f:
        json.dump({"dino_fd": fd}, f, indent=2)
    print(f"Saved to {args.out_path}")

if __name__ == '__main__':
    main()
