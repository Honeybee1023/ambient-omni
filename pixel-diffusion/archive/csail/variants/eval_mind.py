import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""Compute MIND (Monge Inception Distance) = Sliced Wasserstein on Inception features."""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os, glob, json, argparse

def get_inception(device):
    model = models.inception_v3(weights='DEFAULT', transform_input=False)
    # Replace final FC with identity to get 2048-dim pool features
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def extract_features(model, img_dir, transform, device, batch_size=64):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                   glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"  Extracting Inception features from {len(files)} images")
    all_feats = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = torch.stack([transform(Image.open(f).convert('RGB')) for f in batch_files]).to(device)
        feats = model(imgs)
        all_feats.append(feats.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"    {i+len(batch_files)}/{len(files)}")
    return np.concatenate(all_feats, axis=0)

def sliced_wasserstein(feats1, feats2, n_projections=1000, seed=0):
    """Compute sliced Wasserstein distance between two feature sets."""
    rng = np.random.RandomState(seed)
    d = feats1.shape[1]

    # Random projection directions (unit vectors)
    directions = rng.randn(n_projections, d).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    feats1 = feats1.astype(np.float64)
    feats2 = feats2.astype(np.float64)

    # Project and compute 1D Wasserstein for each direction
    proj1 = feats1 @ directions.T  # (n1, M)
    proj2 = feats2 @ directions.T  # (n2, M)

    # For equal sample sizes: sort and compute mean absolute difference
    # For unequal: use quantile matching
    n1, n2 = len(feats1), len(feats2)
    if n1 == n2:
        distances = []
        for m in range(n_projections):
            s1 = np.sort(proj1[:, m])
            s2 = np.sort(proj2[:, m])
            distances.append(np.mean(np.abs(s1 - s2)))
    else:
        # Quantile matching: interpolate to common grid
        n_quantiles = min(n1, n2)
        quantiles = np.linspace(0, 1, n_quantiles, endpoint=False) + 0.5 / n_quantiles
        distances = []
        for m in range(n_projections):
            s1 = np.sort(proj1[:, m])
            s2 = np.sort(proj2[:, m])
            q1 = np.quantile(s1, quantiles)
            q2 = np.quantile(s2, quantiles)
            distances.append(np.mean(np.abs(q1 - q2)))

    return float(np.mean(distances))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', required=True)
    parser.add_argument('--ref_path', default='/data/scratch/honjar/celeba_processed/holdout_64')
    parser.add_argument('--ref_cache', default='/data/scratch/honjar/celeba_processed/inception_holdout_feats.npz')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--n_projections', type=int, default=1000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model = get_inception(args.device)
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

    print(f"Computing MIND (SWD with {args.n_projections} projections)...")
    mind = sliced_wasserstein(gen_feats, ref_feats, n_projections=args.n_projections)
    print(f"\nMIND: {mind:.6f}")

    with open(args.out_path, 'w') as f:
        json.dump({"mind": mind, "n_projections": args.n_projections}, f, indent=2)
    print(f"Saved to {args.out_path}")

if __name__ == '__main__':
    main()
