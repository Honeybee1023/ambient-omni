"""Compute denoising validation loss on holdout images."""
import torch
import numpy as np
from PIL import Image
import pickle
import os, glob, json, argparse, sys

sys.path.insert(0, '/data/scratch/honjar/ambient-omni/pixel-diffusion')
import dnnlib
from torch_utils import persistence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--holdout_dir', default='/data/scratch/honjar/celeba_processed/holdout_64')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f"Loading {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)
    net = data['ema'].to(device).eval()

    # Load holdout images
    files = sorted(glob.glob(os.path.join(args.holdout_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.holdout_dir, "*.png")))
    if args.max_images:
        files = files[:args.max_images]
    print(f"Loading {len(files)} holdout images...")

    images = []
    for f in files:
        img = np.array(Image.open(f).convert('RGB')).astype(np.float32)
        img = img / 127.5 - 1  # [-1, 1] normalization (standard EDM)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        images.append(img)
    images = np.stack(images)
    print(f"Loaded: shape={images.shape}, range=[{images.min():.1f}, {images.max():.1f}]")

    # Fixed seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    P_mean, P_std, sigma_data = -1.2, 1.2, 0.5

    # Sample one sigma per image from training distribution
    log_sigmas = np.random.normal(P_mean, P_std, size=len(images)).astype(np.float32)
    sigmas = np.exp(log_sigmas)

    weighted_losses = []
    unweighted_losses = []

    print("Computing validation loss...")
    with torch.no_grad():
        for i in range(0, len(images), args.batch_size):
            x0 = torch.tensor(images[i:i+args.batch_size]).to(device)
            sigma = torch.tensor(sigmas[i:i+args.batch_size]).to(device)
            sigma_4d = sigma[:, None, None, None]

            noise = torch.randn_like(x0)
            x_t = x0 + sigma_4d * noise

            x0_pred = net(x_t, sigma_4d, None)

            mse = ((x0_pred - x0) ** 2).mean(dim=(1, 2, 3))
            edm_weight = (sigma_data**2 + sigma**2) / (sigma**2 * sigma_data**2)

            weighted_losses.append((edm_weight * mse).cpu().numpy())
            unweighted_losses.append(mse.cpu().numpy())

            if (i // args.batch_size) % 50 == 0:
                print(f"  {i + x0.shape[0]}/{len(images)}")

    weighted = np.concatenate(weighted_losses)
    unweighted = np.concatenate(unweighted_losses)

    result = {
        "weighted_loss_mean": float(weighted.mean()),
        "weighted_loss_std": float(weighted.std()),
        "unweighted_mse_mean": float(unweighted.mean()),
        "unweighted_mse_std": float(unweighted.std()),
        "n_images": len(images),
    }

    print(f"\nWeighted loss: {result['weighted_loss_mean']:.6f} +/- {result['weighted_loss_std']:.6f}")
    print(f"Unweighted MSE: {result['unweighted_mse_mean']:.6f} +/- {result['unweighted_mse_std']:.6f}")

    with open(args.out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out_path}")

if __name__ == '__main__':
    main()
