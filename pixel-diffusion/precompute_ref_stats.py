import argparse
import ambient_utils
import numpy as np

parser = argparse.ArgumentParser(description='Precompute reference stats for FID evaluation')
parser.add_argument("--ref_path", type=str, required=True, help="Path to reference images")
parser.add_argument("--out_path", type=str, required=True, help="Path to save .npz file with mu and sigma")
parser.add_argument("--batch_size", type=int, default=64)

def main(args):
    mu, sigma, _ = ambient_utils.eval_utils.calculate_inception_stats(args.ref_path, max_batch_size=args.batch_size)
    np.savez(args.out_path, mu=mu, sigma=sigma)
    print(f"Saved reference stats to {args.out_path}")
    print(f"  mu shape: {mu.shape}, sigma shape: {sigma.shape}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
