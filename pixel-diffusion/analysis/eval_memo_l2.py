#!/usr/bin/env python3
"""Memorization by the nearest-neighbour ratio test of Gu et al. 2023 ("On memorization in diffusion
models", Sec. 2, after Yoon et al. 2023): a generated image x counts as memorized when its pixel-space
l2 distance to the nearest training image is below 1/3 of its distance to the second-nearest one.

Training uses x-flip augmentation, so each training image is represented by the closer of itself and
its mirror image (taking both as separate neighbours would make every memorized near-symmetric face
look "not memorized", because its own mirror image is the second neighbour).

For an Ambient-o model the search set must be the UNBLURRED originals of all training images (a model
can memorize a face it only saw blurred), so pass the full clean training set (train_clean_64).

Usage: python analysis/eval_memo_l2.py --gen_path DIR --train_path DIR --out_path JSON
"""
import argparse, json, os

import numpy as np
import torch
from PIL import Image


def load_dir(d, limit=None):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit:
        fs = fs[:limit]
    arr = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.uint8) for f in fs])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), fs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_path", required=True)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--ratio", type=float, default=1 / 3)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    dev = torch.device("cuda")

    gen, _ = load_dir(args.gen_path)
    train, names = load_dir(args.train_path)
    n_tr = train.shape[0]
    # Training set on the GPU in float32, both orientations (70k x 12288 x 4 B x 2 = 6.9 GB). TF32 off so the
    # matmul is true fp32: squared distances are differences of numbers in the thousands.
    torch.backends.cuda.matmul.allow_tf32 = False
    tr = (train.to(dev).float() / 255).flatten(1)                      # [N, D]
    trf = (train.flip(3).to(dev).float() / 255).flatten(1)              # mirror images
    tr_sq = (tr ** 2).sum(1)
    trf_sq = (trf ** 2).sum(1)

    d1s, d2s, nn_idx = [], [], []
    for i in range(0, gen.shape[0], args.chunk):
        g = (gen[i:i + args.chunk].to(dev).float() / 255).flatten(1)
        g_sq = (g ** 2).sum(1, keepdim=True)
        # squared distances to every image and to its mirror; keep the closer orientation per image
        d = (g_sq + tr_sq[None] - 2 * (g @ tr.T)).clamp_min(0)
        df = (g_sq + trf_sq[None] - 2 * (g @ trf.T)).clamp_min(0)
        d = torch.minimum(d, df).sqrt()
        top = d.topk(2, dim=1, largest=False)
        d1s.append(top.values[:, 0].cpu()); d2s.append(top.values[:, 1].cpu()); nn_idx.append(top.indices[:, 0].cpu())
    d1, d2, nn = torch.cat(d1s), torch.cat(d2s), torch.cat(nn_idx)
    ratio = d1 / d2.clamp_min(1e-12)
    memo = ratio < args.ratio
    out = {
        "n_gen": int(gen.shape[0]), "n_train": int(n_tr),
        "memorized_frac": float(memo.float().mean()),
        "ratio_threshold": args.ratio,
        "ratio_median": float(ratio.median()), "ratio_p05": float(ratio.quantile(0.05)),
        "nn_l2_median": float(d1.median()), "nn_l2_mean": float(d1.mean()),
        "n_unique_nn_of_memorized": int(nn[memo].unique().numel()),
        "space": "pixel l2 on [0,1] RGB 64x64, per training image min over {x, flip(x)}",
    }
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
