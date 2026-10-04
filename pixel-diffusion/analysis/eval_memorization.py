#!/usr/bin/env python3
"""Two end-of-training memorization diagnostics for finished runs (no training).

1. Copying (Shah, Kalavasis, Klivans, Daras 2025, "Does generation require memorization?"): DINOv2 ViT-B/14
   cosine similarity of every generated sample to its nearest CLEAN training face. Reported: mean, 95th
   percentile, and the share of samples above 0.90 / 0.95. As a control, the same against an equal number of
   held-out faces: a sample can be close to a training face just by being a typical face, so the informative
   number is how much closer samples sit to the training faces than to fresh ones ("excess").
2. Generalization gap per noise level (our earlier memorization idea): denoising error of the final EMA network
   on the clean training faces vs the same number of held-out faces, at fixed noise levels and fixed noise draws.
   Ratio held-out / train > 1 means the model denoises its own training faces better than fresh ones there.

    python analysis/eval_memorization.py --runs mix4bo_true_022 mix4_c1_global72_sloan ...
Writes $AMBIENT_BASE/generated/memo_dyn_<run>_s0.json (skips runs that already have one).
"""
import argparse, glob, json, os, pickle, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = os.environ['AMBIENT_BASE']
sys.path.insert(0, os.path.join(BASE, 'ambient-omni', 'pixel-diffusion'))
import dnnlib  # noqa: E402

GEN = os.path.join(BASE, 'generated')
SIGMAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]


def load_images(paths):
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])).permute(0, 3, 1, 2)


@torch.no_grad()
def dino_feats(model, imgs, dev, bs=250):
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs].to(dev).float() / 255.0
        x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        out.append(F.normalize(model((x - mean) / std), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def denoise_mse(net, imgs, dev, gen, bs=250):
    """Mean squared error of D(x + sigma n) vs x, per sigma, images in [-1, 1]; 4 fixed noise draws."""
    res = []
    x0 = imgs.float() / 127.5 - 1
    for s in SIGMAS:
        tot, cnt = 0.0, 0
        for d in range(4):
            g = torch.Generator().manual_seed(1000 * d + 7)
            for i in range(0, len(x0), bs):
                x = x0[i:i + bs]
                n = torch.randn(x.shape, generator=g)
                xd = (x + s * n).to(dev)
                sig = torch.full((len(x),), s, device=dev)
                den = net(xd, sig, None).float().cpu()
                tot += ((den - x) ** 2).mean(dim=(1, 2, 3)).sum().item()
                cnt += len(x)
        res.append(tot / cnt)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--dataset', default='celeba_mix4_v1')
    ap.add_argument('--holdout', default=os.path.join(BASE, 'celeba_processed_v2b', 'holdout_64'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(0)

    # Reference sets: the 500 clean training faces and the first 500 held-out faces (fixed order).
    train_paths = sorted(glob.glob(os.path.join(BASE, 'annotated_datasets', a.dataset, 'b0_*')))
    hold_paths = sorted(glob.glob(os.path.join(a.holdout, '*.png')) + glob.glob(os.path.join(a.holdout, '*.jpg')))[:len(train_paths)]
    assert len(train_paths) == 500 and len(hold_paths) == 500, (len(train_paths), len(hold_paths))
    tr_img, ho_img = load_images(train_paths), load_images(hold_paths)

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github', trust_repo=True).to(dev).eval()
    tr_f, ho_f = dino_feats(dino, tr_img, dev), dino_feats(dino, ho_img, dev)

    for run in a.runs:
        out = os.path.join(GEN, f'memo_dyn_{run}_s0.json')
        if os.path.exists(out):
            print('skip', run); continue
        gdir = os.path.join(GEN, f'dyn_{run}_s0_5k_gen')
        ckpt = os.path.join(BASE, 'train_outputs', 'dyn_search', f'dyn_{run}_s0', 'network-snapshot-002000.pkl')
        if not (os.path.exists(os.path.join(gdir, '.complete')) and os.path.exists(ckpt)):
            print('not ready', run); continue
        gpaths = sorted(glob.glob(os.path.join(gdir, '*.png')))
        gf = dino_feats(dino, load_images(gpaths), dev)
        s_tr = (gf @ tr_f.T).max(dim=1).values.numpy()
        s_ho = (gf @ ho_f.T).max(dim=1).values.numpy()
        with dnnlib.util.open_url(ckpt, verbose=False) as f:
            net = pickle.load(f)['ema'].to(dev).eval()
        e_tr, e_ho = denoise_mse(net, tr_img, dev, None), denoise_mse(net, ho_img, dev, None)
        del net
        rec = dict(run=run, n_samples=len(gpaths),
                   nn_train_mean=float(s_tr.mean()), nn_train_p95=float(np.percentile(s_tr, 95)),
                   nn_train_gt90=float((s_tr > 0.90).mean()), nn_train_gt95=float((s_tr > 0.95).mean()),
                   nn_hold_mean=float(s_ho.mean()), nn_hold_gt90=float((s_ho > 0.90).mean()),
                   nn_excess=float(s_tr.mean() - s_ho.mean()),
                   sigmas=SIGMAS, mse_train=e_tr, mse_hold=e_ho,
                   gap_ratio=[h / t for h, t in zip(e_ho, e_tr)])
        json.dump(rec, open(out, 'w'), indent=1)
        print(run, f"excess {rec['nn_excess']:+.4f}  >0.95 {rec['nn_train_gt95']:.3f}  gap@sigma " +
              ' '.join(f'{s:g}:{r:.2f}' for s, r in zip(SIGMAS, rec['gap_ratio'])), flush=True)


if __name__ == '__main__':
    main()
