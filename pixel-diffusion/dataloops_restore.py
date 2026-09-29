#!/usr/bin/env python3
"""Ambient Dataloops restoration step, ported from the official code
(github.com/adrianrm99/ambient_dataloops, pixel-diffusion/restore.py, commit f687223).

What the official code does, and what this reproduces:
  - start from the loop-0 (Ambient-o) model and each corrupted image's annotation sigma_tn;
  - add fresh noise at sigma_tn and run the deterministic EDM Heun sampler (18 steps,
    rho=7, S_churn=0) from sigma_tn ALL THE WAY TO ZERO (the official call passes
    zeros as sigma_end; the paper's "stop at t/2^l" is not what the code does);
  - store the result as a new image annotated at sigma_tn / rho_ambient (rho_ambient=8
    in every official script) -- the loop-1 dataset;
  - clean images (sigma_tn == 0) are left untouched.
Loop 1 then trains a NEW model from scratch on that dataset (official train_loop1 scripts
pass no resume), with those annotations fixed for the run.

Differences, all forced by our data rather than chosen:
  - sigma_tn is read from the loop-0 run's annotations_processed.jsonl, i.e. exactly the
    per-image values that run trained with (our annotations store classifier
    probabilities, reduced at load time with --cls_ema_window=1);
  - restored images are written as PNG (our sources are JPEG; re-encoding to JPEG would
    add a second lossy pass the official CIFAR pipeline never has);
  - one GPU, no DDP. Per-image latents are seeded by the image's position in the file
    list, so the output is deterministic.

Usage (CSAIL, 1 GPU):
  python dataloops_restore.py --loop0_run <train_outputs/.../dyn_mix4_ambo_s0> \
      --dataset <annotated_datasets/celeba_mix4_ambo> --out <annotated_datasets/celeba_mix4_loop1>
"""
import argparse, json, os, pickle, re
import numpy as np
import torch
import dnnlib
from ambient_utils import save_image
from ambient_utils.dataset import SyntheticallyCorruptedImageFolderDataset


def edm_restorer(x, sigma_start, sigma_end, net, latents, class_labels=None, randn_like=torch.randn_like,
                 num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, add_noise=True):
    """Verbatim from the official restore.py (EDM Heun sampler with per-sample start/end)."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    is_clean_sample = sigma_start == 0
    sigma_start_clamp = torch.max(sigma_start, torch.full_like(sigma_start, sigma_min)).to(device=x.device, dtype=x.dtype)
    sigma_end_clamp = torch.max(sigma_end, torch.full_like(sigma_end, sigma_min)).to(device=x.device, dtype=x.dtype)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_start_clamp[None, :] ** (1 / rho) + step_indices[:, None] / (num_steps - 1)
               * (sigma_end_clamp[None, :] ** (1 / rho) - sigma_start_clamp[None, :] ** (1 / rho))) ** rho
    final_step = sigma_end.to(device=x.device, dtype=x.dtype)
    t_steps = torch.cat([net.round_sigma(t_steps), final_step[None, :]], 0)
    num_steps = len(t_steps) - 1
    x_next = x + latents.to(torch.float64) * t_steps[0, :, None, None, None] * float(add_noise == True)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = torch.where((S_min <= t_cur) & (t_cur <= S_max), min(S_churn / num_steps, np.sqrt(2) - 1), 0.0)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt()[:, None, None, None] * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next[:, None, None, None]
            x_next = x_hat + (t_next - t_hat)[:, None, None, None] * (0.5 * d_cur + 0.5 * d_prime)
    x_next[is_clean_sample] = x[is_clean_sample].to(dtype=x_next.dtype)
    return x_next


def read_processed(path):
    """annotations_processed.jsonl lines look like  g05_000123.jpg: (0.84, 0.0)"""
    out = {}
    for line in open(path):
        m = re.match(r'(.+?): \(([-\d.eE+]+), ([-\d.eE+]+)\)', line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop0_run', required=True)
    ap.add_argument('--dataset', required=True, help='the dataset loop 0 trained on')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rho_ambient', type=float, default=8.0)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--snapshot', default='network-snapshot-002000.pkl')
    ap.add_argument('--max_batches', type=int, default=None, help='smoke test: stop early')
    args = ap.parse_args()
    device = torch.device('cuda')

    with dnnlib.util.open_url(os.path.join(args.loop0_run, args.snapshot)) as f:
        net = pickle.load(f)['ema'].to(device).eval()
    sigma_tn = read_processed(os.path.join(args.loop0_run, 'annotations_processed.jsonl'))
    opts = json.load(open(os.path.join(args.loop0_run, 'training_options.json')))['dataset_kwargs']
    # As the official restore.py (which uses noise_configs.inference.identity; ours is noise_configs.identity): identity corruptions, and drop the two keys the dataset class rejects.
    import importlib
    opts['corruptions_dict'] = importlib.import_module('noise_configs.identity').corruptions_dict
    opts.pop('noise_config', None); opts.pop('dataset_keep_percentage', None)
    opts.update(path=args.dataset, corruption_probability=0.0)
    ds = SyntheticallyCorruptedImageFolderDataset(**opts)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=8)

    os.makedirs(args.out, exist_ok=True)
    ann_path = os.path.join(args.out, 'annotations.jsonl')
    done = set()
    if os.path.exists(ann_path):                                   # resumable
        done = {json.loads(l)['filename'] for l in open(ann_path)}
    n_seen, n_restored = 0, 0
    with open(ann_path, 'a') as fa:
        for item in loader:
            names = [os.path.basename(n) for n in item['filename']]
            idx0 = n_seen
            n_seen += len(names)
            todo = [i for i, n in enumerate(names) if n not in done and os.path.splitext(n)[0] + '.png' not in done]
            if not todo:
                continue
            s = torch.tensor([sigma_tn[names[i]] for i in todo], dtype=torch.float64, device=device)
            clean = [i for i, v in zip(todo, s.tolist()) if v == 0.0]
            for i in clean:                                         # untouched: link the original
                src = os.path.realpath(os.path.join(args.dataset, names[i]))
                if not os.path.lexists(os.path.join(args.out, names[i])):
                    os.symlink(src, os.path.join(args.out, names[i]))
                fa.write(json.dumps({'filename': names[i], 'sigma_min': 0.0, 'sigma_max': 0.0}) + '\n')
            corrupt = [i for i in todo if sigma_tn[names[i]] > 0.0]
            if corrupt:
                x = item['image'][corrupt].to(device).to(torch.float64)
                st = torch.tensor([sigma_tn[names[i]] for i in corrupt], dtype=torch.float64, device=device)
                g = torch.Generator(device=device)
                lat = torch.stack([torch.randn(x.shape[1:], generator=g.manual_seed(idx0 + i), device=device,
                                               dtype=torch.float64) for i in corrupt])
                with torch.no_grad():
                    y = edm_restorer(x, st, torch.zeros_like(st), net, lat).to(torch.float32)
                for k, i in enumerate(corrupt):
                    new = os.path.splitext(names[i])[0] + '.png'
                    save_image(y[k].cpu(), os.path.join(args.out, new))
                    fa.write(json.dumps({'filename': new, 'sigma_min': st[k].item() / args.rho_ambient,
                                         'sigma_max': 0.0}) + '\n')
                n_restored += len(corrupt)
            fa.flush()
            print(f'{n_seen}/{len(ds)} seen, {n_restored} restored this session', flush=True)
            if args.max_batches and n_seen >= args.max_batches * args.batch:
                break
    print('done')


if __name__ == '__main__':
    main()
