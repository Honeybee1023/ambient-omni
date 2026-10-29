"""Parallel ImageNet-512 preparation, equivalent to `dataset_tool.py convert --transform=center-crop-dhariwal
--resolution=512x512` followed by `dataset_tool.py encode`, without writing the ~1 TB uncompressed-PNG archive.

Each image is loaded exactly as dataset_tool does (PIL -> RGB -> the same center-crop-dhariwal transform). PNG is
lossless, so encoding the crop in memory gives the VAE the same input as encoding the PNG. Differences from
dataset_tool: VAE batches > 1 (float-level differences only), and a load error stops the run instead of silently
skipping the image (a skip would shift every later index and break the bucket file).

Index i = i-th path of the sorted file list (scripts/imagenet_filelist.py), label = sorted synset index; both as in
dataset_tool, so indices line up with the Q-Align bucket file.

Commands:
  encode  --root R --outdir D --shard k --nshards K   latents of indices in shard k -> D/sd_shard_<k>.zip
  merge   --outdir D --nshards K --dest OUT.zip        shards -> one EDM2 latent archive with dataset.json
  pngs    --root R --mask NPZ:KEY --dest OUT.zip       512x512 PNGs (dataset_tool format) of the masked images only
  check   --root R --sd SD.zip --png PNG.zip --classes-every 100 --per-class 200
          rebuild a dataset_tool slice in memory and compare it with archives made by dataset_tool
"""
import argparse, io, json, os, re, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # dataset_tool, training/
import numpy as np
import PIL.Image
import torch

N_IMAGENET = 1281167


def file_list(root):
    from PIL import Image
    ext = set(Image.registered_extensions())
    files = sorted(os.path.join(root, d, f) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
                   for f in os.listdir(os.path.join(root, d)) if os.path.splitext(f)[1].lower() in ext)
    rel = [os.path.relpath(f, root) for f in files]
    synsets = sorted({r.split(os.sep)[0] for r in rel})
    lab = {s: i for i, s in enumerate(synsets)}
    return files, np.array([lab[r.split(os.sep)[0]] for r in rel], np.int64)


_TF = None
def load_crop(path):
    global _TF
    if _TF is None:
        import dataset_tool
        _TF = dataset_tool.make_transform('center-crop-dhariwal', 512, 512)
    img = np.array(PIL.Image.open(path).convert('RGB'))   # as dataset_tool.open_image_folder
    out = _TF(img)
    if out is None or out.shape != (512, 512, 3):
        raise RuntimeError(f'bad crop for {path}')
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, files, idx): self.files, self.idx = files, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = int(self.idx[j]); return i, torch.from_numpy(load_crop(self.files[i])).permute(2, 0, 1)


def encode_indices(files, idx, batch, workers, device='cuda'):
    from training.encoders import StabilityVAEEncoder
    vae = StabilityVAEEncoder(batch_size=batch); vae.init(torch.device(device))
    dl = torch.utils.data.DataLoader(Crops(files, idx), batch_size=batch, num_workers=workers, shuffle=False)
    with torch.no_grad():
        for ii, x in dl:
            ms = vae.encode_pixels(x.to(device)).cpu()
            for i, m in zip(ii.tolist(), ms):
                f = io.BytesIO(); np.save(f, m.numpy()); yield i, f.getvalue()


def sd_name(i): s = f'{i:08d}'; return f'{s[:5]}/img-mean-std-{s}.npy'
def png_name(i): s = f'{i:08d}'; return f'{s[:5]}/img{s}.png'
def png_bytes(arr):
    b = io.BytesIO(); PIL.Image.fromarray(arr).save(b, format='png', compress_level=0, optimize=False); return b.getvalue()


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('encode'); e.add_argument('--root', required=True); e.add_argument('--outdir', required=True)
    e.add_argument('--shard', type=int, required=True); e.add_argument('--nshards', type=int, required=True)
    e.add_argument('--batch', type=int, default=32); e.add_argument('--workers', type=int, default=16)
    m = sub.add_parser('merge'); m.add_argument('--outdir', required=True); m.add_argument('--nshards', type=int, required=True)
    m.add_argument('--dest', required=True); m.add_argument('--root', required=True)
    p = sub.add_parser('pngs'); p.add_argument('--root', required=True); p.add_argument('--mask', required=True)
    p.add_argument('--dest', required=True); p.add_argument('--workers', type=int, default=64)
    c = sub.add_parser('check'); c.add_argument('--root', required=True); c.add_argument('--sd', required=True)
    c.add_argument('--png', required=True); c.add_argument('--classes-every', type=int, default=100)
    c.add_argument('--per-class', type=int, default=200)
    a = ap.parse_args()

    files, labels = file_list(a.root)
    if a.cmd != 'check':
        assert len(files) == N_IMAGENET, len(files)

    if a.cmd == 'encode':
        idx = np.array_split(np.arange(len(files)), a.nshards)[a.shard]
        os.makedirs(a.outdir, exist_ok=True)
        tmp = f'{a.outdir}/sd_shard_{a.shard:03d}.zip.tmp'
        n = 0
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as zf:
            for i, b in encode_indices(files, idx, a.batch, a.workers):
                zf.writestr(sd_name(i), b); n += 1
                if n % 20000 == 0: print(f'shard {a.shard}: {n}/{len(idx)}', flush=True)
        assert n == len(idx), (n, len(idx))
        os.replace(tmp, tmp[:-4]); print(f'shard {a.shard}: done, {n} images ({idx[0]}..{idx[-1]})')

    elif a.cmd == 'merge':
        tmp = a.dest + '.tmp'; n = 0
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as out:
            for k in range(a.nshards):
                with zipfile.ZipFile(f'{a.outdir}/sd_shard_{k:03d}.zip') as z:
                    names = sorted(z.namelist())
                    for nm in names:
                        assert nm == sd_name(n), (nm, n); out.writestr(nm, z.read(nm)); n += 1
                print(f'merged shard {k}: total {n}', flush=True)
            assert n == N_IMAGENET, n
            out.writestr('dataset.json', json.dumps({'labels': [[sd_name(i), int(labels[i])] for i in range(n)],
                                                     'corruption_labels': [[sd_name(i), False] for i in range(n)]}))
        os.replace(tmp, a.dest); print('merge done', n)

    elif a.cmd == 'pngs':
        path, key = a.mask.rsplit(':', 1); idx = np.flatnonzero(np.load(path)[key])
        from multiprocessing import Pool
        tmp = a.dest + '.tmp'
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as zf, Pool(a.workers) as pool:
            for n, (i, b) in enumerate(pool.imap(_png_job, [(files[i], i) for i in idx], chunksize=64)):
                zf.writestr(png_name(i), b)
                if (n + 1) % 20000 == 0: print(f'pngs {n + 1}/{len(idx)}', flush=True)
            zf.writestr('dataset.json', json.dumps({'labels': [[png_name(i), int(labels[i])] for i in idx]}))
        os.replace(tmp, a.dest); print('pngs done', len(idx))

    elif a.cmd == 'check':
        syn = sorted({os.path.relpath(f, a.root).split(os.sep)[0] for f in files})[::a.classes_every]
        idx = [i for s in syn for i in [j for j, f in enumerate(files) if os.path.relpath(f, a.root).split(os.sep)[0] == s][:a.per_class]]
        print(f'slice: {len(idx)} images from {len(syn)} classes')
        with zipfile.ZipFile(a.png) as zp:
            bad = sum(not np.array_equal(load_crop(files[i]), np.array(PIL.Image.open(io.BytesIO(zp.read(png_name(k))))))
                      for k, i in enumerate(idx))
        print(f'pixels vs dataset_tool img512.zip: {len(idx) - bad}/{len(idx)} identical')
        with zipfile.ZipFile(a.sd) as zs:
            ref = np.stack([np.load(io.BytesIO(zs.read(sd_name(k)))) for k in range(len(idx))])
        got = np.stack([np.load(io.BytesIO(b)) for _, b in sorted(encode_indices(files, np.array(idx), 32, 16))])
        d = np.abs(got - ref)
        print(f'latents vs dataset_tool sd.zip: max abs diff {d.max():.2e}, mean {d.mean():.2e}, ref scale {np.abs(ref).mean():.2e}')


def _png_job(args):
    path, i = args
    return i, png_bytes(load_crop(path))


if __name__ == '__main__':
    main()
