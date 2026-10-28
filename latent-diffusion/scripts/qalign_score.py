# Q-Align quality score for every ImageNet-1k train image, in EDM2 dataset_tool order
# (sorted relative paths -> global index; label = sorted synset index).
# Each image gets the 512x512 center-crop-dhariwal crop the model trains on.
# Usage: qalign_score.py <imagenet_train_dir> <filelist> <outdir> <shard> <nshards>
#   filelist: relative paths in dataset_tool order (scripts/imagenet_filelist.py); run shards 0..nshards-1 as an array.
# Needs: transformers==4.37.2 (Q-Align's remote code), model q-future/one-align (~16 GB fp16), one GPU (not B200 with
# torch < 2.7). Throughput on one L40S: ~33 img/s (1.28M images ~ 11 GPU-hours).
# Saves chunk files <outdir>/chunk_<start>.npz (index, score, logits[5]); finished chunks are skipped on restart.
import os, sys, time, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, CLIPImageProcessor
ROOT, LIST, OUT, SH, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
CH, BS = 4096, 64
def center_crop_dhariwal(img, res=512):
    while min(*img.size) >= 2 * res:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    s = res / min(*img.size)
    img = img.resize(tuple(round(x * s) for x in img.size), resample=Image.BICUBIC)
    a = np.array(img); y = (a.shape[0] - res) // 2; x = (a.shape[1] - res) // 2
    return Image.fromarray(a[y:y + res, x:x + res])
paths = [l.rstrip('\n') for l in open(LIST)]; N = len(paths); assert N == 1281167, N
proc = CLIPImageProcessor.from_pretrained('q-future/one-align')
class DS(Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        img = center_crop_dhariwal(Image.open(f'{ROOT}/{paths[s.idx[i]]}').convert('RGB'))
        return proc.preprocess(img, return_tensors='pt')['pixel_values'][0], s.idx[i]
model = AutoModelForCausalLM.from_pretrained('q-future/one-align', trust_remote_code=True,
                                             torch_dtype=torch.float16, device_map='cuda:0').eval()
os.makedirs(OUT, exist_ok=True)
starts = list(range(0, N, CH))[SH::NSH]; t0 = time.time(); done = 0
for st in starts:
    fn = f'{OUT}/chunk_{st:07d}.npz'
    if os.path.exists(fn): continue
    idx = list(range(st, min(st + CH, N))); I, S, L = [], [], []
    for x, ii in DataLoader(DS(idx), batch_size=BS, num_workers=14, pin_memory=True):
        r = model.score(None, image_tensor=x.half().cuda(non_blocking=True), return_dict=True)
        I.append(ii.numpy()); S.append(r['scores'].float().cpu().numpy()); L.append(r['logits'].float().cpu().numpy())
    np.savez(fn + '.tmp.npz', index=np.concatenate(I), score=np.concatenate(S), logits=np.concatenate(L))
    os.replace(fn + '.tmp.npz', fn); done += len(idx)
    print(f'chunk {st} done | {done} imgs {time.time() - t0:.0f}s {done / (time.time() - t0):.1f} img/s', flush=True)
print('SHARD DONE', SH, flush=True)
