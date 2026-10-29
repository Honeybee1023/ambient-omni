# ImageNet-512 runbook: per-bucket noise schedules vs. Ambient-o

Everything runs from `latent-diffusion/`. Paths are shell variables; set them for the cluster:

```bash
IMNET=<ImageNet-1k train dir: one folder per synset>      # 1,281,167 JPEGs
W=<work dir, ~300 GB free>                                # datasets, bucket file, runs
GPUS=4                                                    # GPUs per node
```

## 0. What is being compared

| Run | Data used as clean | Everything else | Length (images) |
|---|---|---|---|
| **Ours** | top 10% per class by Q-Align | 3 k-means buckets, each with its own step schedule (picked by the search) | 2048 Mi (EDM2-XS default) |
| **Ambient-o** | the same top 10% per class | every image at σ_min = 0.2, for the whole run | 1.8 × 2048 Mi (same total compute as search + our run) |

The search costs 8 short runs × 1/10 length = 0.8 of a full run, which is why Ambient-o gets 1.8×.

Model: EDM2 `edm2-img512-xs`, class-conditional, SD-VAE latents, batch 2048, lr 0.012.

**Cost (measured at batch 2048).**

| GPUs | Speed | One full run | All 3.6 runs |
|---|---|---|---|
| 8 × GB200 (2 nodes) | 0.131 s per 1000 images | ≈ 78 h on 8 GPUs ≈ 625 GPU-hours | ≈ 2,250 GPU-hours |
| 1 × GH200 | 1.34 s per 1000 images | ≈ 800 GPU-hours | ≈ 2,900 GPU-hours |
| 2 × L40S | 2.19 s per 1000 images | ≈ 2,600 GPU-hours | ≈ 9,400 GPU-hours |

One short search run on 2 GB200 nodes takes ≈ 8 h.

## 1. Environment

Two setups have passed the smoke test (section 7):

- x86 + L40S/H100: Python 3.9, torch 2.6.0 + CUDA 12.4 (exact pip list: `edm2_freeze.txt`).
- ARM (aarch64) + GH200/GB200: Python 3.12 in a plain venv, torch 2.9.1 + CUDA 12.8 (`--index-url
  https://download.pytorch.org/whl/cu128`; it has kernels for sm_90 and sm_100), scipy 1.13.1 and numpy 2.0.2
  (pinned to the x86 versions so FID numerics match), plus `ambient_utils` from
  `git+https://github.com/giannisdaras/ambient-utils.git@9cb74d288fcdf40485f548a2e5f70d54891df676` (`--no-deps`)
  and its imports (`imagecorruptions webdataset braceexpand open_clip_torch seaborn pandas matplotlib piq
  opencv-python-headless`).

Internet is needed once for: NVIDIA's guide network, DINOv2 (pinned commit, via torch.hub), Inception weights, the
SD VAE. Without internet, pre-fetch into `$TORCH_HOME`, `$HF_HOME` and `$DNNLIB_CACHE_DIR` and copy them over.

The SD VAE always runs without TF32 (`training/encoders.py`): with PyTorch's default TF32 convolutions, some latent
means came out wrong by up to ~10 on Hopper GPUs.

## 2. Dataset (about 1 h on 8 GPUs)

The order must be the sorted relative file paths (dataset_tool's order); every index below depends on it. Check:

```bash
python scripts/imagenet_filelist.py $IMNET $W/filelist.txt
wc -l $W/filelist.txt     # must be 1281167
md5sum $W/filelist.txt    # must be 726caf227020953c02c5b6e3a26467d6 (the bucket file's order)
```

Hugging Face's `ILSVRC/imagenet-1k` (parquet, gated) holds the original JPEG bytes; unpacked to
`<synset>/<synset>_<id>.JPEG` it passes this check.

`scripts/imagenet_prepare.py` does what `dataset_tool.py convert` + `encode` do, in parallel, without the ~1 TB
uncompressed-PNG archive, and stops on a load error instead of skipping the image (a skip would shift every later
index):

```bash
for k in 0 1 2 3 4 5 6 7; do   # one GPU each (e.g. srun, one task per GPU)
  python scripts/imagenet_prepare.py encode --root $IMNET --outdir $W/sd_shards --shard $k --nshards 8
done
python scripts/imagenet_prepare.py merge --root $IMNET --outdir $W/sd_shards --nshards 8 --dest $W/img512-sd.zip
# 512x512 PNGs of the held-out clean images only (the picking reference)
python scripts/imagenet_prepare.py pngs --root $IMNET --mask $W/buckets.npz:heldout_clean_short --dest $W/img512_heldout.zip
```

`imagenet_prepare.py check` compares it with dataset_tool on a 2000-image slice. Result: pixels identical; latents
within 0.53 (mean 4e-4) of an L40S encode.

## 3. Bucket file (done ahead of time; just copy it)

`imagenet_buckets.npz` is built from Q-Align scores of every training image (`scripts/qalign_score.py`, then
`scripts/imagenet_buckets.py`). It holds, per dataset index: `bucket` (0 = clean, 1–3 = k-means buckets, best to
worst), `score`, `label`, `subset_short` (stratified 1/10 for the search) and `heldout_clean_short` (clean images
not in the subset: the FID reference for picking). Copy it to `$W/buckets.npz`.

| Bucket | Images | Share | Mean Q-Align |
|---|---|---|---|
| 0 clean (top 10% per class) | 128,118 | 10% | 4.39 |
| 1 | 631,279 | 49% | 3.93 |
| 2 | 358,482 | 28% | 3.20 |
| 3 | 163,288 | 13% | 2.15 |

Short-run subset: 128,117 images. Held-out clean: 115,300 images. File-list md5 the bucket file was built against:
`726caf227020953c02c5b6e3a26467d6`. If the file-list check above fails,
re-score with `qalign_score.py` (about 11 GPU-hours on L40S) instead of using the shipped file.

## 4. Reference statistics

```bash
# Picking among short runs: held-out clean images only
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py ref --data=$W/img512_heldout.zip --dest=$W/ref_heldout.pkl
# Final comparison: NVIDIA's standard ImageNet-512 reference (comparable to published numbers)
#   https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl
```

## 5. Training

Every run goes through `scripts/run_train.sh` (one node) or `scripts/launch_slurm.sh` (several nodes; one
`run_train.sh` per node, joined by torchrun rendezvous). A stop request makes all ranks finish the step, save a
checkpoint and exit with code 3; rerunning the same command resumes. Under Slurm:

```bash
#SBATCH --signal=B:USR1@900
bash scripts/launch_slurm.sh $GPUS <args>; rc=$?; [ $rc -eq 3 ] && scontrol requeue $SLURM_JOB_ID
```

Several runs can share one allocation: `RUN_NODES=2 NODE_OFFSET=4 MASTER_PORT=29502 bash scripts/launch_slurm.sh ...`
puts a run on nodes 4–5 of the job. On several nodes the stop request goes through `<outdir>/STOP`, not a signal
through srun (srun signals every process on the node, torchrun included, which dies).

Common arguments:

```bash
COMMON="--preset=edm2-img512-xs --data=$W/img512-sd.zip --bucket_path=$W/buckets.npz --batch-gpu=256 --checkpoint=16Mi"
```

### 5a. Search: 8 short runs (1/10 of the data, 1/10 of the length)

The LR schedule shrinks with the run: warm-up 10 → 1 Mimg, decay 70000 → 7000 batches.
Lengths must be whole batches (2048 images) and so must the 1/32 snapshot stride, hence the odd numbers:
209728Ki = 104864 batches = 0.10001 × 2048Mi; 3774912Ki = 1.80002 × 2048Mi.

```bash
python scripts/imagenet_search_design.py $W/search      # writes s00.json .. s07.json + design.json
for i in 00 01 02 03 04 05 06 07; do
  bash scripts/launch_slurm.sh $GPUS $COMMON --outdir=$W/runs/search/s$i --subset=short \
       --duration=209728Ki --schedule_horizon=209728Ki --decay=7000 --rampup_mimg=1 --snapshot=6554Ki \
       --bucket_schedule=@$W/search/s$i.json --seed=0
done
```

### 5b. Pick

Screen every snapshot × EMA (0.050, 0.100) of every run by FID-10k against `ref_heldout.pkl`; score the top 3 per
run at 50k; the winning schedule is the run with the best FID-50k.

```bash
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py gen --net=<snapshot.pkl> --ref=$W/ref_heldout.pkl --num=10000
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py gen --net=<snapshot.pkl> --ref=$W/ref_heldout.pkl --num=50000
```

### 5c. Full runs

```bash
# Ours: the winning schedule, full data, full length
bash scripts/launch_slurm.sh $GPUS $COMMON --outdir=$W/runs/ours_full --bucket_schedule=@$W/search/s<best>.json \
     --duration=2048Mi --snapshot=64Mi --seed=0
# Ambient-o: same clean set, sigma_min 0.2 for every other image, 1.8x length
echo '{"type":"fixed","sigma_min":{"1":0.2,"2":0.2,"3":0.2}}' > $W/ambo.json
bash scripts/launch_slurm.sh $GPUS $COMMON --outdir=$W/runs/ambo_full --bucket_schedule=@$W/ambo.json \
     --duration=3774912Ki --snapshot=117966Ki --seed=0
```

## 6. Final evaluation

For each full run, the best snapshot × EMA by FID-50k and FD-DINOv2 against `img512.pkl`, unguided and with
NVIDIA's XS unconditional guide:

```bash
python generate_images.py --net=<snapshot.pkl> \
    --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.045.pkl \
    --guidance=1.4 --seeds=0-49999 --outdir=$W/gen/<run>
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py calc --images=$W/gen/<run> --ref=img512.pkl
```

## 7. Smoke test before anything long (about 30 min)

A 10-class × 200-image slice: train our schedule on the short subset, stop and resume, train the Ambient-o
baseline, build held-out reference stats, score FID + FD-DINOv2, sample with guidance. Passed on 1 × L40S, 1 × H100,
2 × L40S (torch 2.6.0), 1 × GH200 and 2 nodes × 4 GB200 (multi-node stop and resume; torch 2.9.1). On the full data,
two packed runs (ours on the short subset, Ambient-o) also stopped, resumed and scored FID-10k against the held-out
reference.

## Open choices

- Ambient-o at 1.8× keeps the EDM2 LR schedule (inverse-sqrt decay past 70000 batches); no re-tuning.
- Seeds: one seed per full run first; more if the gap is close.
- Snapshot stride: 1/32 of the run.
