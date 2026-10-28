# ImageNet-512 runbook: per-bucket noise schedules vs. Ambient-o

Everything runs from `latent-diffusion/`. Paths are shell variables; set them for the cluster:

```bash
IMNET=<ImageNet-1k train dir: one folder per synset>      # 1,281,167 JPEGs
W=<work dir, ~400 GB free>                                # datasets, bucket file, runs
GPUS=8                                                    # GPUs per node
```

## 0. What is being compared

| Run | Data used as clean | Everything else | Length (images) |
|---|---|---|---|
| **Ours** | top 10% per class by Q-Align | 3 k-means buckets, each with its own step schedule (picked by the search) | 2048 Mi (EDM2-XS default) |
| **Ambient-o** | the same top 10% per class | every image at σ_min = 0.2, for the whole run | 1.8 × 2048 Mi (same total compute as search + our run) |

The search costs 8 short runs × 1/10 length = 0.8 of a full run, which is why Ambient-o gets 1.8×.

**Cost.** Measured: 2.19 s per 1000 images on 2 × L40S at batch 2048, so one full XS run ≈ 2,600 L40S-hours.
All runs together are 3.6 full runs ≈ 9,400 L40S-hours. H100/H200 were not measured; at a rough 2.5–3× an L40S that
is ≈ 3,100–3,800 GPU-hours, or about 4–5 days on 32 GPUs. Measure on the first hour of a real run before planning.

Model: EDM2 `edm2-img512-xs`, class-conditional, SD-VAE latents, batch 2048, lr 0.012.

## 1. Environment

Python 3.9, torch 2.6.0 + CUDA 12.4 (exact pip list: `edm2_freeze.txt`, ships with this runbook). B200 GPUs need
torch ≥ 2.7; on those, install the newest torch instead and run the smoke test (section 7) before anything long.
Internet is needed once for: NVIDIA's guide network, DINOv2 (pinned commit, via torch.hub), Inception weights.
Without internet, pre-fetch into `$TORCH_HOME` and `$DNNLIB_CACHE_DIR` and copy them over.

## 2. Dataset (once, about 1 day of CPU + a few GPU-hours)

```bash
python dataset_tool.py convert --source=$IMNET --dest=$W/img512.zip --resolution=512x512 --transform=center-crop-dhariwal
python dataset_tool.py encode  --source=$W/img512.zip --dest=$W/img512-sd.zip
```

If the cluster already has EDM2's `img512.zip` / `img512-sd.zip`, use those. The order must be the sorted relative
file paths (dataset_tool's order); every index below depends on it. Check:

```bash
python scripts/imagenet_filelist.py $IMNET $W/filelist.txt
wc -l $W/filelist.txt     # must be 1281167
md5sum $W/filelist.txt    # must be 726caf227020953c02c5b6e3a26467d6 (the bucket file's order)
```

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
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py ref --data=$W/img512.zip \
    --subset=$W/buckets.npz:heldout_clean_short --dest=$W/ref_heldout.pkl
# Final comparison: NVIDIA's standard ImageNet-512 reference (comparable to published numbers)
#   https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl
```

## 5. Training

Every run goes through `scripts/run_train.sh`: a stop signal (USR1/TERM) makes all ranks finish the step, save a
checkpoint and exit with code 3; rerunning the same command resumes. Under Slurm:

```bash
#SBATCH --signal=B:USR1@600
bash scripts/run_train.sh $GPUS <args>; rc=$?; [ $rc -eq 3 ] && scontrol requeue $SLURM_JOB_ID
```

Common arguments:

```bash
COMMON="--preset=edm2-img512-xs --data=$W/img512-sd.zip --bucket_path=$W/buckets.npz --checkpoint=16Mi"
```

### 5a. Search: 8 short runs (1/10 of the data, 1/10 of the length)

The LR schedule shrinks with the run: warm-up 10 → 1 Mimg, decay 70000 → 7000 batches.
Lengths must be whole batches (2048 images) and so must the 1/32 snapshot stride, hence the odd numbers:
209728Ki = 104864 batches = 0.10001 × 2048Mi; 3774912Ki = 1.80002 × 2048Mi.

```bash
python scripts/imagenet_search_design.py $W/search      # writes s00.json .. s07.json + design.json
for i in 00 01 02 03 04 05 06 07; do
  bash scripts/run_train.sh $GPUS $COMMON --outdir=$W/runs/short_s$i --subset=short \
       --duration=209728Ki --schedule_horizon=209728Ki --decay=7000 --rampup_mimg=1 --snapshot=6554Ki \
       --bucket_schedule=@$W/search/s$i.json --seed=0
done
```

### 5b. Pick

For each run, take the best snapshot × EMA (0.050, 0.100) by FID against `ref_heldout.pkl`, then compare runs by
that best FID-50k. Quick screen at 10k images, final call at 50k:

```bash
torchrun --standalone --nproc_per_node=$GPUS calculate_metrics.py gen --net=<snapshot.pkl> --ref=$W/ref_heldout.pkl --num=50000
```

### 5c. Full runs

```bash
# Ours: the winning schedule, full data, full length
bash scripts/run_train.sh $GPUS $COMMON --outdir=$W/runs/ours_full --bucket_schedule=@$W/search/s<best>.json \
     --duration=2048Mi --snapshot=64Mi --seed=0
# Ambient-o: same clean set, sigma_min 0.2 for every other image, 1.8x length
echo '{"type":"fixed","sigma_min":{"1":0.2,"2":0.2,"3":0.2}}' > $W/ambo.json
bash scripts/run_train.sh $GPUS $COMMON --outdir=$W/runs/ambo_full --bucket_schedule=@$W/ambo.json \
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

## 7. Smoke test before anything long (about 30 min on 1 GPU)

A 10-class × 200-image slice: build the dataset, train our schedule on the short subset, stop with USR1 and resume,
train the Ambient-o baseline, build held-out reference stats, score FID + FD-DINOv2, sample with guidance. It passed
on 1 × L40S, 1 × H100 and 2 × L40S (multi-GPU sampler order and resume) with torch 2.6.0. Rerun it on the new cluster first (`smoke.sh`, ships with this runbook).

## Open choices (decide before launching)

- Ambient-o at 1.8× keeps the EDM2 LR schedule (inverse-sqrt decay past 70000 batches); no re-tuning.
- Seeds: one seed per full run first; more if the gap is close.
- Snapshot stride: 1/32 of the run.
