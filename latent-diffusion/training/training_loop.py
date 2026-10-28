# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
import wandb
import ambient_utils
import json
from collections import defaultdict
import zipfile
from calculate_metrics_quality import load_stats
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import signal
from training import buckets as bk

# Slurm sends SIGUSR1 (--signal=B:USR1@<sec>) or SIGTERM before a time limit or preemption. The handler only sets a
# flag; the loop agrees on it across ranks, saves a training-state checkpoint and exits with code 3 (= requeue me).
_STOP_REQUESTED = False
def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images_t, sigma_t, images_tn, sigma_tn, labels=None):
        sigma_t = sigma_t[:, None, None, None]
        sigma_tn = sigma_tn[:, None, None, None]

        images_0_pred, logvar = net(images_t, sigma_t, labels, return_logvar=True) # prediction for images_0
        images_tn_pred = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(images_0_pred, images_t, sigma_t, sigma_tn)

        edm_weight = (self.sigma_data ** 2 + sigma_t ** 2) / (sigma_t ** 2 * self.sigma_data ** 2)
        ambient_factor = sigma_t ** 4 / ((sigma_t ** 2 - sigma_tn ** 2) ** 2)
        ambient_weight = edm_weight * ambient_factor

        loss = (ambient_weight / logvar.exp()) * ((images_tn_pred - images_tn) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),

    annotations_qualities_path = None, # Path to quality annotations
    bad_data_percentage = 0.8, # Percentage of bad data
    bad_data_sigma_min = 0.2, # Sigma_min assignec to bad data
    use_ambient_crops = True, # Wether to use crops to increase low diffusion time data

    # Per-bucket noise floors (training/buckets.py). When bucket_path is set, Ambient-o's CLIP-IQA annotation path
    # above is not used at all: the bucket file defines the clean set (bucket 0) and the bad buckets 1..K.
    bucket_path         = None,     # npz with bucket[i] per dataset index (+ optional subset_<name> masks)
    bucket_schedule     = None,     # JSON spec: per_bucket step functions or fixed floors (see training/buckets.py)
    schedule_horizon_nimg = None,   # run length the per_bucket fractions refer to (None = total_nimg)
    subset_name         = None,     # train only on bucket file mask subset_<name> (e.g. 'short' = the 1/10 data)
):
    # Initialize.
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = ambient_utils.dataset.SyntheticallyCorruptedImageFolderDataset(**dataset_kwargs)

    ## Annotations
    annotations = {}
    use_buckets = bucket_path is not None
    if use_buckets and annotations_qualities_path is not None:
        raise ValueError('give either bucket_path or annotations_qualities_path, not both')
    if use_buckets:
        bucket_of, subset_masks = bk.load_buckets(bucket_path, len(dataset_obj))
        num_buckets = int(bucket_of.max())
        subset_mask = None
        if subset_name is not None:
            if subset_name not in subset_masks:
                raise ValueError(f'bucket file has no subset_{subset_name}; it has {sorted(subset_masks)}')
            subset_mask = subset_masks[subset_name]
        horizon = schedule_horizon_nimg or total_nimg
        schedule = bk.BucketSchedule(bucket_schedule, num_buckets, P_mean=loss_kwargs['P_mean'],
                                     P_std=loss_kwargs['P_std'], horizon_nimg=horizon)
        used = bucket_of if subset_mask is None else bucket_of[subset_mask]
        dist.print0(f'Buckets: {num_buckets} bad + clean | training images {len(used)} of {len(bucket_of)} | per bucket '
                    + ', '.join(f'{b}={int((used == b).sum())}' for b in range(num_buckets + 1)))
        dist.print0(f'Bucket schedule: {json.dumps(schedule.spec)} | horizon {horizon // 1000} kimg')
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            dist.print0(f'  floors at {frac:.2f}: ' + ', '.join(f'{x:.3g}' for x in schedule.floors(frac * horizon)))
    elif annotations_qualities_path is not None:
        if annotations_qualities_path == "adrianrm/ambient-o-clip-iqa-patches-imagenet":
            annotations_qualities_path = hf_hub_download(repo_id='adrianrm/ambient-o-clip-iqa-patches-imagenet', filename="clip_iqa_patch_average.safetensors", repo_type="dataset")
            annotations_qualities = {}
            with safe_open(annotations_qualities_path, framework="pt", device='cpu') as f:
                for k in f.keys():
                    annotations_qualities[k] = f.get_tensor(k)
        else:
            annotations_qualities = load_stats(annotations_qualities_path)

        ### Sigma min
        global_qualities = annotations_qualities['CLIP-IQA'][:, 0]
        assert len(global_qualities) == len(dataset_obj), f'Qualities ({len(global_qualities)}) and dataset_obj ({len(dataset_obj)}) must have equal lengths'

        sorted_indices = torch.argsort(global_qualities, descending=True)
        rank = torch.arange(len(global_qualities))[sorted_indices]
        rank_threshold = int(len(global_qualities) * (1 - bad_data_percentage))
        annotations_sigma_min = torch.where(rank < rank_threshold, 0.0, bad_data_sigma_min)

        ### Sigma max
        latents_receptive_field_to_sigma_max = {4: 0.1, 8:0.175, 16:0.20, 32:1.00, 64:200.0}
        annotations_sigma_max = torch.zeros(len(global_qualities))
        if use_ambient_crops:
            for latents_receptive_field in [4, 8, 16, 32, 64]:
                pixel_receptive_field = 8 * latents_receptive_field
                patch_qualities = annotations_qualities[f'CLIP-IQA-{pixel_receptive_field}']

                sorted_indices = torch.argsort(patch_qualities, descending=True)
                rank = torch.arange(len(patch_qualities))[sorted_indices]

                rank_threshold = int(len(global_qualities) * (1 - bad_data_percentage))
                good_data_sigma_max = latents_receptive_field_to_sigma_max[latents_receptive_field]

                annotations_sigma_max = torch.where(rank < rank_threshold, good_data_sigma_max, annotations_sigma_max)

        ### Annotations tuple
        annotations = {dataset_obj._image_fnames[i]: (annotations_sigma_min[i], annotations_sigma_max[i]) for i in range(len(global_qualities))}
    else:
        annotations = {dataset_obj._image_fnames[i]: (0.0, 0.0) for i in range(len(dataset_obj))}

    ## Count number of samples with annotations zero
    count_zeros = sum(1 for value in annotations.values() if value[0] == 0.0)
    total_samples = len(annotations)
    percentage = (count_zeros / total_samples) * 100 if total_samples > 0 else 0
    print(f"Number of samples with annotation 0.0: {count_zeros} out of {total_samples} ({percentage:.2f}%)")

    ## Set dataset annotations
    dataset_obj.annotations = annotations

    ref_image = dataset_obj[0]['image']
    ref_label = dataset_obj[0]['label']
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=False)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    if use_buckets:
        # Resume the walk where each rank left it (positions saved with the training state). With a different GPU
        # count the saved positions cannot be split exactly; continue from the slowest rank instead.
        saved = state.get('sampler_pos')
        if saved is not None and len(saved) == dist.get_world_size():
            start_pos = int(saved[dist.get_rank()]) + dist.get_world_size()
        elif saved is not None:
            start_pos = int(min(saved)) + dist.get_world_size() + dist.get_rank()
            dist.print0(f'Resuming with {dist.get_world_size()} GPUs from a {len(saved)}-GPU checkpoint: walk restarts at {min(saved)}')
        else:
            start_pos = None
        dataset_sampler = bk.BucketSampler(len(dataset_obj), bucket_of, schedule, rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size(), seed=seed, P_mean=loss_kwargs['P_mean'],
                                           P_std=loss_kwargs['P_std'], subset=subset_mask, start_pos=start_pos,
                                           cur_nimg=state.cur_nimg)
        last_walk_pos = -1 if start_pos is None else start_pos - dist.get_world_size()
    else:
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True



        # Save network snapshot with wandb logging
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                
                # Log model checkpoint to wandb
                if dist.get_rank() == 0:
                    wandb.save(os.path.join(run_dir, fname))

        # Stop requested by Slurm (time limit / preemption)? All ranks must agree before anyone saves.
        stop_flag = torch.tensor([1.0 if _STOP_REQUESTED else 0.0], device=device)
        if dist.get_world_size() > 1:
            torch.distributed.all_reduce(stop_flag)
        stop_now = bool(stop_flag.item() > 0) and not done

        # Save state checkpoint.
        if (checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg) or (stop_now and state.cur_nimg != start_nimg):
            if use_buckets:
                pos = torch.tensor([last_walk_pos], dtype=torch.int64, device=device)
                allpos = [torch.zeros_like(pos) for _ in range(dist.get_world_size())]
                if dist.get_world_size() > 1:
                    torch.distributed.all_gather(allpos, pos)
                else:
                    allpos = [pos]
                state.sampler_pos = [int(x.item()) for x in allpos]
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)
        if stop_now:
            dist.print0(f'Stop requested at {state.cur_nimg // 1000} kimg: state saved, exiting with code 3 for requeue.')
            if dist.get_rank() == 0:
                open(os.path.join(run_dir, 'REQUEUE'), 'w').write(f'{state.cur_nimg}\n')   # read by scripts/run_train.sh
            if dist.get_world_size() > 1:
                torch.distributed.barrier()
            raise SystemExit(3)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                dataset_item = next(dataset_iterator)
                latents = dataset_item['image']
                labels = dataset_item['label']

                encoded_latents = encoder.encode_latents(latents.to(device))
                if use_buckets:
                    # the (sigma, floor) pair the sampler decided for each sample, in yield order
                    dec = [dataset_sampler.decided.popleft() for _ in range(len(dataset_item['raw_idx']))]
                    got = [int(x) for x in dataset_item['raw_idx']]
                    if [d[0] for d in dec] != got:
                        raise RuntimeError('sampler/loader order mismatch: decided indices do not match the batch')
                    sigma_t = torch.tensor([d[1] for d in dec], device=device, dtype=torch.float32)
                    sigma_tn = torch.tensor([d[2] for d in dec], device=device, dtype=torch.float32)
                    last_walk_pos = dec[-1][3]
                    b = torch.as_tensor(bucket_of[got].astype(np.int64))
                    for k in range(num_buckets + 1):
                        training_stats.report(f'Buckets/share_b{k}', (b == k).float())
                    training_stats.report('Buckets/ambient_share', (sigma_tn > 0).float())
                else:
                    sigma_tn = torch.tensor([dataset_obj.annotations.get(filename, (0.0, 0.0))[0] for filename in dataset_item['filename']], device=device)

                    # Get sigma_t
                    batch_sampled_sigmas = [dataset_sampler.sampled_sigmas[filename] for filename in dataset_item['filename']]
                    sigma_t = torch.tensor(batch_sampled_sigmas, device=device)

                # this avoids negative square roots.
                sigma_tn = torch.where(sigma_t <= sigma_tn, torch.zeros_like(sigma_tn), sigma_tn)

                # Bring up to noise level sigma_tn
                encoded_latents_tn = encoded_latents + dataset_item["noise"].to(device).to(encoded_latents.dtype)[:, :encoded_latents.shape[1]] * sigma_tn[:, None, None, None]
                encoded_latents_t = encoded_latents_tn + torch.sqrt(sigma_t**2 - sigma_tn**2)[:, None, None, None] * torch.randn_like(encoded_latents_tn, device=device)

                loss = loss_fn(net=ddp, images_t=encoded_latents_t, sigma_t=sigma_t, images_tn=encoded_latents_tn, sigma_tn=sigma_tn, labels=labels.to(device))

                                
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                save_freq = 200_000
                # for every 50 prints, save images
                if (done or state.cur_nimg % (status_nimg * save_freq) == 0) and dist.get_rank() == 0:
                    ambient_utils.save_images(encoded_latents[:, :3], os.path.join(run_dir, f"batch_latents_{state.cur_nimg}.png"), save_wandb=True)
                    decoded_images = encoder.decode(encoded_latents)
                    decoded_images = (decoded_images / 127.5) - 1  # Convert from [0, 255] to [-1, 1]
                    ambient_utils.save_images(decoded_images, os.path.join(run_dir, f"batch_decoded_{state.cur_nimg}.png"), save_wandb=True)
                    metrics = {
                        'Progress/tick': state.cur_nimg // 1000,
                        'Progress/kimg': state.cur_nimg / 1e3,
                        'Timing/total_sec': state.total_elapsed_time,
                        'Timing/sec_per_tick': cur_time - prev_status_time,
                        'Timing/sec_per_kimg': (cur_time - prev_status_time - cumulative_training_time) / max(state.cur_nimg - prev_status_nimg, 1) * 1e3,
                        'Timing/maintenance_sec': cur_time - prev_status_time - cumulative_training_time,
                        'Resources/cpu_mem_gb': cpu_memory_usage / 2**30,
                        'Resources/peak_gpu_mem_gb': torch.cuda.max_memory_allocated(device) / 2**30,
                        'Resources/peak_gpu_mem_reserved_gb': torch.cuda.max_memory_reserved(device) / 2**30,
                        'Loss/loss': training_stats.report0('Loss/loss', loss)
                    }
                    wandb.log(metrics, step=state.cur_nimg // 1000)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if use_buckets:
            dataset_sampler.cur_nimg = state.cur_nimg
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------
