"""Main training loop for the JAX port."""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from scipy.stats import truncnorm

import ambient_utils
import wandb

from training.loss import AmbientEDMLoss
from training.training_state import TrainState, create_train_state, pack_variables, restore_checkpoint, save_checkpoint
from utils.distributed import get_rank, get_world_size, print0, should_stop, update_progress
from utils.misc import InfiniteSampler


def _resolve_class(class_name):
    module_name, _, attr_name = class_name.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class name: {class_name}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _load_original_augment_pipe():
    repo_root = Path(__file__).resolve().parents[2]
    augment_path = repo_root / "pixel-diffusion" / "training" / "augment.py"
    spec = importlib.util.spec_from_file_location("pixel_diffusion_pt_augment", augment_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.AugmentPipe


def sample_t(current_sigma, loc, scale):
    numpy_cutoff = current_sigma.squeeze().cpu().numpy()
    numpy_cutoff = (np.log(numpy_cutoff) - loc) / scale
    sampled = truncnorm.rvs(numpy_cutoff, np.inf, loc=loc, scale=scale)
    return torch.tensor(sampled, device=current_sigma.device).exp().view_as(current_sigma)


def apply_ema(data, window=3):
    if len(data) == 0:
        return np.asarray([])
    alpha = 2.0 / (window + 1.0)
    out = []
    ema = None
    for value in data:
        ema = value if ema is None else alpha * value + (1 - alpha) * ema
        out.append(ema)
    return np.asarray(out)


def _to_jax(array):
    return jnp.asarray(array.detach().cpu().numpy())


def _torch_to_numpy(array):
    if array is None:
        return None
    return array.detach().cpu().numpy()


def _load_network(network_kwargs, dataset_obj):
    network_kwargs = dict(network_kwargs or {})
    class_name = network_kwargs.pop("class_name", "models.networks.EDMPrecond")
    model_cls = _resolve_class(class_name)
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    model = model_cls(**interface_kwargs, **network_kwargs)
    return model


def _create_optimizer(optimizer_kwargs, clip):
    optimizer_kwargs = dict(optimizer_kwargs or {})
    lr = optimizer_kwargs.pop("lr", optimizer_kwargs.pop("learning_rate", 1e-4))
    betas = optimizer_kwargs.pop("betas", (0.9, 0.999))
    eps = optimizer_kwargs.pop("eps", 1e-8)
    weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)
    txs = []
    if clip is not None and clip > 0:
        txs.append(optax.clip_by_global_norm(clip))
    if weight_decay:
        txs.append(optax.add_decayed_weights(weight_decay))
    txs.append(optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps))
    return optax.chain(*txs)


def _build_batch(dataset_item, annotations, dataset_sampler, augment_pipe=None, crop_size=None):
    images = dataset_item["image"]
    labels = dataset_item["label"]
    cls_labels = dataset_item.get("corruption_label", torch.zeros_like(labels))

    batch_annotations = [annotations[x] for x in dataset_item["filename"]]
    sigma_tn = torch.tensor([min(x[0], x[1]) for x in batch_annotations], device=images.device)
    sigma_t = torch.tensor([dataset_sampler.sampled_sigmas[x] for x in dataset_item["filename"]], device=images.device)
    sigma_tn = torch.where(sigma_tn > sigma_t, torch.zeros_like(sigma_tn), sigma_tn)

    if augment_pipe is not None:
        x0, augment_labels = augment_pipe(images)
    else:
        x0, augment_labels = images, None
    x_tn = x0 + dataset_item["noise"].to(images.dtype) * sigma_tn[:, None, None, None]

    if crop_size is not None:
        crop_start_h = torch.randint(0, images.shape[2] - crop_size + 1, (1,))
        crop_end_h = crop_start_h + crop_size
        crop_start_w = torch.randint(0, images.shape[3] - crop_size + 1, (1,))
        crop_end_w = crop_start_w + crop_size
        x_tn[:, :, :crop_start_h, :] = 0
        x_tn[:, :, crop_end_h:, :] = 0
        x_tn[:, :, :, :crop_start_w] = 0
        x_tn[:, :, :, crop_end_w:] = 0

    return {
        "x_tn": x_tn,
        "x0": x0,
        "labels": labels,
        "cls_labels": cls_labels,
        "augment_labels": augment_labels,
        "sigma_tn": sigma_tn,
        "sigma_t": sigma_t,
        "images": images,
    }


def training_loop(
    run_dir=".",
    dataset_kwargs={},
    data_loader_kwargs={},
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    augment_kwargs=None,
    seed=0,
    batch_size=512,
    batch_gpu=None,
    total_kimg=200000,
    ema_halflife_kimg=500,
    ema_rampup_ratio=0.05,
    lr_rampup_kimg=10000,
    loss_scaling=1,
    kimg_per_tick=50,
    snapshot_ticks=50,
    state_dump_ticks=500,
    resume_pkl=None,
    resume_state_dump=None,
    resume_kimg=0,
    cudnn_benchmark=True,
    device=None,
    clip=1.0,
    cls_epsilon=0.05,
    cls_ema_window=32,
    overwrite_cls_labels_path=None,
    crop_size=None,
    sampler_kwargs={},
):
    del cudnn_benchmark, device

    np.random.seed(seed % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    jax_key = jax.random.PRNGKey(seed)

    if overwrite_cls_labels_path is not None:
        dataset_cls_labels = {}
        with open(overwrite_cls_labels_path, "r") as f:
            for line in f:
                json_item = json.loads(line)
                dataset_cls_labels[json_item["image_file"]] = json_item["label"]
    else:
        dataset_cls_labels = None

    print0("Loading dataset...")
    dataset_obj = ambient_utils.dataset.SyntheticallyCorruptedImageFolderDataset(**dataset_kwargs)

    indices = [476716, 801177, 208667, 84697, 708005, 481119, 882784, 314948, 241315, 900832, 937237, 522057, 844026, 1021191, 789191, 668501]
    indices = [index % len(dataset_obj) for index in indices]
    images_to_save = [torch.tensor(dataset_obj[i]["image"]) for i in indices]
    if get_rank() == 0:
        ambient_utils.save_images(torch.stack(images_to_save), os.path.join(run_dir, "dataset.png"), save_wandb=True)

    annotations_file = os.path.join(dataset_kwargs["path"], "annotations.jsonl")
    annotations = defaultdict(lambda: (0.0, 0.0))
    lines_read = 0
    if os.path.exists(annotations_file):
        sigmas_path = os.path.join(dataset_kwargs["path"], "sigmas.txt")
        if os.path.exists(sigmas_path):
            with open(sigmas_path, "r") as f:
                sigmas = [float(line.strip()) for line in f]
            sigmas = torch.tensor(sigmas)
            sigmas = sigmas.sort(dim=0)[0]
        with open(annotations_file, "r") as f:
            for line in f:
                lines_read += 1
                line_json = json.loads(line)
                filename = line_json["filename"]
                if "probabilities" in line_json:
                    probs = np.array(line_json["probabilities"]).mean(axis=-1)
                    ema_probs = apply_ema(probs, window=cls_ema_window)
                    first_confusion = ambient_utils.classifier.analyze_classifier_trajectory(torch.tensor(ema_probs), sigmas, epsilon=cls_epsilon)["first_confusion"]
                    annotations[filename] = (first_confusion.cpu().item(), 0.0)
                elif any(key.startswith("crop_predictions") for key in line_json):
                    patch_size_to_probs = {}
                    for key, value in line_json.items():
                        if key.startswith("crop_predictions"):
                            patch_size = int(key.split("_")[-1])
                            patch_size_to_probs[patch_size] = np.mean(value)
                    for patch_size in sorted(patch_size_to_probs.keys(), reverse=True):
                        if patch_size_to_probs[patch_size] > 0.25:
                            break
                    else:
                        patch_size = 1
                    patch_to_sigma = {
                        1: 0.01,
                        4: 0.05,
                        8: 0.15,
                        16: 0.2,
                        24: 0.35,
                        32: 0.55,
                        48: 0.7,
                        64: 1.0,
                    }
                    sigma_max = patch_to_sigma[patch_size]
                    annotations[filename] = (300.0, sigma_max)
                elif "annotation" in line_json or "sigma" in line_json:
                    annotations[filename] = (line_json["annotation"], 0.0) if "annotation" in line_json else (line_json["sigma"], 0.0)
                elif "sigma_min" in line_json and "sigma_max" in line_json:
                    annotations[filename] = (line_json["sigma_min"], line_json["sigma_max"])
                else:
                    raise ValueError(f"Could not parse line {line}")

    if get_rank() == 0:
        with open(os.path.join(run_dir, "annotations_processed.jsonl"), "w") as f:
            for filename, annotation in annotations.items():
                f.write(f"{filename}: {annotation}\n")
        print(f"Num annotations: {len(list(annotations.keys()))}, Lines read: {lines_read}")

    dataset_obj.annotations = dict(annotations)
    print0("Rank", get_rank(), "Size", get_world_size())
    print0("Sampler kwargs", sampler_kwargs)
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj,
        rank=get_rank(),
        num_replicas=get_world_size(),
        seed=seed,
        **sampler_kwargs,
    )
    print0("Constructed sampler")
    batch_gpu_total = batch_size // get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * get_world_size()

    dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs)
    dataset_iterator = iter(dataloader)

    print0("Constructing network...")
    model = _load_network(network_kwargs, dataset_obj)
    loss_obj = _resolve_class(loss_kwargs.get("class_name", "training.loss.AmbientEDMLoss"))(**{k: v for k, v in loss_kwargs.items() if k != "class_name"})
    tx = _create_optimizer(optimizer_kwargs, clip)
    augment_pipe = _load_original_augment_pipe()(**augment_kwargs) if augment_kwargs is not None else None

    dummy_x = jnp.zeros((batch_gpu, dataset_obj.num_channels, dataset_obj.resolution, dataset_obj.resolution), dtype=jnp.float32)
    dummy_sigma = jnp.ones((batch_gpu,), dtype=jnp.float32)
    dummy_labels = jnp.zeros((batch_gpu, dataset_obj.label_dim), dtype=jnp.float32)
    init_key, jax_key = jax.random.split(jax_key)
    variables = model.init(init_key, dummy_x, dummy_sigma, dummy_labels, train=True)
    state = create_train_state(model.apply, variables, tx)

    if resume_state_dump is not None:
        restored = restore_checkpoint(os.path.dirname(resume_state_dump), step=int(resume_kimg))
        state = state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"],
            ema_params=restored["ema_params"],
            model_variables=restored["model_variables"],
        )

    if resume_pkl is not None:
        raise NotImplementedError("Restoring PyTorch snapshots into the JAX port is not yet supported.")

    print0(f"Training for {total_kimg} kimg...")
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg

    def compute_grads(state, batch, rng):
        def loss_fn(params):
            variables = {"params": params, **(state.model_variables or {})}
            loss, x0_pred, sigma, noisy_input = loss_obj(
                state.apply_fn,
                variables,
                rng,
                batch["x_tn"],
                batch["sigma_tn"],
                batch["sigma_t"],
                labels=batch["labels"],
                augment_labels=batch["augment_labels"],
                train=True,
            )
            return loss.sum() * (loss_scaling / batch_size), (loss, x0_pred, sigma, noisy_input)

        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e5, neginf=-1e5), grads)
        return loss_value, aux, grads

    while True:
        accum_grads = None
        loss_values = []
        aux_last = None
        for round_idx in range(num_accumulation_rounds):
            dataset_item = next(dataset_iterator)
            batch = _build_batch(dataset_item, annotations, dataset_sampler, augment_pipe=augment_pipe, crop_size=crop_size)
            if dataset_cls_labels is not None:
                batch["cls_labels"] = torch.tensor([dataset_cls_labels[x] for x in dataset_item["filename"]], device=batch["labels"].device)
            batch_jax = {
                "x_tn": _to_jax(batch["x_tn"]),
                "sigma_tn": _to_jax(batch["sigma_tn"]),
                "sigma_t": _to_jax(batch["sigma_t"]),
                "labels": _to_jax(batch["labels"]),
                "augment_labels": _to_jax(batch["augment_labels"]) if batch["augment_labels"] is not None else None,
            }
            jax_key, step_key = jax.random.split(jax_key)

            loss_value, aux, grads = compute_grads(state, batch_jax, step_key)
            loss_values.append(loss_value)
            accum_grads = grads if accum_grads is None else jax.tree_util.tree_map(lambda a, b: a + b, accum_grads, grads)
            aux_last = aux

        updates, opt_state = state.tx.update(accum_grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, max(cur_nimg, 1) * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        ema_params = jax.tree_util.tree_map(lambda e, p: p + ema_beta * (e - p), state.ema_params, params)
        state = state.replace(params=params, opt_state=opt_state, ema_params=ema_params)

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = 0.0
        print0(f"tick {cur_tick:<5d} kimg {cur_nimg / 1e3:<9.1f}")

        if (not done) and should_stop():
            done = True
            print0("Aborting...")

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            if get_rank() == 0:
                save_checkpoint(run_dir, state, cur_nimg // 1000)

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and get_rank() == 0:
            save_checkpoint(run_dir, state, cur_nimg // 1000)

        if get_rank() == 0:
            wandb.log({"tick": cur_tick, "kimg": cur_nimg / 1e3}, step=cur_tick)

        update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        if done:
            break

    print0("Exiting...")
