"""CLI entry point for the JAX ambient diffusion port."""

from __future__ import annotations

import copy
import importlib
import json
import os
import random
import re
import string

import click
import wandb

import ambient_utils
from training import training_loop


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def _try_load_corruptions_dict(noise_config):
    if not noise_config or noise_config == "identity":
        return {}
    try:
        module = importlib.import_module(f"noise_configs.{noise_config}")
        return getattr(module, "corruptions_dict", {})
    except Exception:
        return {}


@click.command()
@click.option("--outdir", type=str, required=True)
@click.option("--data", type=str, required=True)
@click.option("--cond", type=bool, default=False, show_default=True)
@click.option("--arch", type=click.Choice(["ddpmpp", "ncsnpp", "adm"]), default="ddpmpp", show_default=True)
@click.option("--precond", type=click.Choice(["edm", "edmcls"]), default="edm", show_default=True)
@click.option("--duration", type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option("--batch", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--batch-gpu", type=click.IntRange(min=1))
@click.option("--cbase", type=int)
@click.option("--cres", type=parse_int_list)
@click.option("--lr", type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option("--weight_decay", type=click.FloatRange(min=0, min_open=False), default=0.0, show_default=True)
@click.option("--ema", type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option("--dropout", type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option("--augment", type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option("--clip", type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)
@click.option("--fp16", type=bool, default=False, show_default=True)
@click.option("--ls", type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option("--bench", type=bool, default=True, show_default=True)
@click.option("--cache", type=bool, default=True, show_default=True)
@click.option("--workers", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--expr_id", type=str, default="test")
@click.option("--desc", type=str)
@click.option("--nosubdir", is_flag=True)
@click.option("--tick", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--snap", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--dump", type=click.IntRange(min=1), default=500, show_default=True)
@click.option("--seed", type=int)
@click.option("--transfer", type=str)
@click.option("--resume", type=str)
@click.option("-n", "--dry-run", is_flag=True)
@click.option("--noise_config", type=str, default="identity")
@click.option("--corruption_probability", type=float, default=0.5)
@click.option("--dataset_keep_percentage", type=float, default=1.0, show_default=True)
@click.option("--cls_epsilon", type=float, default=0.05, show_default=True)
@click.option("--cls_ema_window", type=int, default=32, show_default=True)
@click.option("--overwrite_cls_labels_path", type=str, default=None)
@click.option("--crop_size", type=int, default=None)
@click.option("--s_max", type=float, default=None, show_default=True)
def main(**kwargs):
    opts = kwargs
    corruptions_dict = _try_load_corruptions_dict(opts["noise_config"])

    dataset_kwargs = dict(
        path=opts["data"],
        use_labels=opts["cond"],
        cache=opts["cache"],
        corruptions_dict=corruptions_dict,
        corruption_probability=opts["corruption_probability"],
        only_positive=False,
    )

    try:
        dataset_obj = ambient_utils.dataset.SyntheticallyCorruptedImageFolderDataset(**dataset_kwargs)
        dataset_kwargs["dataset_keep_percentage"] = opts["dataset_keep_percentage"]
        dataset_kwargs["resolution"] = dataset_obj.resolution
        dataset_kwargs["max_size"] = int(len(dataset_obj) * opts["dataset_keep_percentage"])
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f"--data: {err}")

    if opts["arch"] == "ddpmpp":
        model_kwargs = dict(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif opts["arch"] == "ncsnpp":
        model_kwargs = dict(
            model_type="SongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    else:
        model_kwargs = dict(
            model_type="DhariwalUNet",
            model_channels=192,
            channel_mult=[1, 2, 3, 4],
        )

    if opts["cbase"] is not None:
        model_kwargs["model_channels"] = opts["cbase"]
    if opts["cres"] is not None:
        model_kwargs["channel_mult"] = opts["cres"]

    model_kwargs.update(dropout=opts["dropout"], use_fp16=opts["fp16"], augment_dim=9 if opts["augment"] else 0)

    network_kwargs = dict(
        class_name="models.networks.EDMPrecond",
        model_kwargs=model_kwargs,
    )

    loss_kwargs = dict(class_name="training.loss.AmbientEDMLoss")
    optimizer_kwargs = dict(lr=opts["lr"], betas=(0.9, 0.999), weight_decay=opts["weight_decay"])
    augment_kwargs = None
    if opts["augment"]:
        augment_kwargs = dict(
            p=opts["augment"],
            xflip=1e8,
            yflip=1,
            scale=1,
            rotate_frac=1,
            aniso=1,
            translate_frac=1,
        )

    total_kimg = max(int(opts["duration"] * 1000), 1)

    desc = f"{dataset_kwargs['path'].split('/')[-1]}-{'cond' if opts['cond'] else 'uncond'}-{opts['arch']}-{opts['precond']}"
    if opts["desc"] is not None:
        desc += f"-{opts['desc']}"
    if opts["seed"] is None:
        opts["seed"] = random.randint(0, (1 << 31) - 1)

    if opts["nosubdir"]:
        run_dir = opts["outdir"]
    else:
        random_string = "".join(random.choices(string.ascii_letters + string.digits, k=5))
        run_dir = os.path.join(opts["outdir"], f"{desc}-{random_string}")
        os.makedirs(run_dir, exist_ok=True)

    if opts["dry_run"]:
        print(json.dumps(
            dict(
                run_dir=run_dir,
                dataset_kwargs=dataset_kwargs,
                network_kwargs=network_kwargs,
                loss_kwargs=loss_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                augment_kwargs=augment_kwargs,
                total_kimg=total_kimg,
            ),
            indent=2,
            default=str,
        ))
        return

    wandb.init(project="ambient_syn", config=copy.deepcopy(opts), name=opts["expr_id"], dir=opts["outdir"])

    training_loop.training_loop(
        run_dir=run_dir,
        dataset_kwargs=dataset_kwargs,
        data_loader_kwargs=dict(pin_memory=True, num_workers=opts["workers"], prefetch_factor=8),
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        augment_kwargs=augment_kwargs,
        seed=opts["seed"],
        batch_size=opts["batch"],
        batch_gpu=opts["batch_gpu"],
        total_kimg=total_kimg,
        ema_halflife_kimg=int(opts["ema"] * 1000),
        kimg_per_tick=opts["tick"],
        snapshot_ticks=opts["snap"],
        state_dump_ticks=opts["dump"],
        resume_pkl=opts["transfer"],
        resume_state_dump=opts["resume"],
        clip=opts["clip"],
        cls_epsilon=opts["cls_epsilon"],
        cls_ema_window=opts["cls_ema_window"],
        overwrite_cls_labels_path=opts["overwrite_cls_labels_path"],
        crop_size=opts["crop_size"],
        sampler_kwargs=dict(s_max=opts["s_max"]),
    )


if __name__ == "__main__":
    main()
