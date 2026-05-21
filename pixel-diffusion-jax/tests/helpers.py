from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import flax
from flax.core import freeze, unfreeze
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
JAX_ROOT = REPO_ROOT / "pixel-diffusion-jax"
PT_ROOT = REPO_ROOT / "pixel-diffusion"

if str(JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(JAX_ROOT))


def load_pt_module(rel_path: str, alias: str):
    module_path = PT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pt_networks():
    return load_pt_module("training/networks.py", "pixel_diffusion_pt_networks")


def load_pt_generate():
    return load_pt_module("generate.py", "pixel_diffusion_pt_generate")


def flatten_pt_state_dict(state_dict):
    items = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if key.endswith("resample_filter") or key.endswith("freqs"):
            continue
        if isinstance(value, torch.Tensor):
            items.append((key, value.detach().cpu().numpy()))
    return items


def assign_pt_weights_to_jax(variables, pt_state_dict):
    variables = unfreeze(variables)
    flat_params = flax.traverse_util.flatten_dict(variables["params"], keep_empty_nodes=True)
    pt_items = flatten_pt_state_dict(pt_state_dict)
    param_keys = list(flat_params.keys())
    assert len(param_keys) == len(pt_items), (len(param_keys), len(pt_items))
    for key, (_, pt_value) in zip(param_keys, pt_items):
        param_value = flat_params[key]
        assert param_value.shape == pt_value.shape, (key, param_value.shape, pt_value.shape)
        flat_params[key] = np.asarray(pt_value, dtype=param_value.dtype)
    variables["params"] = flax.traverse_util.unflatten_dict(flat_params)
    return freeze(variables)


def make_jax_input(batch=2, channels=3, resolution=64, augment_dim=9):
    x = np.random.randn(batch, channels, resolution, resolution).astype(np.float32)
    sigma = np.exp(np.random.randn(batch).astype(np.float32))
    labels = np.zeros((batch, 0), dtype=np.float32)
    augment = np.zeros((batch, augment_dim), dtype=np.float32)
    return x, sigma, labels, augment
