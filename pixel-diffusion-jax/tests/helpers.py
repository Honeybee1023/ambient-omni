from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from collections.abc import Mapping

import flax
from flax.core import freeze, unfreeze
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
JAX_ROOT = REPO_ROOT / "pixel-diffusion-jax"
PT_ROOT = REPO_ROOT / "pixel-diffusion"

if str(JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(JAX_ROOT))
if str(PT_ROOT) not in sys.path:
    sys.path.append(str(PT_ROOT))


def load_pt_module(rel_path: str, alias: str):
    module_path = PT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_pt_networks():
    return load_pt_module("training/networks.py", "pixel_diffusion_pt_networks")


def load_pt_generate():
    return load_pt_module("generate.py", "pixel_diffusion_pt_generate")


def load_pt_loss():
    return load_pt_module("training/loss.py", "pixel_diffusion_pt_loss")


def flatten_tree_ordered(tree, prefix=()):
    items = []
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            if isinstance(key, int):
                key = str(key)
            items.extend(flatten_tree_ordered(value, prefix + (key,)))
    else:
        items.append((prefix, tree))
    return items


def tree_get(tree, path):
    value = tree
    for key in path:
        value = value[key]
    return value


def flatten_pt_state_dict(state_dict):
    items = []
    for key, value in flatten_tree_ordered(state_dict):
        if key[-1].endswith("resample_filter") or key[-1].endswith("freqs"):
            continue
        if isinstance(value, torch.Tensor):
            items.append((key, value.detach().cpu().numpy()))
    return items


def assign_pt_weights_to_jax(variables, pt_state_dict):
    variables = unfreeze(variables)
    flat_params = flatten_tree_ordered(variables["params"])
    pt_items = flatten_pt_state_dict(pt_state_dict)
    assert len(flat_params) == len(pt_items), (len(flat_params), len(pt_items))
    for (key, param_value), (_, pt_value) in zip(flat_params, pt_items):
        assert param_value.shape == pt_value.shape, (key, param_value.shape, pt_value.shape)
    def assign(tree, values, idx=0):
        if isinstance(tree, Mapping):
            new_tree = {}
            for key, value in tree.items():
                new_value, idx = assign(value, values, idx)
                new_tree[key] = new_value
            return type(tree)(new_tree), idx
        return np.asarray(values[idx], dtype=tree.dtype), idx + 1

    pt_values = [pt_value for _, pt_value in pt_items]
    variables["params"], consumed = assign(variables["params"], pt_values)
    assert consumed == len(pt_values), (consumed, len(pt_values))
    return freeze(variables)


def make_jax_input(batch=2, channels=3, resolution=64, augment_dim=9):
    x = np.random.randn(batch, channels, resolution, resolution).astype(np.float32)
    sigma = np.exp(np.random.randn(batch).astype(np.float32))
    labels = np.zeros((batch, 0), dtype=np.float32)
    augment = np.zeros((batch, augment_dim), dtype=np.float32)
    return x, sigma, labels, augment
