"""Training state and checkpoint helpers for JAX."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import flax
from flax.training import train_state
import orbax.checkpoint as ocp
import numpy as np
import torch


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    ema_params: object = None
    model_variables: object = None


def create_train_state(apply_fn, variables, tx):
    params = variables["params"]
    model_variables = {k: v for k, v in variables.items() if k != "params"}
    return TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        ema_params=params,
        model_variables=model_variables,
    )


def pack_variables(state):
    vars_out = {"params": state.params}
    vars_out.update(state.model_variables or {})
    return vars_out


def _flatten_tree_ordered(tree, prefix=()):
    items = []
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            items.extend(_flatten_tree_ordered(value, prefix + (key,)))
    else:
        items.append((prefix, tree))
    return items


def _load_snapshot_data(snapshot_path):
    parsed = urlparse(str(snapshot_path))
    if parsed.scheme in {"http", "https"}:
        with urlopen(str(snapshot_path)) as f:
            return pickle.load(f)
    with open(snapshot_path, "rb") as f:
        return pickle.load(f)


def _flatten_torch_state_dict(state_dict):
    items = []
    for key, value in state_dict.items():
        if key.endswith("resample_filter"):
            continue
        if isinstance(value, torch.Tensor):
            items.append((key, value.detach().cpu().numpy()))
    return items


def load_torch_snapshot_into_variables(variables, snapshot_path):
    data = _load_snapshot_data(snapshot_path)
    ema = data.get("ema", data)
    state_dict = ema.state_dict() if hasattr(ema, "state_dict") else ema

    variables = flax.core.unfreeze(variables)
    flat_vars = _flatten_tree_ordered(variables)
    flat_state = _flatten_torch_state_dict(state_dict)
    assert len(flat_vars) == len(flat_state), (len(flat_vars), len(flat_state))

    def assign(tree, values, idx=0):
        if isinstance(tree, Mapping):
            new_tree = {}
            for key, value in tree.items():
                new_value, idx = assign(value, values, idx)
                new_tree[key] = new_value
            return type(tree)(new_tree), idx
        return np.asarray(values[idx], dtype=tree.dtype), idx + 1

    variables, consumed = assign(variables, [value for _, value in flat_state])
    assert consumed == len(flat_state), (consumed, len(flat_state))
    return flax.core.freeze(variables)


def save_checkpoint(path, state, step):
    ckptr = ocp.StandardCheckpointer()
    payload = {
        "step": step,
        "params": state.params,
        "opt_state": state.opt_state,
        "ema_params": state.ema_params,
        "model_variables": state.model_variables,
    }
    ckptr.save(Path(path) / f"checkpoint_{step}", payload)
    ckptr.wait_until_finished()


def restore_checkpoint(path, target=None, step=None):
    ckptr = ocp.StandardCheckpointer()
    ckpt_path = Path(path) if step is None else Path(path) / f"checkpoint_{step}"
    return ckptr.restore(ckpt_path, target)
