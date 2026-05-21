"""Training state and checkpoint helpers for JAX."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import flax
from flax import struct
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import jax


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
