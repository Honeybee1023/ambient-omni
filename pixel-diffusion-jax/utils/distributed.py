"""Single-device distributed helpers for the JAX port."""

from __future__ import annotations

import jax


def get_world_size():
    return jax.process_count()


def get_rank():
    return jax.process_index()


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def should_stop():
    return False


def update_progress(*args, **kwargs):
    return None

