"""JAX helpers that mirror the ambient_utils behavior used by pixel-diffusion."""

from __future__ import annotations

import jax.numpy as jnp


def from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, x_t, sigma_t, sigma_tn):
    sigma_t = jnp.asarray(sigma_t, dtype=jnp.float32)
    sigma_tn = jnp.asarray(sigma_tn, dtype=jnp.float32)
    if sigma_t.ndim == 1:
        sigma_t = sigma_t[:, None, None, None]
    if sigma_tn.ndim == 1:
        sigma_tn = sigma_tn[:, None, None, None]

    return (1.0 - jnp.square(sigma_tn / sigma_t)) * x0_pred + jnp.square(sigma_tn / sigma_t) * x_t
