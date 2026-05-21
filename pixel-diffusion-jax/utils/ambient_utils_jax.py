"""JAX helpers that mirror the ambient_utils behavior used by pixel-diffusion."""

from __future__ import annotations

import jax.numpy as jnp


def from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, x_t, sigma_t, sigma_tn):
    """Bring an x0 prediction down to the trust level sigma_tn.

    The upstream ambient-utils helper is used to convert a clean prediction
    at noise level sigma_t into a prediction for the partially corrupted
    target x_tn. The surrounding documentation describes this as "bring this
    to the trust level" and the loss uses a simple linear interpolation
    between the model prediction and the already-corrupted input.

    This implementation preserves the endpoint behavior:
    - sigma_tn == 0   -> return x0_pred
    - sigma_tn == sigma_t -> return x_t
    """

    sigma_t = jnp.asarray(sigma_t, dtype=jnp.float32)
    sigma_tn = jnp.asarray(sigma_tn, dtype=jnp.float32)
    if sigma_t.ndim == 1:
        sigma_t = sigma_t[:, None, None, None]
    if sigma_tn.ndim == 1:
        sigma_tn = sigma_tn[:, None, None, None]

    t_weight = jnp.square(sigma_tn) / jnp.square(sigma_t)
    return (1.0 - t_weight) * x0_pred + t_weight * x_t

