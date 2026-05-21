"""Ambient EDM losses in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from utils.ambient_utils_jax import from_x0_pred_to_xnature_pred_ve_to_ve


class AmbientEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, *args, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        apply_fn,
        params,
        rng,
        x_tn,
        sigma_tn,
        sigma_t,
        labels=None,
        augment_labels=None,
        train: bool = True,
        **model_kwargs,
    ):
        sigma_tn = jnp.asarray(sigma_tn, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        sigma_t = jnp.asarray(sigma_t, dtype=jnp.float32).reshape(-1, 1, 1, 1)

        noise_tn_to_t = jax.random.normal(rng, x_tn.shape, dtype=x_tn.dtype) * jnp.sqrt(jnp.square(sigma_t) - jnp.square(sigma_tn))
        x_t = x_tn + noise_tn_to_t

        x0_pred = apply_fn(
            {"params": params},
            x_t,
            sigma_t.squeeze(),
            labels,
            augment_labels=augment_labels,
            train=train,
            **model_kwargs,
        )
        x_tn_pred = from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, x_t, sigma_t, sigma_tn)

        edm_weight = (self.sigma_data**2 + sigma_t**2) / (sigma_t**2 * self.sigma_data**2)
        ambient_factor = sigma_t**4 / jnp.square(sigma_t**2 - sigma_tn**2)
        ambient_weight = edm_weight * ambient_factor

        loss = ambient_weight * jnp.square(x_tn_pred - x_tn)
        return loss, x0_pred, sigma_t, x_t


class AmbientEDMCLSLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, *args, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        apply_fn,
        params,
        rng,
        x0,
        sigma_t,
        cls_labels,
        augment_labels=None,
        labels=None,
        train: bool = True,
        **model_kwargs,
    ):
        sigma_t = jnp.asarray(sigma_t, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        x_t = x0 + jax.random.normal(rng, x0.shape, dtype=x0.dtype) * sigma_t
        output = apply_fn(
            {"params": params},
            x_t,
            sigma_t.squeeze(),
            labels,
            augment_labels=augment_labels,
            train=train,
            **model_kwargs,
        )
        x0_pred = output["x0_pred"]
        cls_logits = output["cls_logits"]
        loss = jnp.maximum(cls_logits, 0) - cls_logits * cls_labels + jnp.log1p(jnp.exp(-jnp.abs(cls_logits)))
        return loss, x0_pred, sigma_t, x_t
