from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp

from helpers import assign_pt_weights_to_jax, load_pt_networks, make_jax_input
from models.networks import EDMPrecond
from training.loss import AmbientEDMLoss


def test_ambient_loss_matches_pytorch_forward_and_grads():
    pt = load_pt_networks()
    cfg = dict(
        img_resolution=64,
        img_channels=3,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNet",
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_noise=1,
        resample_filter=[1, 1],
        dropout=0.13,
        augment_dim=9,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
    )

    pt_model = pt.EDMPrecond(**cfg).eval()
    jax_model = EDMPrecond(**cfg)
    loss_pt = pt.AmbientEDMLoss()
    loss_jax = AmbientEDMLoss()

    x, sigma, labels, augment = make_jax_input()
    x_tn = torch.from_numpy(np.random.randn(*x.shape).astype(np.float32))
    sigma_tn = torch.from_numpy(np.abs(np.random.randn(x.shape[0]).astype(np.float32)) * 0.1)
    sigma_t = torch.from_numpy(np.abs(np.random.randn(x.shape[0]).astype(np.float32)) + 0.5)
    augment_pt = torch.from_numpy(augment)

    pt_model.zero_grad(set_to_none=True)
    pt_loss, pt_x0_pred, pt_sigma, pt_x_t = loss_pt(
        pt_model,
        x_tn,
        sigma_tn,
        sigma_t,
        labels=None,
        augment_labels=augment_pt,
    )
    pt_scalar = pt_loss.sum()
    pt_scalar.backward()
    pt_grads = [p.grad.detach().cpu().numpy() for p in pt_model.parameters() if p.grad is not None]

    variables = jax_model.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(sigma), None, augment_labels=jnp.asarray(augment), train=False)
    variables = assign_pt_weights_to_jax(variables, pt_model.state_dict())
    rng = jax.random.PRNGKey(1)

    def apply_fn(vars_, x_, sigma_, labels_, augment_labels=None, train=False):
        return jax_model.apply(vars_, x_, sigma_, labels_, augment_labels=augment_labels, train=train)

    def loss_fn(params):
        vars_ = dict(variables)
        vars_["params"] = params
        loss, x0_pred, sigma_out, x_t = loss_jax(
            apply_fn,
            vars_,
            rng,
            jnp.asarray(x_tn.numpy()),
            jnp.asarray(sigma_tn.numpy()),
            jnp.asarray(sigma_t.numpy()),
            labels=None,
            augment_labels=jnp.asarray(augment),
            train=False,
        )
        return loss.sum()

    grads = jax.grad(loss_fn)(variables["params"])
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_pt_grads = [g for g in pt_grads]
    assert len(flat_grads) == len(flat_pt_grads)
    for jg, pg in zip(flat_grads, flat_pt_grads):
        assert np.allclose(np.asarray(jg), pg, atol=1e-5)

