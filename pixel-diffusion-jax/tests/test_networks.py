from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp

from helpers import assign_pt_weights_to_jax, load_pt_networks, make_jax_input
from models.networks import EDMPrecond


def test_edm_precond_matches_pytorch():
    pt = load_pt_networks()
    pt_cfg = dict(
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

    jax_cfg = dict(
        img_resolution=64,
        img_channels=3,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNet",
        model_kwargs=dict(
            model_channels=128,
            channel_mult=[2, 2, 2],
            channel_mult_noise=1,
            resample_filter=[1, 1],
            dropout=0.13,
            augment_dim=9,
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        ),
    )

    pt_model = pt.EDMPrecond(**pt_cfg).eval()
    jax_model = EDMPrecond(**jax_cfg)

    x, sigma, labels, augment = make_jax_input()
    x_pt = torch.from_numpy(x)
    sigma_pt = torch.from_numpy(sigma)
    augment_pt = torch.from_numpy(augment)
    with torch.no_grad():
        pt_out = pt_model(x_pt, sigma_pt, class_labels=None, augment_labels=augment_pt).cpu().numpy()

    variables = jax_model.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(sigma), None, augment_labels=jnp.asarray(augment), train=False)
    variables = assign_pt_weights_to_jax(variables, pt_model.state_dict())
    jax_out = jax_model.apply(variables, jnp.asarray(x), jnp.asarray(sigma), None, augment_labels=jnp.asarray(augment), train=False)
    assert np.allclose(pt_out, np.asarray(jax_out), atol=1e-5)
