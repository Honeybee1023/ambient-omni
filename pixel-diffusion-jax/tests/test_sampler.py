from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp

from helpers import assign_pt_weights_to_jax, load_pt_generate, load_pt_networks
from generate import StackedRandomGenerator, edm_sampler
from models.networks import EDMPrecond


def test_edm_sampler_matches_pytorch():
    pt_networks = load_pt_networks()
    pt_generate = load_pt_generate()
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

    pt_model = pt_networks.EDMPrecond(**cfg).eval()
    jax_model = EDMPrecond(**cfg)
    x = np.random.randn(2, 3, 64, 64).astype(np.float32)
    latents = np.random.randn(2, 3, 64, 64).astype(np.float64)
    augment = np.zeros((2, 9), dtype=np.float32)
    variables = jax_model.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(np.ones(2, dtype=np.float32)), None, augment_labels=jnp.asarray(augment), train=False)
    variables = assign_pt_weights_to_jax(variables, pt_model.state_dict())

    pt_model = pt_model.eval()
    pt_model = pt_model.to(torch.float32)
    pt_latents = torch.from_numpy(latents.astype(np.float32))
    pt_gen = pt_generate.StackedRandomGenerator("cpu", [0, 1])
    jax_gen = StackedRandomGenerator([0, 1])

    with torch.no_grad():
        pt_out = pt_generate.edm_sampler(
            pt_model,
            pt_latents,
            class_labels=None,
            randn_like=pt_gen.randn_like,
            num_steps=3,
            sigma_min=0.002,
            sigma_max=2.0,
            rho=7,
        ).cpu().numpy()

    jax_out = edm_sampler(
        jax_model,
        variables,
        jnp.asarray(latents),
        class_labels=None,
        randn_like=jax_gen.randn_like,
        num_steps=3,
        sigma_min=0.002,
        sigma_max=2.0,
        rho=7,
    )
    assert np.allclose(pt_out, np.asarray(jax_out), atol=1e-5)

