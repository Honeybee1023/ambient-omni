from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp

from helpers import assign_pt_weights_to_jax, load_pt_networks
from generate import StackedRandomGenerator, edm_sampler
from models.networks import EDMPrecond


def pt_edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float("inf"), S_noise=1, stop_variance=0.0):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        if t_next ** 2 < stop_variance:
            return denoised
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


class PTStackedRandomGenerator:
    def __init__(self, seeds):
        self.generators = [torch.Generator().manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn_like(self, input):
        assert input.shape[0] == len(self.generators)
        return torch.stack([torch.randn(input.shape[1:], generator=gen, dtype=input.dtype) for gen in self.generators])


def test_edm_sampler_matches_pytorch():
    pt_networks = load_pt_networks()
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

    pt_model = pt_networks.EDMPrecond(**pt_cfg).eval()
    jax_model = EDMPrecond(**jax_cfg)
    x = np.random.randn(2, 3, 64, 64).astype(np.float32)
    latents = np.random.randn(2, 3, 64, 64).astype(np.float64)
    augment = np.zeros((2, 9), dtype=np.float32)
    variables = jax_model.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(np.ones(2, dtype=np.float32)), None, augment_labels=jnp.asarray(augment), train=False)
    variables = assign_pt_weights_to_jax(variables, pt_model.state_dict())

    pt_latents = torch.from_numpy(latents.astype(np.float32))
    pt_gen = PTStackedRandomGenerator([0, 1])
    jax_gen = StackedRandomGenerator([0, 1])

    with torch.no_grad():
        pt_out = pt_edm_sampler(
            pt_model.eval().to(torch.float32),
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
