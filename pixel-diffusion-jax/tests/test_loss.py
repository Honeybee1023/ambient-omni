from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp

from helpers import assign_pt_weights_to_jax, flatten_pt_state_dict, flatten_tree_ordered, load_pt_networks, make_jax_input, tree_get
from models.networks import EDMPrecond


def test_ambient_loss_matches_pytorch_forward_and_grads():
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
    x_tn = torch.from_numpy(np.random.randn(*x.shape).astype(np.float32))
    sigma_tn = torch.from_numpy(np.abs(np.random.randn(x.shape[0]).astype(np.float32)) * 0.1)
    sigma_t = torch.from_numpy(np.abs(np.random.randn(x.shape[0]).astype(np.float32)) + 0.5)
    shared_noise = torch.from_numpy(np.random.randn(*x.shape).astype(np.float32))
    augment_pt = torch.from_numpy(augment)

    def ambient_loss_from_pred(x0_pred, x_t, x_tn_tensor, sigma_tn_tensor, sigma_t_tensor):
        sigma_tn_4 = sigma_tn_tensor.reshape(-1, 1, 1, 1)
        sigma_t_4 = sigma_t_tensor.reshape(-1, 1, 1, 1)
        sigma_ratio_sq = (sigma_tn_4 / sigma_t_4) ** 2
        x_tn_pred = (1.0 - sigma_ratio_sq) * x0_pred + sigma_ratio_sq * x_t
        edm_weight = (0.5**2 + sigma_t_4**2) / (sigma_t_4**2 * 0.5**2)
        ambient_factor = sigma_t_4**4 / (sigma_t_4**2 - sigma_tn_4**2) ** 2
        ambient_weight = edm_weight * ambient_factor
        loss = ambient_weight * (x_tn_pred - x_tn_tensor.reshape(x_tn_pred.shape)) ** 2
        return loss

    sigma_tn_4 = sigma_tn.reshape(-1, 1, 1, 1)
    sigma_t_4 = sigma_t.reshape(-1, 1, 1, 1)
    sigma_tn_4_jax = jnp.asarray(sigma_tn_4.numpy())
    sigma_t_4_jax = jnp.asarray(sigma_t_4.numpy())
    x_tn_jax = jnp.asarray(x_tn.numpy())
    pt_x_t = x_tn + shared_noise * torch.sqrt(sigma_t_4**2 - sigma_tn_4**2)
    pt_x0_pred = pt_model(pt_x_t, sigma_t, class_labels=None, augment_labels=augment_pt)
    pt_loss = ambient_loss_from_pred(pt_x0_pred, pt_x_t, x_tn, sigma_tn, sigma_t)
    pt_scalar = pt_loss.sum()
    pt_model.zero_grad(set_to_none=True)
    pt_scalar.backward()
    variables = jax_model.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(sigma), None, augment_labels=jnp.asarray(augment), train=False)
    param_paths = [path for path, _ in flatten_tree_ordered(variables["params"])]
    variables = assign_pt_weights_to_jax(variables, pt_model.state_dict())
    shared_noise_jax = jnp.asarray(shared_noise.numpy())

    def jax_loss_fn(params):
        vars_ = dict(variables)
        vars_["params"] = params
        pt_x_t_jax = x_tn_jax + shared_noise_jax * jnp.sqrt(sigma_t_4_jax**2 - sigma_tn_4_jax**2)
        x0_pred = jax_model.apply(vars_, pt_x_t_jax, jnp.asarray(sigma_t.numpy()), None, augment_labels=jnp.asarray(augment), train=False)
        loss = ambient_loss_from_pred(
            x0_pred,
            pt_x_t_jax,
            x_tn_jax,
            sigma_tn_4_jax,
            sigma_t_4_jax,
        )
        return loss.sum()

    grads = jax.grad(jax_loss_fn)(variables["params"])
    flat_grads = [tree_get(grads, path) for path in param_paths]
    expected_pt_names = [".".join(path) for path, _ in flatten_pt_state_dict(pt_model.state_dict())]
    pt_grad_map = {name: p.grad.detach().cpu().numpy() for name, p in pt_model.named_parameters() if p.grad is not None}
    flat_pt_grads = [pt_grad_map[name] for name in expected_pt_names]
    assert len(flat_grads) == len(flat_pt_grads)
    for jg, pg in zip(flat_grads, flat_pt_grads):
        assert np.allclose(np.asarray(jg), pg, atol=2e-2, rtol=1e-4)
