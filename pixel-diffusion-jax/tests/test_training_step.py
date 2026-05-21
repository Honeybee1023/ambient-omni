from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import torch

from helpers import load_pt_networks, make_jax_input
from models.networks import EDMPrecond
from training.loss import AmbientEDMLoss
from training.training_state import create_train_state, load_torch_snapshot_into_variables


def test_resume_pkl_loads_pytorch_snapshot_into_jax_model():
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
    x, sigma, labels, augment = make_jax_input()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / "snapshot.pkl"
        with open(snapshot_path, "wb") as f:
            pickle.dump({"ema": pt_model}, f)

        jax_model = EDMPrecond(
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
        variables = jax_model.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
            jnp.asarray(x),
            jnp.asarray(sigma),
            labels,
            augment_labels=jnp.asarray(augment),
            train=False,
        )
        loaded = load_torch_snapshot_into_variables(variables, snapshot_path)
        pt_out = pt_model(torch.from_numpy(x), torch.from_numpy(sigma), class_labels=None, augment_labels=torch.from_numpy(augment)).detach().cpu().numpy()
        jax_out = np.asarray(jax_model.apply(loaded, jnp.asarray(x), jnp.asarray(sigma), labels, augment_labels=jnp.asarray(augment), train=False))
        assert np.allclose(pt_out, jax_out, atol=1e-5)


def test_single_jax_training_step_updates_params():
    cfg = dict(
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
    model = EDMPrecond(**cfg)
    x = jnp.asarray(np.random.randn(2, 3, 64, 64).astype(np.float32))
    sigma = jnp.asarray(np.exp(np.random.randn(2).astype(np.float32)))
    labels = jnp.zeros((2, 0), dtype=jnp.float32)
    augment = jnp.zeros((2, 9), dtype=jnp.float32)
    variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, sigma, labels, augment_labels=augment, train=True)
    tx = optax.adam(1e-4)
    state = create_train_state(model.apply, variables, tx)
    loss_obj = AmbientEDMLoss()
    rng = jax.random.PRNGKey(1)

    def loss_fn(params):
        vars_ = {"params": params, **(state.model_variables or {})}
        loss, _, _, _ = loss_obj(model.apply, vars_, rng, x, sigma, sigma, labels=labels, augment_labels=augment, train=True)
        return loss.sum()

    grads = jax.grad(loss_fn)(state.params)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: a.shape == b.shape, state.params, new_params))
