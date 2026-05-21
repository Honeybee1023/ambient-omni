"""Generate images with the JAX/Flax EDM sampler."""

from __future__ import annotations

import re
from typing import Callable, Optional

import click
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def edm_sampler(
    net,
    variables,
    latents,
    class_labels=None,
    randn_like: Optional[Callable] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    stop_variance: float = 0.0,
):
    sigma_min = max(sigma_min, float(net.sigma_min))
    sigma_max = min(sigma_max, float(net.sigma_max))

    if randn_like is None:
        randn_like = lambda x: jax.random.normal(jax.random.PRNGKey(0), x.shape, dtype=x.dtype)

    step_indices = jnp.arange(num_steps, dtype=jnp.float64)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = jnp.concatenate([net.round_sigma(t_steps), jnp.zeros_like(t_steps[:1])])

    def model_apply(x, sigma):
        return net.apply(variables, x, sigma, class_labels, train=False)

    x_next = jnp.asarray(latents, dtype=jnp.float64) * t_steps[0]
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + jnp.sqrt(jnp.maximum(t_hat**2 - t_cur**2, 0)) * S_noise * randn_like(x_cur)

        denoised = model_apply(x_hat, t_hat).astype(jnp.float64)
        if t_next**2 < stop_variance:
            return denoised

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = model_apply(x_next, t_next).astype(jnp.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class StackedRandomGenerator:
    def __init__(self, seeds):
        super().__init__()
        self.keys = [jax.random.PRNGKey(int(seed) % (1 << 32)) for seed in seeds]

    def _split(self):
        new_keys = []
        subkeys = []
        for key in self.keys:
            key, subkey = jax.random.split(key)
            new_keys.append(key)
            subkeys.append(subkey)
        self.keys = new_keys
        return subkeys

    def randn(self, size, dtype=jnp.float32):
        assert size[0] == len(self.keys)
        subkeys = self._split()
        return jnp.stack([jax.random.normal(k, size[1:], dtype=dtype) for k in subkeys])

    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype)

    def randint(self, *args, size, dtype=jnp.int32):
        assert size[0] == len(self.keys)
        subkeys = self._split()
        return jnp.stack([jax.random.randint(k, size[1:], *args, dtype=dtype) for k in subkeys])


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def load_hf_checkpoint(repo_id):
    raise NotImplementedError("HF checkpoint loading has not been ported yet.")


@click.command()
@click.option("--network", "network_pkl", help="Network checkpoint path or URL", type=str, required=True)
@click.option("--outdir", help="Where to save the output images", type=str, required=True)
@click.option("--seeds", help="Random seeds (e.g. 1,2,5-10)", type=parse_int_list, default="0-63", show_default=True)
@click.option("--subdirs", is_flag=True, help="Create subdirectory for every 1000 seeds")
@click.option("--class", "class_idx", type=click.IntRange(min=0), default=None)
@click.option("--batch", "max_batch_size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--steps", "num_steps", type=click.IntRange(min=1), default=18, show_default=True)
@click.option("--sigma_min", type=click.FloatRange(min=0, min_open=True))
@click.option("--sigma_max", type=click.FloatRange(min=0, min_open=True))
@click.option("--rho", type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option("--S_churn", type=click.FloatRange(min=0), default=0, show_default=True)
@click.option("--S_min", type=click.FloatRange(min=0), default=0, show_default=True)
@click.option("--S_max", type=click.FloatRange(min=0), default="inf", show_default=True)
@click.option("--S_noise", type=float, default=1, show_default=True)
@click.option("--stop_variance", type=float, default=0.0)
def main(**kwargs):
    raise NotImplementedError("CLI generation is not wired up yet.")


if __name__ == "__main__":
    main()

