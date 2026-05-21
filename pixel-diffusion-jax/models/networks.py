"""JAX/Flax port of the diffusion network stack used by pixel-diffusion.

This file mirrors the PyTorch module structure closely so that weights can be
transferred module-by-module during parity testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import numpy as np


def _weight_init(shape, mode, fan_in, fan_out, rng):
    if mode == "xavier_uniform":
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return scale * (jax.random.uniform(rng, shape, dtype=jnp.float32) * 2.0 - 1.0)
    if mode == "xavier_normal":
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return scale * jax.random.normal(rng, shape, dtype=jnp.float32)
    if mode == "kaiming_uniform":
        scale = np.sqrt(3.0 / fan_in)
        return scale * (jax.random.uniform(rng, shape, dtype=jnp.float32) * 2.0 - 1.0)
    if mode == "kaiming_normal":
        scale = np.sqrt(1.0 / fan_in)
        return scale * jax.random.normal(rng, shape, dtype=jnp.float32)
    raise ValueError(f'Invalid init mode "{mode}"')


def _make_initializer(mode: str, fan_in: int, fan_out: int, init_weight: float = 1.0):
    def init_fn(key, shape, dtype=jnp.float32):
        return jnp.asarray(_weight_init(shape, mode, fan_in, fan_out, key) * init_weight, dtype)

    return init_fn


def _make_bias_initializer(mode: str, fan_in: int, fan_out: int, init_bias: float = 0.0):
    def init_fn(key, shape, dtype=jnp.float32):
        return jnp.asarray(_weight_init(shape, mode, fan_in, fan_out, key) * init_bias, dtype)

    return init_fn


class Linear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    init_mode: str = "kaiming_normal"
    init_weight: float = 1.0
    init_bias: float = 0.0

    def setup(self):
        weight_init = _make_initializer(self.init_mode, self.in_features, self.out_features, self.init_weight)
        self.weight = self.param("weight", weight_init, (self.out_features, self.in_features))
        if self.bias:
            bias_init = _make_bias_initializer(self.init_mode, self.in_features, self.out_features, self.init_bias)
            self.bias_param = self.param("bias", bias_init, (self.out_features,))
        else:
            self.bias_param = None

    def __call__(self, x):
        x = jnp.matmul(x, self.weight.T.astype(x.dtype))
        if self.bias_param is not None:
            x = x + self.bias_param.astype(x.dtype)
        return x


def _resample_kernel(resample_filter):
    f = jnp.asarray(resample_filter, dtype=jnp.float32)
    f = jnp.outer(f, f)[None, None, :, :] / jnp.square(jnp.sum(f))
    return f


def _conv2d(x, w, padding, groups=1, strides=(1, 1), lhs_dilation=None):
    return lax.conv_general_dilated(
        x,
        w,
        window_strides=strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=None,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=groups,
    )


def _conv_transpose2d(x, w, strides=(2, 2), padding="VALID", groups=1):
    # Implement transposed convolution via lhs dilation so we can preserve
    # the exact grouped depthwise structure used by the PyTorch code.
    return lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding=padding,
        lhs_dilation=strides,
        rhs_dilation=None,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=groups,
    )


class Conv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel: int
    bias: bool = True
    up: bool = False
    down: bool = False
    resample_filter: tuple = (1, 1)
    fused_resample: bool = False
    init_mode: str = "kaiming_normal"
    init_weight: float = 1.0
    init_bias: float = 0.0

    def setup(self):
        assert not (self.up and self.down)
        if self.kernel:
            fan = self.in_channels * self.kernel * self.kernel
            fout = self.out_channels * self.kernel * self.kernel
            weight_init = _make_initializer(self.init_mode, fan, fout, self.init_weight)
            self.weight = self.param("weight", weight_init, (self.out_channels, self.in_channels, self.kernel, self.kernel))
            if self.bias:
                bias_init = _make_bias_initializer(self.init_mode, fan, fout, self.init_bias)
                self.bias_param = self.param("bias", bias_init, (self.out_channels,))
            else:
                self.bias_param = None
        else:
            self.weight = None
            self.bias_param = None

        self.resample_filter_kernel = None
        if self.up or self.down:
            self.resample_filter_kernel = _resample_kernel(self.resample_filter)

    def __call__(self, x):
        w = None if self.weight is None else self.weight.astype(x.dtype)
        b = None if self.bias_param is None else self.bias_param.astype(x.dtype)
        f = self.resample_filter_kernel.astype(x.dtype) if self.resample_filter_kernel is not None else None
        w_pad = 0 if w is None else w.shape[-1] // 2
        f_pad = 0 if f is None else (f.shape[-1] - 1) // 2

        if self.fused_resample and self.up and w is not None:
            x = _conv_transpose2d(
                x,
                jnp.tile(f * 4.0, (self.in_channels, 1, 1, 1)),
                strides=(2, 2),
                padding=((f.shape[-1] - 1 - max(f_pad - w_pad, 0), f.shape[-1] - 1 - max(f_pad - w_pad, 0)),) * 2,
                groups=self.in_channels,
            )
            x = _conv2d(x, w, padding=((max(w_pad - f_pad, 0), max(w_pad - f_pad, 0)),) * 2)
        elif self.fused_resample and self.down and w is not None:
            x = _conv2d(x, w, padding=((w_pad + f_pad, w_pad + f_pad),) * 2)
            x = _conv2d(
                x,
                jnp.tile(f, (self.out_channels, 1, 1, 1)),
                padding=((f_pad, f_pad),) * 2,
                strides=(2, 2),
                groups=self.out_channels,
            )
        else:
            if self.up:
                x = _conv_transpose2d(
                    x,
                    jnp.tile(f * 4.0, (self.in_channels, 1, 1, 1)),
                    strides=(2, 2),
                    padding=((f.shape[-1] - 1 - f_pad, f.shape[-1] - 1 - f_pad),) * 2,
                    groups=self.in_channels,
                )
            if self.down:
                x = _conv2d(
                    x,
                    jnp.tile(f, (self.in_channels, 1, 1, 1)),
                    padding=((f_pad, f_pad),) * 2,
                    strides=(2, 2),
                    groups=self.in_channels,
                )
            if w is not None:
                x = _conv2d(x, w, padding=((w_pad, w_pad),) * 2)

        if b is not None:
            x = x + b.reshape(1, -1, 1, 1).astype(x.dtype)
        return x


class GroupNorm(nn.Module):
    num_channels: int
    num_groups: int = 32
    min_channels_per_group: int = 4
    eps: float = 1e-5

    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, (self.num_channels,))
        self.bias = self.param("bias", nn.initializers.zeros, (self.num_channels,))

    def __call__(self, x):
        n, c, h, w = x.shape
        g = min(self.num_groups, max(1, c // self.min_channels_per_group))
        assert c % g == 0, (c, g)
        x = x.reshape(n, g, c // g, h, w)
        mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=(2, 3, 4), keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x.reshape(n, c, h, w)
        return x * self.weight.reshape(1, -1, 1, 1).astype(x.dtype) + self.bias.reshape(1, -1, 1, 1).astype(x.dtype)


class AttentionOp:
    @staticmethod
    def apply(q, k):
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32) / np.sqrt(k.shape[1])
        w = jnp.einsum("ncq,nck->nqk", q_f32, k_f32)
        w = jax.nn.softmax(w, axis=2)
        return w.astype(q.dtype)


class UNetBlock(nn.Module):
    in_channels: int
    out_channels: int
    emb_channels: int
    up: bool = False
    down: bool = False
    attention: bool = False
    num_heads: Optional[int] = None
    channels_per_head: int = 64
    dropout: float = 0.0
    skip_scale: float = 1.0
    eps: float = 1e-5
    resample_filter: tuple = (1, 1)
    resample_proj: bool = False
    adaptive_scale: bool = True
    init: dict = None
    init_zero: dict = None
    init_attn: Optional[dict] = None

    def setup(self):
        num_heads = 0 if not self.attention else (self.num_heads if self.num_heads is not None else self.out_channels // self.channels_per_head)
        object.__setattr__(self, "num_heads", num_heads)
        init = self.init or {}
        init_zero = self.init_zero or {}
        init_attn = self.init_attn or init

        self.norm0 = GroupNorm(num_channels=self.in_channels, eps=self.eps)
        self.conv0 = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel=3,
            up=self.up,
            down=self.down,
            resample_filter=self.resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=self.emb_channels,
            out_features=self.out_channels * (2 if self.adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=self.out_channels, eps=self.eps)
        self.conv1 = Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel=3,
            **init_zero,
        )
        self.skip = None
        if self.out_channels != self.in_channels or self.up or self.down:
            kernel = 1 if self.resample_proj or self.out_channels != self.in_channels else 0
            self.skip = Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel=kernel,
                up=self.up,
                down=self.down,
                resample_filter=self.resample_filter,
                **init,
            )

        if num_heads:
            self.norm2 = GroupNorm(num_channels=self.out_channels, eps=self.eps)
            self.qkv = Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels * 3,
                kernel=1,
                **init_attn,
            )
            self.proj = Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel=1,
                **init_zero,
            )
        else:
            self.norm2 = None
            self.qkv = None
            self.proj = None

    def __call__(self, x, emb, train: bool = False):
        orig = x
        x = self.conv0(jax.nn.silu(self.norm0(x)))

        params = self.affine(emb)[:, :, None, None].astype(x.dtype)
        if self.adaptive_scale:
            scale, shift = jnp.split(params, 2, axis=1)
            x = jax.nn.silu(shift + self.norm1(x) * (scale + 1.0))
        else:
            x = jax.nn.silu(self.norm1(x + params))

        if self.dropout > 0 and train:
            keep_prob = 1.0 - self.dropout
            mask = jax.random.bernoulli(self.make_rng("dropout"), keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, jnp.zeros_like(x))
        x = self.conv1(x)
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            qkv = self.qkv(self.norm2(x))
            b, c3, h, w = qkv.shape
            qkv = qkv.reshape(b * self.num_heads, c3 // self.num_heads // 3, 3, -1)
            q, k, v = jnp.moveaxis(qkv, 2, 0)
            weights = AttentionOp.apply(q, k)
            a = jnp.einsum("nqk,nck->ncq", weights, v)
            x = self.proj(a.reshape(b, self.out_channels, h, w)) + x
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False

    def __call__(self, x):
        x = jnp.asarray(x).reshape(-1, 1)
        freqs = jnp.arange(start=0, stop=self.num_channels // 2, dtype=jnp.float32)
        denom = self.num_channels // 2 - (1 if self.endpoint else 0)
        freqs = freqs / denom
        freqs = (1.0 / self.max_positions) ** freqs
        x = x * freqs.astype(x.dtype)
        return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)


class FourierEmbedding(nn.Module):
    num_channels: int
    scale: float = 16.0

    def setup(self):
        self.freqs = self.variable(
            "constants",
            "freqs",
            lambda: jax.random.normal(jax.random.PRNGKey(0), (self.num_channels // 2,)) * self.scale,
        )

    def __call__(self, x):
        x = jnp.asarray(x).reshape(-1, 1)
        x = x * (2 * np.pi * self.freqs.value.astype(x.dtype))
        return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)


class SongUNet(nn.Module):
    img_resolution: int
    in_channels: int
    out_channels: int
    label_dim: int = 0
    augment_dim: int = 0
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2)
    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: tuple = (16,)
    dropout: float = 0.10
    label_dropout: float = 0.0
    embedding_type: str = "positional"
    channel_mult_noise: int = 1
    encoder_type: str = "standard"
    decoder_type: str = "standard"
    resample_filter: tuple = (1, 1)

    def setup(self):
        assert self.embedding_type in ["fourier", "positional"]
        assert self.encoder_type in ["standard", "skip", "residual"]
        assert self.decoder_type in ["standard", "skip"]

        emb_channels = self.model_channels * self.channel_mult_emb
        noise_channels = self.model_channels * self.channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=self.dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=self.resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if self.embedding_type == "positional" else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=self.label_dim, out_features=noise_channels, **init) if self.label_dim else None
        self.map_augment = Linear(in_features=self.augment_dim, out_features=noise_channels, bias=False, **init) if self.augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        enc = []
        cout = self.in_channels
        caux = self.in_channels
        for level, mult in enumerate(self.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = self.model_channels
                enc.append((f"{res}x{res}_conv", Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)))
            else:
                enc.append((f"{res}x{res}_down", UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)))
                if self.encoder_type == "skip":
                    enc.append((f"{res}x{res}_aux_down", Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=self.resample_filter)))
                    enc.append((f"{res}x{res}_aux_skip", Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)))
                if self.encoder_type == "residual":
                    enc.append((f"{res}x{res}_aux_residual", Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=self.resample_filter, fused_resample=True, **init)))
                    caux = cout
            for idx in range(self.num_blocks):
                cin = cout
                cout = self.model_channels * mult
                attn = res in self.attn_resolutions
                enc.append((f"{res}x{res}_block{idx}", UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)))
        self.enc = enc
        skips = [block.out_channels for name, block in self.enc if "aux" not in name]

        dec = []
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = self.img_resolution >> level
            if level == len(self.channel_mult) - 1:
                dec.append((f"{res}x{res}_in0", UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)))
                dec.append((f"{res}x{res}_in1", UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)))
            else:
                dec.append((f"{res}x{res}_up", UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)))
            for idx in range(self.num_blocks + 1):
                cin = cout + skips.pop()
                cout = self.model_channels * mult
                attn = idx == self.num_blocks and res in self.attn_resolutions
                dec.append((f"{res}x{res}_block{idx}", UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)))
            if self.decoder_type == "skip" or level == 0:
                if self.decoder_type == "skip" and level < len(self.channel_mult) - 1:
                    dec.append((f"{res}x{res}_aux_up", Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel=0, up=True, resample_filter=self.resample_filter)))
                dec.append((f"{res}x{res}_aux_norm", GroupNorm(num_channels=cout, eps=1e-6)))
                dec.append((f"{res}x{res}_aux_conv", Conv2d(in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero)))
        self.dec = dec

    def _map(self, noise_labels, class_labels, augment_labels, train):
        emb = self.map_noise(noise_labels)
        emb = jnp.flip(emb.reshape(emb.shape[0], 2, -1), axis=1).reshape(*emb.shape)
        if self.map_label is not None:
            tmp = class_labels
            if train and self.label_dropout:
                rng = self.make_rng("dropout")
                keep = jax.random.uniform(rng, (noise_labels.shape[0], 1)) >= self.label_dropout
                tmp = tmp * keep.astype(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = jax.nn.silu(self.map_layer0(emb))
        emb = jax.nn.silu(self.map_layer1(emb))
        return emb

    def __call__(self, x, noise_labels, class_labels, augment_labels=None, train: bool = False):
        emb = self._map(noise_labels, class_labels, augment_labels, train=train)

        skips = []
        aux = x
        for name, block in self.enc:
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb, train=train) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec:
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(jax.nn.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = jnp.concatenate([x, skips.pop()], axis=1)
                x = block(x, emb, train=train)
        return aux

    def forward_features(self, x, noise_labels, class_labels, augment_labels=None, train: bool = False):
        emb = self._map(noise_labels, class_labels, augment_labels, train=train)

        skips = []
        aux = x
        for name, block in self.enc:
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb, train=train) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec:
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = jax.nn.silu(tmp)
                aux = tmp
            else:
                if x.shape[1] != block.in_channels:
                    x = jnp.concatenate([x, skips.pop()], axis=1)
                x = block(x, emb, train=train)
        return aux


class DhariwalUNet(nn.Module):
    img_resolution: int
    in_channels: int
    out_channels: int
    label_dim: int = 0
    augment_dim: int = 0
    model_channels: int = 192
    channel_mult: tuple = (1, 2, 3, 4)
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_resolutions: tuple = (32, 16, 8)
    dropout: float = 0.10
    label_dropout: float = 0.0

    def setup(self):
        emb_channels = self.model_channels * self.channel_mult_emb
        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=self.dropout, init=init, init_zero=init_zero)

        self.map_noise = PositionalEmbedding(num_channels=self.model_channels)
        self.map_augment = Linear(in_features=self.augment_dim, out_features=self.model_channels, bias=False, **init_zero) if self.augment_dim else None
        self.map_layer0 = Linear(in_features=self.model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=self.label_dim, out_features=emb_channels, bias=False, init_mode="kaiming_normal", init_weight=np.sqrt(self.label_dim)) if self.label_dim else None

        enc = []
        cout = self.in_channels
        for level, mult in enumerate(self.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = self.model_channels * mult
                enc.append((f"{res}x{res}_conv", Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)))
            else:
                enc.append((f"{res}x{res}_down", UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)))
            for idx in range(self.num_blocks):
                cin = cout
                cout = self.model_channels * mult
                enc.append((f"{res}x{res}_block{idx}", UNetBlock(in_channels=cin, out_channels=cout, attention=(res in self.attn_resolutions), **block_kwargs)))
        self.enc = enc
        skips = [block.out_channels for _, block in self.enc]

        dec = []
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = self.img_resolution >> level
            if level == len(self.channel_mult) - 1:
                dec.append((f"{res}x{res}_in0", UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)))
                dec.append((f"{res}x{res}_in1", UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)))
            else:
                dec.append((f"{res}x{res}_up", UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)))
            for idx in range(self.num_blocks + 1):
                cin = cout + skips.pop()
                cout = self.model_channels * mult
                dec.append((f"{res}x{res}_block{idx}", UNetBlock(in_channels=cin, out_channels=cout, attention=(res in self.attn_resolutions), **block_kwargs)))
        self.dec = dec
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero)

    def _map(self, noise_labels, class_labels, augment_labels, train):
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = jax.nn.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if train and self.label_dropout:
                rng = self.make_rng("dropout")
                keep = jax.random.uniform(rng, (noise_labels.shape[0], 1)) >= self.label_dropout
                tmp = tmp * keep.astype(tmp.dtype)
            emb = emb + self.map_label(tmp)
        return jax.nn.silu(emb)

    def __call__(self, x, noise_labels, class_labels, augment_labels=None, train: bool = False):
        emb = self._map(noise_labels, class_labels, augment_labels, train=train)
        skips = []
        for _, block in self.enc:
            x = block(x, emb, train=train) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        for _, block in self.dec:
            if x.shape[1] != block.in_channels:
                x = jnp.concatenate([x, skips.pop()], axis=1)
            x = block(x, emb, train=train)
        x = self.out_conv(jax.nn.silu(self.out_norm(x)))
        return x


class VPPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    beta_d: float = 19.9
    beta_min: float = 0.1
    M: int = 1000
    epsilon_t: float = 1e-5
    model_type: str = "SongUNet"
    model_kwargs: dict = None

    def setup(self):
        self.sigma_min = float(self.sigma(self.epsilon_t))
        self.sigma_max = float(self.sigma(1))
        model_cls = globals()[self.model_type]
        self.model = model_cls(
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.img_channels,
            label_dim=self.label_dim,
            **(self.model_kwargs or {}),
        )

    def sigma(self, t):
        t = jnp.asarray(t)
        return jnp.sqrt(jnp.exp(0.5 * self.beta_d * (t ** 2) + self.beta_min * t) - 1)

    def sigma_inv(self, sigma):
        sigma = jnp.asarray(sigma)
        return (jnp.sqrt(self.beta_min**2 + 2 * self.beta_d * jnp.log1p(sigma**2)) - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, augment_labels=None, train: bool = False, **model_kwargs):
        x = jnp.asarray(x, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(-1, self.label_dim)
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32
        c_skip = 1
        c_out = -sigma
        c_in = 1 / jnp.sqrt(sigma**2 + 1)
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        fx = self.model((c_in * x).astype(dtype), c_noise.flatten(), class_labels=class_labels, augment_labels=augment_labels, train=train, **model_kwargs)
        return c_skip * x + c_out * fx.astype(jnp.float32)


class VEPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min: float = 0.02
    sigma_max: float = 100
    model_type: str = "SongUNet"
    model_kwargs: dict = None

    def setup(self):
        model_cls = globals()[self.model_type]
        self.model = model_cls(
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.img_channels,
            label_dim=self.label_dim,
            **(self.model_kwargs or {}),
        )

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, augment_labels=None, train: bool = False, **model_kwargs):
        x = jnp.asarray(x, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(-1, self.label_dim)
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = jnp.log(0.5 * sigma)
        fx = self.model((c_in * x).astype(dtype), c_noise.flatten(), class_labels=class_labels, augment_labels=augment_labels, train=train, **model_kwargs)
        return c_skip * x + c_out * fx.astype(jnp.float32)


class iDDPMPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    C_1: float = 0.001
    C_2: float = 0.008
    M: int = 1000
    model_type: str = "DhariwalUNet"
    model_kwargs: dict = None

    def setup(self):
        self.model = globals()[self.model_type](
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.img_channels * 2,
            label_dim=self.label_dim,
            **(self.model_kwargs or {}),
        )
        u = jnp.zeros((self.M + 1,), dtype=jnp.float32)
        for j in range(self.M, 0, -1):
            u = u.at[j - 1].set(jnp.sqrt(((u[j] ** 2 + 1) / jnp.clip(self.alpha_bar(j - 1) / self.alpha_bar(j), a_min=self.C_1)) - 1))
        self.u = u
        self.sigma_min = float(u[self.M - 1])
        self.sigma_max = float(u[0])

    def alpha_bar(self, j):
        j = jnp.asarray(j)
        return jnp.sin(0.5 * np.pi * j / self.M / (self.C_2 + 1)) ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = jnp.asarray(sigma)
        diffs = jnp.abs(sigma.reshape(1, -1, 1) - self.u.reshape(1, -1, 1))
        index = jnp.argmin(diffs, axis=2)
        result = index if return_index else self.u[index.flatten()].astype(sigma.dtype)
        return result.reshape(sigma.shape)

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, augment_labels=None, train: bool = False, **model_kwargs):
        x = jnp.asarray(x, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(-1, self.label_dim)
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32
        c_skip = 1
        c_out = -sigma
        c_in = 1 / jnp.sqrt(sigma**2 + 1)
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).astype(jnp.float32)
        fx = self.model((c_in * x).astype(dtype), c_noise.flatten(), class_labels=class_labels, augment_labels=augment_labels, train=train, **model_kwargs)
        return c_skip * x + c_out * fx[:, : self.img_channels].astype(jnp.float32)


class EDMPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min: float = 0.0
    sigma_max: float = float("inf")
    sigma_data: float = 0.5
    model_type: str = "DhariwalUNet"
    model_kwargs: dict = None

    def setup(self):
        model_cls = globals()[self.model_type]
        self.model = model_cls(
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.img_channels,
            label_dim=self.label_dim,
            **(self.model_kwargs or {}),
        )

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, augment_labels=None, train: bool = False, **model_kwargs):
        x = jnp.asarray(x, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = jnp.zeros((1, self.label_dim), dtype=jnp.float32)
        else:
            class_labels = jnp.asarray(class_labels, dtype=jnp.float32).reshape(-1, self.label_dim)
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / jnp.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = jnp.log(sigma) / 4

        fx = self.model((c_in * x).astype(dtype), c_noise.flatten(), class_labels=class_labels, augment_labels=augment_labels, train=train, **model_kwargs)
        return c_skip * x + c_out * fx.astype(jnp.float32)
