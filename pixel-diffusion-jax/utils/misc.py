"""Miscellaneous helpers for the JAX port."""

from __future__ import annotations

import contextlib
import numpy as np
import torch


def set_random_seed(seed, rank=0):
    np.random.seed((seed * 1 + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5, s_max=None):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.sampled_sigmas = {}
        self.buffer_factor = (1 + 1 / (s_max - 1)) ** 0.5 if s_max is not None else 1

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        rnd_normal = np.random.normal(0, 1)
        sigma = np.exp(rnd_normal * 1.2 - 1.2)
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                filename = self.dataset._image_fnames[order[i]]
                sample_annotation = self.dataset.annotations.get(filename, (0.0, 300.0))
                if isinstance(sample_annotation, float):
                    sample_annotation = (sample_annotation, 0.0)
                sample_sigma_min = sample_annotation[0]
                sample_sigma_max = sample_annotation[1]
                if (sigma > sample_sigma_min * self.buffer_factor) or (sigma < sample_sigma_max):
                    self.sampled_sigmas[filename] = sigma
                    yield order[i]
                    rnd_normal = np.random.normal(0, 1)
                    sigma = np.exp(rnd_normal * 1.2 - 1.2)
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

