"""JAX-side augmentation wrapper.

The current training path applies augmentations on the PyTorch dataset tensors
before conversion to JAX. This module exists so the JAX tree has the same
entry point as the original code and can be expanded to a full Flax/JAX
implementation later without changing callers.
"""

from __future__ import annotations


class AugmentPipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, images):
        return images, None

