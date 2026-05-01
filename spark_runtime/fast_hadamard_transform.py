"""Correctness-first fallback for `fast_hadamard_transform`.

The upstream extension does not currently build cleanly in the tested Spark
arm64 CUDA 13 container. DeepSeek-V4 uses Hadamard rotation in small activation
paths, so this fallback keeps the official model code import-compatible while
we work on faster kernels.
"""

from __future__ import annotations

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply a Walsh-Hadamard transform over the last dimension."""
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"last dimension must be a power of two, got {n}")

    y = x.float()
    original_shape = y.shape
    y = y.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (h * 2), h * 2)
        left = y[..., :h]
        right = y[..., h : h * 2]
        y = torch.cat((left + right, left - right), dim=-1)
        h *= 2
    y = y.reshape(original_shape)
    if scale != 1.0:
        y = y * scale
    return y.to(x.dtype)
