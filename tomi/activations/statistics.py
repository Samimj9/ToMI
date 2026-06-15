"""
tomi/activations/statistics.py
--------------------------------
Statistical analysis helpers for activation tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tomi.activations.activation_cache import ActivationCache
from tomi.utils.tensor import to_numpy


@dataclass
class ActivationStats:
    """Summary statistics for a single activation tensor.

    Attributes
    ----------
    name:    Hook name.
    shape:   Tensor shape.
    mean:    Element-wise mean.
    std:     Element-wise std.
    min:     Global minimum.
    max:     Global maximum.
    norm:    L2 norm.
    """

    name: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    norm: float


def compute_stats(name: str, tensor: torch.Tensor) -> ActivationStats:
    """Compute summary statistics for *tensor*.

    Parameters
    ----------
    name:
        Descriptive label.
    tensor:
        Activation tensor (any shape).

    Returns
    -------
    ActivationStats
    """
    t = tensor.detach().float()
    return ActivationStats(
        name=name,
        shape=tuple(t.shape),
        mean=float(t.mean().item()),
        std=float(t.std().item()),
        min=float(t.min().item()),
        max=float(t.max().item()),
        norm=float(t.norm().item()),
    )


def compare_activations(
    cache_a: ActivationCache,
    cache_b: ActivationCache,
    names: Optional[List[str]] = None,
) -> Dict[str, float]:

    if names is None:
        names = sorted(set(cache_a.keys()) & set(cache_b.keys()))

    distances: Dict[str, float] = {}

    for name in names:
        a = cache_a.get(name)
        b = cache_b.get(name)

        if a is None or b is None:
            continue

        # =========================================================
        #  FIX: ALIGN SEQUENCE LENGTH (CRITICAL)
        # =========================================================
        if a.dim() >= 3 and b.dim() >= 3:
            min_len = min(a.shape[1], b.shape[1])
            a = a[:, :min_len]
            b = b[:, :min_len]

        elif a.shape != b.shape:
            # fallback safety for unexpected cases
            min_len = min(a.numel(), b.numel())
            a = a.flatten()[:min_len]
            b = b.flatten()[:min_len]

        diff = a.float() - b.float()
        distances[name] = float(diff.norm().item())

    return distances


def summarise_cache(cache: ActivationCache) -> List[ActivationStats]:
    """Return :class:`ActivationStats` for every entry in *cache*.

    Parameters
    ----------
    cache:
        An :class:`ActivationCache`.

    Returns
    -------
    List[ActivationStats]
    """
    return [compute_stats(name, tensor) for name, tensor in cache.items()]


def top_neurons(
    tensor: torch.Tensor,
    n: int = 20,
    reduction: str = "abs_mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify the top *n* neurons by importance.

    Parameters
    ----------
    tensor:
        MLP activation tensor of shape ``(batch, seq_len, d_mlp)`` or
        ``(seq_len, d_mlp)`` or ``(d_mlp,)``.
    n:
        Number of top neurons to return.
    reduction:
        How to aggregate over non-neuron dimensions:
        ``"abs_mean"`` (default), ``"max"``, ``"mean"``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(values, indices)`` of the top *n* neurons.
    """
    t = tensor.detach().float()
    if t.dim() > 1:
        # Flatten all but last dimension
        t = t.reshape(-1, t.shape[-1])
        if reduction == "abs_mean":
            scores = t.abs().mean(dim=0)
        elif reduction == "max":
            scores = t.abs().max(dim=0).values
        elif reduction == "mean":
            scores = t.mean(dim=0)
        else:
            raise ValueError(f"Unknown reduction: '{reduction}'")
    else:
        scores = t.abs()

    return scores.topk(min(n, scores.shape[0]))
