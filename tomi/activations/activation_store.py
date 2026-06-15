"""
tomi/activations/activation_store.py
--------------------------------------
``ActivationStore`` accumulates activations across many forward passes.

Useful when:

* Running many prompts and aggregating statistics.
* Building datasets of activations for probing or SAE training.
* Collecting baseline vs patched activations for attribution patching.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from tomi.activations.activation_cache import ActivationCache
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class ActivationStore:
    """Accumulates :class:`ActivationCache` objects across multiple runs.

    Parameters
    ----------
    max_runs:
        Maximum number of caches to store before the oldest is evicted.
        ``None`` means unlimited.
    """

    def __init__(self, max_runs: Optional[int] = None) -> None:
        self._caches: List[ActivationCache] = []
        self._max_runs = max_runs

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add(self, cache: ActivationCache) -> None:
        """Append *cache* to the store.

        Parameters
        ----------
        cache:
            An :class:`ActivationCache` from a single forward pass.
        """
        if self._max_runs is not None and len(self._caches) >= self._max_runs:
            self._caches.pop(0)
        self._caches.append(cache)

    def clear(self) -> None:
        """Discard all stored caches."""
        self._caches.clear()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def mean(self, name: str) -> torch.Tensor:
        """Compute the element-wise mean of cached *name* across all runs.

        Parameters
        ----------
        name:
            Hook name.

        Returns
        -------
        torch.Tensor
        """
        tensors = self._gather(name)
        return torch.stack(tensors, dim=0).mean(dim=0)

    def std(self, name: str) -> torch.Tensor:
        """Compute the element-wise standard deviation across all runs.

        Parameters
        ----------
        name:
            Hook name.

        Returns
        -------
        torch.Tensor
        """
        tensors = self._gather(name)
        return torch.stack(tensors, dim=0).std(dim=0)

    def cat(self, name: str, dim: int = 0) -> torch.Tensor:
        """Concatenate tensors for *name* along *dim*.

        Parameters
        ----------
        name:
            Hook name.
        dim:
            Concatenation dimension (typically the batch dimension).

        Returns
        -------
        torch.Tensor
        """
        tensors = self._gather(name)
        return torch.cat(tensors, dim=dim)

    def stack(self, name: str) -> torch.Tensor:
        """Stack tensors for *name* into a new leading dimension.

        Parameters
        ----------
        name:
            Hook name.

        Returns
        -------
        torch.Tensor
            Shape ``(n_runs, ...)``.
        """
        tensors = self._gather(name)
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gather(self, name: str) -> List[torch.Tensor]:
        """Return a list of tensors for *name* across all caches."""
        if not self._caches:
            raise RuntimeError("ActivationStore is empty.")
        tensors = []
        for i, cache in enumerate(self._caches):
            t = cache.get(name)
            if t is None:
                log.warning("Cache %d missing key '%s'; skipping.", i, name)
                continue
            tensors.append(t)
        if not tensors:
            raise KeyError(f"No caches contain key '{name}'.")
        return tensors

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_runs(self) -> int:
        """Number of stored caches."""
        return len(self._caches)

    @property
    def available_keys(self) -> List[str]:
        """Sorted list of hook names present in at least one cache."""
        keys: set = set()
        for cache in self._caches:
            keys.update(cache.keys())
        return sorted(keys)

    def __len__(self) -> int:
        return len(self._caches)

    def __repr__(self) -> str:
        return (
            f"ActivationStore("
            f"n_runs={self.n_runs}, "
            f"max_runs={self._max_runs})"
        )
