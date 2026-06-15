"""
tomi/activations/activation_cache.py
-------------------------------------
``ActivationCache`` — a dict-like container for named activation tensors.

Inspired by TransformerLens's ``ActivationCache`` but architecture-agnostic
and extended with statistics helpers and NumPy conversion utilities.

Example
-------
::

    cache = model.run_with_cache(tokens)

    # Access individual activations
    resid = cache["blocks.5.resid.post"]     # (B, S, d_model)
    attn  = cache["blocks.3.attn.pattern"]   # (B, H, S, S)

    # Batch helpers
    print(cache.keys())
    cpu_cache = cache.to_cpu()
    np_cache  = cache.to_numpy()

    # Stack residual stream across layers
    resid_stack = cache.get_residual_stream()   # (n_layers, B, S, d_model)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tomi.utils.logging import get_logger

log = get_logger(__name__)


class ActivationCache:
    """A dict-like store for named activation tensors.

    Parameters
    ----------
    cache:
        Mapping from hook name → activation tensor.
    """

    def __init__(self, cache: Dict[str, torch.Tensor]) -> None:
        self._cache: Dict[str, torch.Tensor] = dict(cache)

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the activation for *name*.

        Raises
        ------
        KeyError
            If *name* is not in the cache.
        """
        if name not in self._cache:
            raise KeyError(
                f"No activation cached for '{name}'.  "
                f"Available keys: {self.keys()[:10]}"
            )
        return self._cache[name]

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def get(
        self,
        name: str,
        default: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the activation for *name*, or *default* if not found."""
        return self._cache.get(name, default)

    def keys(self) -> List[str]:
        """Sorted list of cached hook names."""
        return sorted(self._cache.keys())

    def values(self) -> List[torch.Tensor]:
        """List of cached tensors (in sorted key order)."""
        return [self._cache[k] for k in self.keys()]

    def items(self) -> List[Tuple[str, torch.Tensor]]:
        """List of ``(name, tensor)`` pairs (sorted by name)."""
        return [(k, self._cache[k]) for k in self.keys()]

    # ------------------------------------------------------------------
    # Device / format conversion
    # ------------------------------------------------------------------

    def to_cpu(self) -> "ActivationCache":
        """Return a new :class:`ActivationCache` with all tensors on CPU.

        Returns
        -------
        ActivationCache
        """
        return ActivationCache({k: v.cpu() for k, v in self._cache.items()})

    def to_device(self, device: torch.device) -> "ActivationCache":
        """Return a new cache with all tensors moved to *device*.

        Parameters
        ----------
        device:
            Target device.

        Returns
        -------
        ActivationCache
        """
        return ActivationCache({k: v.to(device) for k, v in self._cache.items()})

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Return a plain dict of ``name → np.ndarray``.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        return {
            k: v.detach().cpu().float().numpy()
            for k, v in self._cache.items()
        }

    def detach(self) -> "ActivationCache":
        """Return a new cache with all tensors detached from the computation graph."""
        return ActivationCache({k: v.detach() for k, v in self._cache.items()})

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter(self, prefix: str) -> "ActivationCache":
        """Return a subset cache containing only keys that start with *prefix*.

        Parameters
        ----------
        prefix:
            Key prefix, e.g. ``"blocks.3"`` or ``"blocks.3.attn"``.

        Returns
        -------
        ActivationCache
        """
        subset = {k: v for k, v in self._cache.items() if k.startswith(prefix)}
        return ActivationCache(subset)

    def filter_by_component(self, component: str) -> "ActivationCache":
        """Return keys whose component matches *component* (``attn``, ``mlp``, ``resid``).

        Parameters
        ----------
        component:
            Component type string.

        Returns
        -------
        ActivationCache
        """
        subset = {
            k: v
            for k, v in self._cache.items()
            if f".{component}." in k
        }
        return ActivationCache(subset)

    # ------------------------------------------------------------------
    # Stacking helpers
    # ------------------------------------------------------------------

    def get_residual_stream(
        self,
        slot: str = "post",
        layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Stack the residual stream across layers.

        Parameters
        ----------
        slot:
            Residual slot: ``"pre"``, ``"mid"``, or ``"post"``.
        layers:
            Layer indices to include.  ``None`` includes all cached layers.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, seq_len, d_model)``.
        """
        available = sorted(
            [k for k in self._cache if f".resid.{slot}" in k],
            key=lambda k: int(k.split(".")[1]),
        )
        if layers is not None:
            available = [k for k in available if int(k.split(".")[1]) in layers]

        if not available:
            raise ValueError(
                f"No residual-stream activations found for slot='{slot}'. "
                f"Available: {self.keys()}"
            )
        tensors = [self._cache[k] for k in available]
        return torch.stack(tensors, dim=0)

    def get_attention_patterns(
        self,
        layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Stack attention patterns across layers.

        Parameters
        ----------
        layers:
            Layer indices to include.  ``None`` includes all.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, n_heads, seq_len, seq_len)``.
        """
        available = sorted(
            [k for k in self._cache if ".attn.pattern" in k],
            key=lambda k: int(k.split(".")[1]),
        )
        if layers is not None:
            available = [k for k in available if int(k.split(".")[1]) in layers]

        if not available:
            raise ValueError("No attention pattern activations found in cache.")
        tensors = [self._cache[k] for k in available]
        return torch.stack(tensors, dim=0)

    def get_mlp_outputs(
        self,
        layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Stack MLP post-activations across layers.

        Parameters
        ----------
        layers:
            Layer indices to include.  ``None`` includes all.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, seq_len, d_mlp)``.
        """
        available = sorted(
            [k for k in self._cache if ".mlp.post" in k],
            key=lambda k: int(k.split(".")[1]),
        )
        if layers is not None:
            available = [k for k in available if int(k.split(".")[1]) in layers]

        if not available:
            raise ValueError("No MLP post-activation tensors found in cache.")
        tensors = [self._cache[k] for k in available]
        return torch.stack(tensors, dim=0)

    def accumulate(self, other: "ActivationCache") -> "ActivationCache":
        """Merge *other* into this cache (keys from *other* overwrite).

        Parameters
        ----------
        other:
            Another ``ActivationCache``.

        Returns
        -------
        ActivationCache
        """
        merged = dict(self._cache)
        merged.update(other._cache)
        return ActivationCache(merged)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._cache)
        sample = self.keys()[:5]
        return f"ActivationCache(n_keys={n}, sample={sample})"
