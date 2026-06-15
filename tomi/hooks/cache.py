"""
tomi/hooks/cache.py
-------------------
Utilities for building hook functions that cache activations.

These are the building blocks used by ``HookManager.run_with_cache`` to
capture activations without modifying the forward pass.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import torch

from tomi.hooks.hook_point import HookFn, HookPoint


def make_cache_hook(
    storage: Dict[str, torch.Tensor],
    name: str,
    detach: bool = True,
    clone: bool = True,
) -> HookFn:
    """Create a hook function that stores the activation in *storage*.

    Parameters
    ----------
    storage:
        A dictionary to write results into.  The key will be *name*.
    name:
        Hook name (used as the dictionary key and for logging).
    detach:
        Whether to detach the tensor from the computation graph.
    clone:
        Whether to clone the tensor (avoids aliasing issues when the same
        buffer is reused across calls).

    Returns
    -------
    HookFn
        A callable suitable for use with :meth:`HookPoint.add_hook`.
    """
    def _cache_hook(tensor: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        t = tensor
        if detach:
            t = t.detach()
        if clone:
            t = t.clone()
        storage[name] = t
        return tensor  # always return the original (unmodified)

    _cache_hook.__name__ = f"cache_hook[{name}]"
    return _cache_hook


def make_patch_hook(
    patch_tensor: torch.Tensor,
    position: Optional[int] = None,
) -> HookFn:
    """Create a hook that replaces an activation with *patch_tensor*.

    Parameters
    ----------
    patch_tensor:
        The tensor to substitute in place of the original activation.
    position:
        If provided, only replace the token at this sequence position
        (i.e. ``tensor[:, position, :]``).

    Returns
    -------
    HookFn
    """
    def _patch_hook(tensor: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        if position is not None:
            tensor = tensor.clone()
            tensor[:, position, :] = patch_tensor[:, position, :]
            return tensor
        return patch_tensor

    _patch_hook.__name__ = "patch_hook"
    return _patch_hook


def make_fn_patch_hook(fn: "HookFn") -> HookFn:
    """Wrap an arbitrary function as a patching hook.

    Parameters
    ----------
    fn:
        A callable ``(tensor, hook) -> tensor``.

    Returns
    -------
    HookFn
    """
    return fn
