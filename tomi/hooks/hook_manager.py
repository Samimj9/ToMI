"""
tomi/hooks/hook_manager.py
--------------------------
``HookManager`` coordinates all ``HookPoint`` instances for a wrapped model.

It provides the primary high-level API for:

* Registering temporary hook functions on named hook points.
* Running the model forward with automatic hook cleanup.
* Collecting activation caches.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from tomi.activations.activation_cache import ActivationCache
from tomi.hooks.cache import make_cache_hook, make_patch_hook
from tomi.hooks.hook_point import HookFn, HookPoint
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class HookManager:
    """Manages a registry of :class:`~tomi.hooks.hook_point.HookPoint` objects.

    Parameters
    ----------
    hook_points:
        Pre-built mapping from hook name → ``HookPoint``.  Models populate
        this at construction time.
    """

    def __init__(self, hook_points: Dict[str, HookPoint]) -> None:
        self._hook_points: Dict[str, HookPoint] = hook_points
        self._temporary_keys: List[Tuple[str, str]] = []  # (hook_name, key)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def hook_names(self) -> List[str]:
        """Sorted list of all available hook point names."""
        return sorted(self._hook_points.keys())

    def __getitem__(self, name: str) -> HookPoint:
        """Return the ``HookPoint`` for *name*.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._hook_points:
            available = "\n  ".join(self.hook_names[:20])
            raise KeyError(
                f"Hook '{name}' not found.  Available hooks (first 20):\n  {available}"
            )
        return self._hook_points[name]

    def __contains__(self, name: str) -> bool:
        return name in self._hook_points

    # ------------------------------------------------------------------
    # Temporary hook registration
    # ------------------------------------------------------------------

    def add_hook(self, name: str, key: str, fn: HookFn) -> None:
        """Register a temporary hook function on a named hook point.

        Parameters
        ----------
        name:
            The ToMI hook name (e.g. ``"blocks.3.attn.out"``).
        key:
            A unique label for this hook (e.g. ``"cache"``, ``"patch"``).
        fn:
            The hook callable.
        """
        self[name].add_hook(key, fn)
        self._temporary_keys.append((name, key))

    def remove_all_hooks(self) -> None:
        """Remove **all** hook functions from all hook points."""
        for point in self._hook_points.values():
            point.remove_hooks()
        self._temporary_keys.clear()
        log.debug("Removed all hooks from all hook points")

    def remove_temporary_hooks(self) -> None:
        """Remove only the hooks added through this manager (via :meth:`add_hook`)."""
        for hook_name, key in self._temporary_keys:
            if hook_name in self._hook_points:
                self._hook_points[hook_name].remove_hook(key)
        self._temporary_keys.clear()

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def hooks(
        self,
        hook_fns: Dict[str, HookFn],
        key_prefix: str = "ctx",
    ) -> Iterator[None]:
        """Context manager: register *hook_fns*, yield, then clean up.

        Parameters
        ----------
        hook_fns:
            Mapping from hook name → hook function.
        key_prefix:
            Prefix for the auto-generated hook keys.

        Yields
        ------
        None

        Example
        -------
        ::

            with manager.hooks({"blocks.0.attn.out": my_fn}):
                model(inputs)
        """
        added: List[Tuple[str, str]] = []
        try:
            for i, (name, fn) in enumerate(hook_fns.items()):
                key = f"{key_prefix}_{i}"
                self[name].add_hook(key, fn)
                added.append((name, key))
            yield
        finally:
            for hook_name, key in added:
                self._hook_points[hook_name].remove_hook(key)

    # ------------------------------------------------------------------
    # High-level: run_with_cache
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        forward_fn: Callable[[], Any],
        names_filter: Optional[List[str]] = None,
        detach: bool = True,
        clone: bool = True,
    ) -> ActivationCache:
        """Run *forward_fn* and collect activations into an :class:`ActivationCache`.

        Parameters
        ----------
        forward_fn:
            A zero-argument callable that performs a forward pass.  It should
            not return a value; the output is captured via hooks.
        names_filter:
            If provided, only cache activations for the given hook names.
            Defaults to all registered hook points.
        detach:
            Detach tensors from the computation graph before caching.
        clone:
            Clone tensors before storing.

        Returns
        -------
        ActivationCache
            A cache object containing all captured activations.
        """
        target_names = names_filter if names_filter is not None else self.hook_names

        storage: Dict[str, torch.Tensor] = {}
        hook_fns: Dict[str, HookFn] = {}
        for name in target_names:
            if name in self._hook_points:
                hook_fns[name] = make_cache_hook(storage, name, detach=detach, clone=clone)

        with self.hooks(hook_fns, key_prefix="cache_run"):
            forward_fn()

        return ActivationCache(storage)

    def run_with_patch(
        self,
        forward_fn: Callable[[], Any],
        patch_name: str,
        patch_tensor: torch.Tensor,
        position: Optional[int] = None,
    ) -> None:
        """Run *forward_fn* with a specific activation patched.

        Parameters
        ----------
        forward_fn:
            Zero-argument callable performing the forward pass.
        patch_name:
            Hook name to patch.
        patch_tensor:
            The tensor to substitute.
        position:
            Optional sequence position to patch.
        """
        patch_fn = make_patch_hook(patch_tensor, position=position)
        with self.hooks({patch_name: patch_fn}, key_prefix="patch_run"):
            forward_fn()

    def clear_all_caches(self) -> None:
        """Clear the ``_cached_output`` of every hook point."""
        for point in self._hook_points.values():
            point.clear_cache()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HookManager("
            f"n_hooks={len(self._hook_points)}, "
            f"active_temp_hooks={len(self._temporary_keys)})"
        )

    def __len__(self) -> int:
        return len(self._hook_points)
