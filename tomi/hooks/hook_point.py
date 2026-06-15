"""
tomi/hooks/hook_point.py
------------------------
``HookPoint`` is the fundamental unit of the ToMI hooking system.

A ``HookPoint`` wraps a single point in a model's computation graph.
It can be attached to any ``nn.Module`` as a forward hook and supports:

* **caching** — recording activations for later inspection.
* **patching** — replacing activations with arbitrary tensors or functions.
* **multiple hooks** — several named callbacks can be registered simultaneously.

Usage example::

    point = HookPoint(name="blocks.3.attn.out")

    # Register a caching callback
    point.add_hook("cache", lambda tensor, hook: tensor)

    # Attach to a module
    point.attach(module)

    # After a forward pass
    print(point.output)

    # Remove all hooks
    point.remove_hooks()
    point.detach()
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tomi.utils.logging import get_logger

log = get_logger(__name__)

# Type alias for hook functions
HookFn = Callable[[torch.Tensor, "HookPoint"], torch.Tensor]


class HookPoint(nn.Module):
    """A transparent pass-through module that exposes a named hook point.

    Parameters
    ----------
    name:
        The canonical ToMI hook name for this point (e.g. ``"blocks.3.attn.out"``).

    Notes
    -----
    ``HookPoint`` is itself an ``nn.Module`` so it can be inserted inline in
    a model's ``forward`` method.  When no hooks are registered the forward
    pass is a no-op (identity).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name
        self._hooks: Dict[str, HookFn] = {}
        self._cached_output: Optional[torch.Tensor] = None
        self._pytorch_hooks: List[Any] = []  # handle references for removal

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_hook(self, key: str, fn: HookFn) -> "HookPoint":
        """Register a hook function.

        Parameters
        ----------
        key:
            Unique identifier for this hook (e.g. ``"cache"``, ``"patch"``).
        fn:
            A callable ``(tensor, hook_point) -> tensor``.  It receives the
            current activation tensor and this ``HookPoint``, and should
            return the (possibly modified) tensor.

        Returns
        -------
        HookPoint
            Self, for chaining.
        """
        if key in self._hooks:
            log.debug("Replacing existing hook '%s' on '%s'", key, self.name)
        self._hooks[key] = fn
        log.debug("Added hook '%s' to '%s'", key, self.name)
        return self

    def remove_hook(self, key: str) -> "HookPoint":
        """Remove a specific hook by *key*.

        Parameters
        ----------
        key:
            The identifier used when the hook was registered.

        Returns
        -------
        HookPoint
            Self, for chaining.
        """
        if key in self._hooks:
            del self._hooks[key]
            log.debug("Removed hook '%s' from '%s'", key, self.name)
        return self

    def remove_hooks(self) -> "HookPoint":
        """Remove **all** registered hook functions.

        Returns
        -------
        HookPoint
            Self, for chaining.
        """
        self._hooks.clear()
        self._cached_output = None
        log.debug("Cleared all hooks from '%s'", self.name)
        return self

    def has_hooks(self) -> bool:
        """Return ``True`` if any hook functions are registered."""
        return bool(self._hooks)

    def hook_keys(self) -> List[str]:
        """Return the list of currently registered hook keys."""
        return list(self._hooks.keys())

    # ------------------------------------------------------------------
    # Attachment to external PyTorch modules
    # ------------------------------------------------------------------

    def attach(self, module: nn.Module) -> "HookPoint":
        """Register a PyTorch forward hook on *module* that routes through this ``HookPoint``.

        This is used when we cannot insert the ``HookPoint`` inline (e.g. for
        adapted models whose internals we do not control).

        Parameters
        ----------
        module:
            The ``nn.Module`` whose output we want to intercept.

        Returns
        -------
        HookPoint
            Self, for chaining.
        """
        handle = module.register_forward_hook(self._pytorch_hook_fn)
        self._pytorch_hooks.append(handle)
        log.debug("Attached '%s' to module %s", self.name, type(module).__name__)
        return self

    def detach(self) -> "HookPoint":
        """Remove all PyTorch-level hooks registered via :meth:`attach`.

        Returns
        -------
        HookPoint
            Self, for chaining.
        """
        for handle in self._pytorch_hooks:
            handle.remove()
        self._pytorch_hooks.clear()
        log.debug("Detached '%s' from all external modules", self.name)
        return self

    def _pytorch_hook_fn(
        self,
        module: nn.Module,
        input: Tuple[Any, ...],
        output: Any,
    ) -> Optional[torch.Tensor]:
        """Internal PyTorch forward-hook callback."""
        if isinstance(output, torch.Tensor):
            return self.forward(output)
        # For tuple outputs, intercept only the first tensor element
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            patched = self.forward(output[0])
            return (patched,) + output[1:]
        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all registered hooks on *x* in registration order.

        Parameters
        ----------
        x:
            The activation tensor at this point in the model.

        Returns
        -------
        torch.Tensor
            The (possibly modified) tensor after all hooks have run.
        """
        for key, fn in self._hooks.items():
            try:
                x = fn(x, self)
            except Exception as exc:
                log.error(
                    "Hook '%s' on '%s' raised an exception: %s", key, self.name, exc
                )
                raise
        self._cached_output = x
        return x

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def output(self) -> Optional[torch.Tensor]:
        """The most recently cached output tensor, or ``None``."""
        return self._cached_output

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the last cached output, or ``None``."""
        if self._cached_output is not None:
            return tuple(self._cached_output.shape)
        return None

    def clear_cache(self) -> None:
        """Discard the cached output tensor."""
        self._cached_output = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        hooks = list(self._hooks.keys())
        return (
            f"HookPoint(name='{self.name}', "
            f"hooks={hooks}, "
            f"cached={'yes' if self._cached_output is not None else 'no'})"
        )
