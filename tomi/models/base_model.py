"""
tomi/models/base_model.py
--------------------------
``ToMModel`` — the abstract base class for all architecture-specific
model wrappers in ToMI.

Every concrete adapter (QwenAdapter, PythiaAdapter, …) must subclass
``ToMModel`` and implement the abstract methods.  The base class provides:

* Common properties (``n_layers``, ``d_model``, etc.)
* The public API that all research code should depend on
  (``run_with_cache``, ``patch_activation``, …)
* Helper methods that forward to the :class:`~tomi.hooks.HookManager`
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tomi.activations.activation_cache import ActivationCache
from tomi.hooks.hook_manager import HookManager
from tomi.hooks.hook_point import HookFn
from tomi.utils.logging import get_logger
from tomi.utils.tokenizer import decode_tokens, tokenize

log = get_logger(__name__)


class ToMModel(abc.ABC):
    """Abstract base class for all ToMI model wrappers.

    Subclasses must implement the ``_build_hook_points`` method and expose
    the underlying HuggingFace model as ``self.model``.

    Parameters
    ----------
    model:
        A loaded HuggingFace ``PreTrainedModel``.
    tokenizer:
        Corresponding tokenizer.
    device:
        Device the model is on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.device: torch.device = device

        # Ensure pad token is set (many models don't set one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.debug("Set pad_token = eos_token ('%s')", self.tokenizer.eos_token)

        # Build the hook infrastructure (delegated to subclass)
        self._hook_manager: HookManager = self._build_hook_manager()
        log.info(
            "Loaded %s with %d hook points on %s",
            type(self).__name__,
            len(self._hook_manager),
            device,
        )

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by each adapter)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_hook_manager(self) -> HookManager:
        """Construct the :class:`HookManager` for this architecture.

        Subclasses should:
        1. Identify all relevant sub-modules.
        2. Create a :class:`~tomi.hooks.hook_point.HookPoint` for each.
        3. Attach the hook points to the sub-modules.
        4. Return a :class:`HookManager` wrapping the resulting dict.

        Returns
        -------
        HookManager
        """
        ...

    @property
    @abc.abstractmethod
    def n_layers(self) -> int:
        """Number of transformer layers."""
        ...

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        """Model (residual stream) dimensionality."""
        ...

    @property
    @abc.abstractmethod
    def n_heads(self) -> int:
        """Number of attention heads per layer."""
        ...

    @property
    @abc.abstractmethod
    def d_head(self) -> int:
        """Dimensionality of each attention head."""
        ...

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenise *text* and return input tensors on the model's device.

        Parameters
        ----------
        text:
            A string or list of strings.
        padding:
            Pad sequences to the same length.
        truncation:
            Truncate to *max_length*.
        max_length:
            Optional maximum token length.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``{input_ids, attention_mask, …}`` on ``self.device``.
        """
        return tokenize(
            self.tokenizer,
            text,
            device=self.device,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Run a forward pass and return raw HuggingFace model output.

        Parameters
        ----------
        input_ids:
            Token IDs on the correct device.
        attention_mask:
            Optional attention mask.
        **kwargs:
            Additional keyword arguments forwarded to the model.

        Returns
        -------
        transformers.ModelOutput
        """
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        names_filter: Optional[List[str]] = None,
        return_logits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], ActivationCache]:
        """Run a forward pass and return (logits, cache).

        Parameters
        ----------
        input_ids:
            Token ID tensor ``(batch, seq_len)`` on ``self.device``.
        attention_mask:
            Optional mask.
        names_filter:
            Restrict caching to specific hook names.
        return_logits:
            Whether to include logits in the return value.

        Returns
        -------
        Tuple[Optional[torch.Tensor], ActivationCache]
            ``(logits, cache)`` where *logits* has shape
            ``(batch, seq_len, vocab_size)`` or ``None``.
        """
        logits_holder: List[torch.Tensor] = []

        def _forward_fn() -> None:
            output = self._forward(input_ids, attention_mask)
            if return_logits:
                logits_holder.append(output.logits)

        cache = self._hook_manager.run_with_cache(
            forward_fn=_forward_fn,
            names_filter=names_filter,
        )
        logits = logits_holder[0] if logits_holder else None
        return logits, cache

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits for *input_ids*.

        Parameters
        ----------
        input_ids:
            Token IDs ``(batch, seq_len)``.
        attention_mask:
            Optional mask.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, vocab_size)``.
        """
        output = self._forward(input_ids, attention_mask)
        return output.logits

    def get_activation(
        self,
        input_ids: torch.Tensor,
        hook_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the activation at a specific hook point.

        Parameters
        ----------
        input_ids:
            Token IDs ``(batch, seq_len)``.
        hook_name:
            The ToMI hook name to capture.
        attention_mask:
            Optional mask.

        Returns
        -------
        torch.Tensor
        """
        _, cache = self.run_with_cache(
            input_ids,
            attention_mask,
            names_filter=[hook_name],
        )
        return cache[hook_name]

    def patch_activation(
        self,
        input_ids: torch.Tensor,
        hook_name: str,
        patch_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position: Optional[int] = None,
    ) -> torch.Tensor:
        """Run a forward pass with a specific activation replaced.

        Parameters
        ----------
        input_ids:
            Token IDs ``(batch, seq_len)``.
        hook_name:
            The hook point to patch.
        patch_tensor:
            Tensor to substitute.
        attention_mask:
            Optional mask.
        position:
            Optional sequence position to restrict the patch to.

        Returns
        -------
        torch.Tensor
            Logits from the patched forward pass.
        """
        logits_holder: List[torch.Tensor] = []

        def _forward_fn() -> None:
            output = self._forward(input_ids, attention_mask)
            logits_holder.append(output.logits)

        self._hook_manager.run_with_patch(
            forward_fn=_forward_fn,
            patch_name=hook_name,
            patch_tensor=patch_tensor,
            position=position,
        )
        return logits_holder[0]

    def add_hook(self, hook_name: str, fn: HookFn, key: str = "user") -> None:
        """Permanently register a hook function on *hook_name*.

        Parameters
        ----------
        hook_name:
            Target hook point.
        fn:
            The hook callable.
        key:
            Identifier (default ``"user"``).
        """
        self._hook_manager.add_hook(hook_name, key, fn)

    def remove_hooks(self) -> None:
        """Remove all currently registered hook functions."""
        self._hook_manager.remove_all_hooks()

    # ------------------------------------------------------------------
    # Convenient activation accessors
    # ------------------------------------------------------------------

    def get_attention_heads(
        self,
        input_ids: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return attention output tensors stacked over layers.

        Parameters
        ----------
        input_ids:
            Token IDs.
        layers:
            Layer indices to include (default: all).
        attention_mask:
            Optional mask.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, seq_len, d_model)``.
        """
        if layers is None:
            layers = list(range(self.n_layers))
        names = [f"blocks.{l}.attn.out" for l in layers]
        _, cache = self.run_with_cache(input_ids, attention_mask, names_filter=names)
        tensors = [cache[n] for n in names]
        return torch.stack(tensors, dim=0)

    def get_mlp_outputs(
        self,
        input_ids: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return MLP output tensors stacked over layers.

        Parameters
        ----------
        input_ids:
            Token IDs.
        layers:
            Layer indices (default: all).
        attention_mask:
            Optional mask.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, seq_len, d_mlp)``.
        """
        if layers is None:
            layers = list(range(self.n_layers))
        names = [f"blocks.{l}.mlp.post" for l in layers]
        _, cache = self.run_with_cache(input_ids, attention_mask, names_filter=names)
        tensors = [cache[n] for n in names]
        return torch.stack(tensors, dim=0)

    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        slot: str = "post",
        layers: Optional[Sequence[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return residual stream tensors stacked over layers.

        Parameters
        ----------
        input_ids:
            Token IDs.
        slot:
            ``"pre"``, ``"mid"``, or ``"post"``.
        layers:
            Layer indices (default: all).
        attention_mask:
            Optional mask.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, batch, seq_len, d_model)``.
        """
        if layers is None:
            layers = list(range(self.n_layers))
        names = [f"blocks.{l}.resid.{slot}" for l in layers]
        _, cache = self.run_with_cache(input_ids, attention_mask, names_filter=names)
        return cache.get_residual_stream(slot=slot, layers=layers)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text tokens using HuggingFace ``generate``.

        Parameters
        ----------
        input_ids:
            Prompt token IDs.
        max_new_tokens:
            Number of tokens to generate.
        temperature:
            Sampling temperature.
        do_sample:
            Enable sampling (vs greedy decoding).
        attention_mask:
            Optional mask.
        **kwargs:
            Additional kwargs forwarded to ``model.generate``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len + max_new_tokens)``.
        """
        gen_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            return self.model.generate(**gen_kwargs)

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def hook_manager(self) -> HookManager:
        """The underlying :class:`HookManager`."""
        return self._hook_manager

    @property
    def hook_names(self) -> List[str]:
        """Sorted list of all available hook point names."""
        return self._hook_manager.hook_names

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_layers={self.n_layers}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"device={self.device})"
        )
