"""
tomi/models/registry.py
------------------------
Registry that maps HuggingFace model class names to their ToMI adapter.

New architectures can be added by calling :func:`register_adapter` at import
time or at runtime.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)

# Mapping: HuggingFace model class name (str) → ToMModel subclass
_REGISTRY: Dict[str, Type[ToMModel]] = {}


def register_adapter(hf_class_name: str, adapter_cls: Type[ToMModel]) -> None:
    """Register an adapter for a given HuggingFace model class name.

    Parameters
    ----------
    hf_class_name:
        The ``type(model).__name__`` string for the HuggingFace model,
        e.g. ``"Qwen2ForCausalLM"``.
    adapter_cls:
        The :class:`~tomi.models.base_model.ToMModel` subclass to use.
    """
    _REGISTRY[hf_class_name] = adapter_cls
    log.debug("Registered adapter %s → %s", hf_class_name, adapter_cls.__name__)


def lookup_adapter(hf_class_name: str) -> Optional[Type[ToMModel]]:
    """Return the adapter class for *hf_class_name*, or ``None``.

    Parameters
    ----------
    hf_class_name:
        ``type(model).__name__`` of a loaded HuggingFace model.

    Returns
    -------
    Optional[Type[ToMModel]]
    """
    return _REGISTRY.get(hf_class_name)


def get_adapter_for_model(model) -> Type[ToMModel]:
    """Infer and return the correct adapter class for *model*.

    Parameters
    ----------
    model:
        A loaded HuggingFace ``PreTrainedModel``.

    Returns
    -------
    Type[ToMModel]

    Raises
    ------
    ValueError
        If no adapter is registered for the model's class.
    """
    class_name = type(model).__name__
    adapter = lookup_adapter(class_name)
    if adapter is not None:
        return adapter

    # Fallback: check by substring to handle minor naming differences
    for key, val in _REGISTRY.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            log.warning(
                "Fuzzy-matched model class '%s' to adapter '%s' via key '%s'.",
                class_name, val.__name__, key,
            )
            return val

    raise ValueError(
        f"No ToMI adapter registered for model class '{class_name}'.  "
        f"Registered classes: {list(_REGISTRY.keys())}.  "
        "Use tomi.models.registry.register_adapter() to add support."
    )


# ---------------------------------------------------------------------------
# Built-in registrations (populated when models are imported)
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    """Register all built-in adapters."""
    from tomi.models.qwen import QwenAdapter
    from tomi.models.pythia import PythiaAdapter
    from tomi.models.llama import LlamaAdapter

    # Qwen family
    for name in [
        "QwenForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen2_5ForCausalLM",
        "QwenLMHeadModel",
    ]:
        register_adapter(name, QwenAdapter)

    # Pythia / GPT-NeoX family
    for name in [
        "GPTNeoXForCausalLM",
        "PythiaForCausalLM",
    ]:
        register_adapter(name, PythiaAdapter)

    # LLaMA / Mistral family
    for name in [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "Llama3ForCausalLM",
    ]:
        register_adapter(name, LlamaAdapter)


_register_defaults()
