"""
tomi/models/model_loader.py
----------------------------
High-level entry point: :func:`load_model`.

Loads a HuggingFace model + tokenizer and wraps it in the appropriate
ToMI adapter.

Example
-------
::

    import tomi

    model = tomi.load_model("Qwen/Qwen2-0.5B-Instruct")
    logits, cache = model.run_with_cache(tokens)
"""

from __future__ import annotations

from typing import Optional, Type, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from tomi.models.base_model import ToMModel
from tomi.models.registry import get_adapter_for_model
from tomi.utils.device import resolve_device
from tomi.utils.logging import get_logger

log = get_logger(__name__)


def load_model(
    model_name_or_path: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    adapter_cls: Optional[Type[ToMModel]] = None,
    trust_remote_code: bool = True,
    **hf_kwargs,
) -> ToMModel:
    """Load a model and wrap it in a ToMI adapter.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model name or local path (e.g. ``"Qwen/Qwen2-0.5B"``).
    device:
        Target device.  ``None`` auto-selects CUDA > MPS > CPU.
    dtype:
        Model dtype (e.g. ``torch.float16``).  ``None`` uses the model's
        default (usually ``float32``).
    adapter_cls:
        Explicit adapter class to use.  ``None`` auto-detects via the
        registry.
    trust_remote_code:
        Passed to HuggingFace ``from_pretrained``.
    **hf_kwargs:
        Additional keyword arguments forwarded to
        ``AutoModelForCausalLM.from_pretrained``.

    Returns
    -------
    ToMModel
        An architecture-specific wrapper exposing the ToMI API.

    Examples
    --------
    >>> model = load_model("Qwen/Qwen2-0.5B")
    >>> tokens = model.tokenize("Hello, world!")
    >>> logits, cache = model.run_with_cache(tokens["input_ids"])
    """
    resolved_device = resolve_device(device)
    log.info("Loading '%s' on %s …", model_name_or_path, resolved_device)

    # Determine dtype
    if dtype is None and resolved_device.type == "cuda":
        dtype = torch.float16
    if dtype is None:
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    log.debug("Tokenizer loaded: %s", type(tokenizer).__name__)

    # Load model
    load_kwargs = dict(
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    load_kwargs.update(hf_kwargs)

    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    hf_model = hf_model.to(resolved_device)
    hf_model.eval()
    log.debug("HuggingFace model loaded: %s", type(hf_model).__name__)

    # Select adapter
    if adapter_cls is None:
        adapter_cls = get_adapter_for_model(hf_model)
    log.info("Using adapter: %s", adapter_cls.__name__)

    return adapter_cls(
        model=hf_model,
        tokenizer=tokenizer,
        device=resolved_device,
    )
