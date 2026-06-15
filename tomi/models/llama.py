"""
tomi/models/llama.py
---------------------
``LlamaAdapter`` — wraps LLaMA 2/3 and Mistral models to expose the
universal ToMI hook interface.

LLaMA uses the same block layout as Qwen, so ``LlamaAdapter`` subclasses
``QwenAdapter`` and only overrides the layer-accessor methods where the
HuggingFace attribute names differ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tomi.models.qwen import QwenAdapter
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class LlamaAdapter(QwenAdapter):
    """ToMI wrapper for LLaMA-2, LLaMA-3, and Mistral models.

    The HuggingFace implementations share the same attribute structure as
    Qwen2 (``model.model.layers``, ``model.lm_head``, etc.), so only minor
    overrides are needed.

    Parameters
    ----------
    model:
        A loaded ``LlamaForCausalLM`` or ``MistralForCausalLM``.
    tokenizer:
        Corresponding tokenizer.
    device:
        Device the model resides on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        super().__init__(model, tokenizer, device)

    # LLaMA/Mistral use the same HF structure as Qwen2:
    # - model.model.layers
    # - model.model.embed_tokens
    # - model.model.norm
    # - model.lm_head
    # So no overrides are needed beyond what QwenAdapter already does.

    # The only architecture difference worth noting: Mistral uses sliding-
    # window attention, but that doesn't affect hook registration.
