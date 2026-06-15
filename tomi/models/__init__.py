"""tomi.models — model loading and architecture adapters."""

from tomi.models.base_model import ToMModel
from tomi.models.llama import LlamaAdapter
from tomi.models.model_loader import load_model
from tomi.models.pythia import PythiaAdapter
from tomi.models.qwen import QwenAdapter
from tomi.models.registry import get_adapter_for_model, lookup_adapter, register_adapter

__all__ = [
    "ToMModel",
    "QwenAdapter",
    "PythiaAdapter",
    "LlamaAdapter",
    "load_model",
    "register_adapter",
    "lookup_adapter",
    "get_adapter_for_model",
]
