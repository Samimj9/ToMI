"""tomi.patching — activation, head, neuron, residual, and attribution patching."""

from tomi.patching.activation_patching import PatchingResult, activation_patching
from tomi.patching.attribution_patching import (
    AttributionPatchingResult,
    attribution_patching,
)
from tomi.patching.head_patching import HeadPatchingResult, head_patching
from tomi.patching.neuron_patching import NeuronPatchingResult, neuron_patching
from tomi.patching.residual_patching import ResidualPatchingResult, residual_patching

__all__ = [
    "activation_patching",
    "PatchingResult",
    "head_patching",
    "HeadPatchingResult",
    "neuron_patching",
    "NeuronPatchingResult",
    "residual_patching",
    "ResidualPatchingResult",
    "attribution_patching",
    "AttributionPatchingResult",
]
