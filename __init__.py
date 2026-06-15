"""
tomi — Theory of Mind Interpretability
=======================================

A mechanistic interpretability framework for discovering Theory-of-Mind
circuits in large language models.

Quick start
-----------
::

    import tomi

    # Load a model
    model = tomi.load_model("Qwen/Qwen2-0.5B")

    # Tokenise
    tokens = model.tokenize("Where will Sally look for the marble?")
    input_ids = tokens["input_ids"]

    # Run with activation cache
    logits, cache = model.run_with_cache(input_ids)

    # Inspect a residual stream
    resid = cache["blocks.11.resid.post"]

    # Activation patching
    result = tomi.activation_patching(
        model=model,
        clean_tokens=clean_ids,
        corrupted_tokens=corrupted_ids,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
    )
"""

from __future__ import annotations

# ---- version ---------------------------------------------------------------
__version__ = "0.1.0"
__author__ = "ToMI Contributors"
__license__ = "MIT"

# ---- public API ------------------------------------------------------------

# Model loading
from tomi.models.model_loader import load_model
from tomi.models.base_model import ToMModel
from tomi.models.registry import register_adapter

# Activations
from tomi.activations.activation_cache import ActivationCache
from tomi.activations.activation_store import ActivationStore

# Hooks
from tomi.hooks.hook_point import HookPoint
from tomi.hooks.hook_manager import HookManager
from tomi.hooks.naming import (
    attn_hook,
    mlp_hook,
    resid_hook,
    embed_hook,
    unembed_hook,
    parse_hook_name,
    validate_hook_name,
)

# Metrics
from tomi.metrics.logit_diff import logit_diff, log_prob_diff, correct_token_prob
from tomi.metrics.causal_effect import causal_effect
from tomi.metrics.belief_metrics import belief_score, false_belief_accuracy

# Patching
from tomi.patching.activation_patching import activation_patching, PatchingResult
from tomi.patching.head_patching import head_patching, HeadPatchingResult
from tomi.patching.neuron_patching import neuron_patching, NeuronPatchingResult
from tomi.patching.residual_patching import residual_patching, ResidualPatchingResult
from tomi.patching.attribution_patching import attribution_patching, AttributionPatchingResult

# Circuits
from tomi.circuits.node import CircuitNode, NodeType
from tomi.circuits.edge import CircuitEdge
from tomi.circuits.graph import CircuitGraph
from tomi.circuits.circuit_finder import CircuitFinder

# Theory of Mind datasets
from tomi.theory_of_mind.false_belief import (
    FalseBelief,
    make_sally_anne,
    make_maxi,
    build_false_belief_dataset,
)
from tomi.theory_of_mind.perspective_taking import (
    PerspectiveTakingTask,
    make_visual_access_task,
    build_perspective_taking_dataset,
)
from tomi.theory_of_mind.belief_tracking import (
    BeliefTrackingTask,
    build_belief_tracking_dataset,
)
from tomi.theory_of_mind.evaluation import ToMEvaluator, EvaluationReport

# Visualization
from tomi.visualization.heatmaps import plot_patching_heatmap, plot_layer_importance
from tomi.visualization.attention import plot_attention_pattern, plot_head_importance
from tomi.visualization.circuits import plot_circuit_graph
from tomi.visualization.patching import (
    plot_activation_patching_results,
    plot_head_patching_results,
    plot_residual_patching_results,
)

# Utilities
from tomi.utils.logging import set_log_level, enable_debug, enable_info, silence
from tomi.utils.device import get_best_device


__all__ = [
    # Core
    "load_model",
    "ToMModel",
    "register_adapter",
    # Activations
    "ActivationCache",
    "ActivationStore",
    # Hooks
    "HookPoint",
    "HookManager",
    "attn_hook",
    "mlp_hook",
    "resid_hook",
    "embed_hook",
    "unembed_hook",
    "parse_hook_name",
    "validate_hook_name",
    # Metrics
    "logit_diff",
    "log_prob_diff",
    "correct_token_prob",
    "causal_effect",
    "belief_score",
    "false_belief_accuracy",
    # Patching
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
    # Circuits
    "CircuitNode",
    "NodeType",
    "CircuitEdge",
    "CircuitGraph",
    "CircuitFinder",
    # Theory of Mind
    "FalseBelief",
    "make_sally_anne",
    "make_maxi",
    "build_false_belief_dataset",
    "PerspectiveTakingTask",
    "make_visual_access_task",
    "build_perspective_taking_dataset",
    "BeliefTrackingTask",
    "build_belief_tracking_dataset",
    "ToMEvaluator",
    "EvaluationReport",
    # Visualization
    "plot_patching_heatmap",
    "plot_layer_importance",
    "plot_attention_pattern",
    "plot_head_importance",
    "plot_circuit_graph",
    "plot_activation_patching_results",
    "plot_head_patching_results",
    "plot_residual_patching_results",
    # Utilities
    "set_log_level",
    "enable_debug",
    "enable_info",
    "silence",
    "get_best_device",
]
