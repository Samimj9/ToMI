"""
tomi/patching/neuron_patching.py
----------------------------------
MLP neuron-level activation patching.

Identifies which individual MLP neurons (post-activation) are causally
important by patching them one at a time from the clean cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from tomi.metrics.logit_diff import logit_diff
from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)

MetricFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class NeuronPatchingResult:
    """Results of neuron-level patching.

    Attributes
    ----------
    layer:
        The layer that was analysed.
    importance_per_neuron:
        Normalised causal effect for each neuron.  Shape ``(d_mlp,)``.
    top_neurons:
        Indices of the *k* most important neurons.
    baseline_clean:
        Clean metric.
    baseline_corrupted:
        Corrupted metric.
    """

    layer: int
    importance_per_neuron: torch.Tensor
    top_neurons: torch.Tensor
    baseline_clean: float
    baseline_corrupted: float


def neuron_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    layer: int = 0,
    token_position: int = -1,
    top_k: int = 20,
    show_progress: bool = True,
) -> NeuronPatchingResult:
    """Patch individual MLP neurons in *layer* and measure causal importance.

    Parameters
    ----------
    model:
        Wrapped model.
    clean_tokens:
        Clean prompt tokens.
    corrupted_tokens:
        Corrupted prompt tokens.
    correct_token_id:
        Correct answer token.
    incorrect_token_id:
        Foil token.
    layer:
        Transformer layer to analyse.
    token_position:
        Token position at which to patch the neuron.
    top_k:
        Number of top neurons to highlight.
    show_progress:
        Display tqdm bar.

    Returns
    -------
    NeuronPatchingResult
    """
    metric = _make_metric(correct_token_id, incorrect_token_id, -1)
    hook_name = f"blocks.{layer}.mlp.post"

    log.info("Caching clean MLP post-activations for layer %d …", layer)
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=[hook_name])
    baseline_clean = float(metric(clean_logits).item())

    corrupted_logits = model.get_logits(corrupted_tokens)
    baseline_corrupted = float(metric(corrupted_logits).item())

    clean_post = clean_cache.get(hook_name)
    if clean_post is None:
        raise RuntimeError(f"Could not cache '{hook_name}'. Check hook registration.")

    d_mlp = clean_post.shape[-1]
    importance = torch.zeros(d_mlp)

    # Use position -1 if negative
    pos = token_position if token_position >= 0 else clean_tokens.shape[1] + token_position

    iterator = tqdm(range(d_mlp), desc=f"Neuron patching (layer {layer})", disable=not show_progress)

    for neuron in iterator:
        def _patch_fn(tensor, hook, _n=neuron, _clean=clean_post, _pos=pos):
            patched = tensor.clone()
            patched[:, _pos, _n] = _clean[:, _pos, _n]
            return patched

        with model.hook_manager.hooks({hook_name: _patch_fn}, key_prefix=f"neuron_{neuron}"):
            out_logits = model.get_logits(corrupted_tokens)

        score = float(metric(out_logits).item())
        denom = abs(baseline_clean - baseline_corrupted) + 1e-8
        importance[neuron] = (score - baseline_corrupted) / denom

    top_vals, top_idx = importance.topk(min(top_k, d_mlp))

    return NeuronPatchingResult(
        layer=layer,
        importance_per_neuron=importance,
        top_neurons=top_idx,
        baseline_clean=baseline_clean,
        baseline_corrupted=baseline_corrupted,
    )


def _make_metric(correct_id: int, incorrect_id: int, position: int) -> MetricFn:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, correct_id, incorrect_id, position=position)
    return metric
