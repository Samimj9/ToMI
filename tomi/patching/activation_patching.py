"""
tomi/patching/activation_patching.py
--------------------------------------
Core activation patching engine.

Activation patching (also called "causal tracing") works by:

1. Running the model on a **clean** prompt and caching all activations.
2. Running the model on a **corrupted** prompt (e.g. subject is changed).
3. For each hook point, running the corrupted forward pass but replacing the
   activation at that point with the clean cached value.
4. Measuring the metric (e.g. logit diff) after each substitution.

Components with high metric recovery are causally important.

Example
-------
::

    from tomi.patching import activation_patching

    result = activation_patching(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
    )

    print(result.importance_matrix)     # shape (n_layers, n_components)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from tomi.activations.activation_cache import ActivationCache
from tomi.hooks.cache import make_patch_hook
from tomi.metrics.logit_diff import logit_diff
from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)

MetricFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class PatchingResult:
    """Container for activation patching results.

    Attributes
    ----------
    hook_names:
        The hook points that were patched, in order.
    metric_values:
        Metric value achieved after patching each hook (same order).
    baseline_clean:
        Metric on the fully-clean run.
    baseline_corrupted:
        Metric on the fully-corrupted run.
    importance_matrix:
        Normalised causal effect for each hook.  Shape matches
        *hook_names*.
    """

    hook_names: List[str]
    metric_values: List[float]
    baseline_clean: float
    baseline_corrupted: float
    importance_matrix: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.importance_matrix:
            denom = abs(self.baseline_clean - self.baseline_corrupted) + 1e-8
            self.importance_matrix = [
                (v - self.baseline_corrupted) / denom
                for v in self.metric_values
            ]

    def as_layer_matrix(
        self,
        n_layers: int,
        components: Sequence[str] = ("resid.post", "attn.out", "mlp.post"),
    ) -> torch.Tensor:
        """Reshape importance scores into a ``(n_layers, n_components)`` matrix.

        Parameters
        ----------
        n_layers:
            Number of layers.
        components:
            Component slots to include.

        Returns
        -------
        torch.Tensor
            Shape ``(n_layers, len(components))``.
        """
        mat = torch.zeros(n_layers, len(components))
        name_to_score = dict(zip(self.hook_names, self.importance_matrix))
        for layer in range(n_layers):
            for j, comp in enumerate(components):
                key = f"blocks.{layer}.{comp}"
                mat[layer, j] = name_to_score.get(key, 0.0)
        return mat


def activation_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    names_filter: Optional[List[str]] = None,
    position: int = -1,
    show_progress: bool = True,
) -> PatchingResult:
    """Run activation patching over all (or selected) hook points.

    Parameters
    ----------
    model:
        A :class:`~tomi.models.base_model.ToMModel` instance.
    clean_tokens:
        Token ids for the clean prompt ``(1, seq_len)``.
    corrupted_tokens:
        Token ids for the corrupted prompt ``(1, seq_len)``.
    correct_token_id:
        Token id of the correct answer.
    incorrect_token_id:
        Token id of the incorrect (foil) answer.
    names_filter:
        Restrict patching to these hook names.  Defaults to all.
    position:
        Sequence position at which to evaluate the metric.
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    PatchingResult
    """
    metric = _make_logit_diff_metric(correct_token_id, incorrect_token_id, position)
    return _run_patching(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        metric=metric,
        names_filter=names_filter,
        show_progress=show_progress,
    )


def _make_logit_diff_metric(
    correct_id: int,
    incorrect_id: int,
    position: int,
) -> MetricFn:
    """Construct a logit-diff metric callable."""
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, correct_id, incorrect_id, position=position)
    return metric


def _run_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric: MetricFn,
    names_filter: Optional[List[str]] = None,
    show_progress: bool = True,
) -> PatchingResult:
    """Internal patching loop (separated for reuse by head/neuron patchers)."""
    # Step 1: cache clean activations
    log.info("Caching clean activations …")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    baseline_clean = float(metric(clean_logits).item())

    # Step 2: baseline metric on corrupted run
    log.info("Computing corrupted baseline …")
    corrupted_logits = model.get_logits(corrupted_tokens)
    baseline_corrupted = float(metric(corrupted_logits).item())

    log.info(
        "Baseline — clean: %.4f | corrupted: %.4f",
        baseline_clean,
        baseline_corrupted,
    )

    # Step 3: iterate over hook points
    target_names = names_filter if names_filter is not None else model.hook_names
    hook_names: List[str] = []
    metric_values: List[float] = []

    iterator = tqdm(target_names, desc="Patching", disable=not show_progress)
    for name in iterator:
        if name not in clean_cache:
            continue
        clean_act = clean_cache[name]
        patched_logits = model.patch_activation(
            input_ids=corrupted_tokens,
            hook_name=name,
            patch_tensor=clean_act,
        )
        score = float(metric(patched_logits).item())
        hook_names.append(name)
        metric_values.append(score)

    return PatchingResult(
        hook_names=hook_names,
        metric_values=metric_values,
        baseline_clean=baseline_clean,
        baseline_corrupted=baseline_corrupted,
    )
