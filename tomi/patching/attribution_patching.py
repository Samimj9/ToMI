"""
tomi/patching/attribution_patching.py
----------------------------------------
Attribution patching (AP) — a fast gradient-based approximation to
activation patching.

Instead of running a full forward pass for every hook point (O(n_hooks)
passes), attribution patching uses the gradient of the metric w.r.t.
each activation multiplied by the clean–corrupted activation difference.
This reduces the cost to ~2 forward passes + 1 backward pass.

Reference:
    Nanda et al., "Attribution Patching Outperforms Automated Circuit Discovery",
    2023. https://arxiv.org/abs/2310.10348
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from tomi.activations.activation_cache import ActivationCache
from tomi.metrics.logit_diff import logit_diff
from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)

MetricFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AttributionPatchingResult:
    """Results of attribution patching.

    Attributes
    ----------
    hook_names:
        Hook names in the order they were processed.
    attribution_scores:
        Estimated causal importance for each hook (same order).
    baseline_clean:
        Metric on the fully-clean run.
    baseline_corrupted:
        Metric on the fully-corrupted run.
    """

    hook_names: List[str]
    attribution_scores: List[float]
    baseline_clean: float
    baseline_corrupted: float

    def as_dict(self) -> Dict[str, float]:
        """Return a mapping from hook name → attribution score."""
        return dict(zip(self.hook_names, self.attribution_scores))

    def top_k(self, k: int = 10) -> List[tuple[str, float]]:
        """Return the *k* highest-scoring (name, score) pairs.

        Parameters
        ----------
        k:
            Number of top results.

        Returns
        -------
        List[tuple[str, float]]
        """
        pairs = sorted(
            zip(self.hook_names, self.attribution_scores),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return pairs[:k]


def attribution_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    names_filter: Optional[List[str]] = None,
    position: int = -1,
) -> AttributionPatchingResult:
    """Run gradient-based attribution patching.

    Parameters
    ----------
    model:
        Wrapped model.
    clean_tokens:
        Token ids for the clean prompt ``(1, seq_len)``.
    corrupted_tokens:
        Token ids for the corrupted prompt ``(1, seq_len)``.
    correct_token_id:
        Correct answer token id.
    incorrect_token_id:
        Foil token id.
    names_filter:
        Hook names to include.  ``None`` uses all registered hooks.
    position:
        Sequence position for metric evaluation.

    Returns
    -------
    AttributionPatchingResult
    """
    metric = _make_metric(correct_token_id, incorrect_token_id, position)

    target_names = names_filter if names_filter is not None else model.hook_names

    # ------------------------------------------------------------------ #
    # Step 1: Cache clean activations (no grad needed)                     #
    # ------------------------------------------------------------------ #
    log.info("Caching clean activations for attribution patching …")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=target_names
        )
    baseline_clean = float(metric(clean_logits).item())

    # ------------------------------------------------------------------ #
    # Step 2: Cache corrupted activations WITH gradients                   #
    # ------------------------------------------------------------------ #
    log.info("Caching corrupted activations (with gradient) …")
    grad_storage: Dict[str, torch.Tensor] = {}
    act_storage: Dict[str, torch.Tensor] = {}

    # Register hooks that capture activations and retain grad
    handles = []
    for name in target_names:
        if name not in model.hook_manager:
            continue

        def _make_grad_hook(n):
            def _hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
                tensor = tensor.detach().requires_grad_(True)
                act_storage[n] = tensor

                def _backward_hook(grad):
                    grad_storage[n] = grad.detach()

                tensor.register_hook(_backward_hook)
                return tensor
            return _hook_fn

        model.hook_manager.add_hook(name, f"attr_{name}", _make_grad_hook(name))

    try:
        # Forward on corrupted
        corrupted_logits, _ = model.run_with_cache(
            corrupted_tokens, names_filter=target_names
        )
        baseline_corrupted = float(metric(corrupted_logits.detach()).item())

        # Backward on corrupted logits
        score = metric(corrupted_logits)
        score.backward()

    finally:
        model.remove_hooks()

    # ------------------------------------------------------------------ #
    # Step 3: Compute attribution scores                                    #
    # ------------------------------------------------------------------ #
    hook_names: List[str] = []
    attribution_scores: List[float] = []

    for name in target_names:
        grad = grad_storage.get(name)
        act_c = act_storage.get(name)
        act_clean = clean_cache.get(name)

        if grad is None or act_c is None or act_clean is None:
            continue

        # Δ = clean − corrupted
        delta = (act_clean.float() - act_c.float())
        # Attribution ≈ ⟨∇_a metric, Δa⟩
        score_val = float((grad.float() * delta).sum().item())
        hook_names.append(name)
        attribution_scores.append(score_val)

    log.info(
        "Attribution patching complete: %d hooks scored.", len(hook_names)
    )

    return AttributionPatchingResult(
        hook_names=hook_names,
        attribution_scores=attribution_scores,
        baseline_clean=baseline_clean,
        baseline_corrupted=baseline_corrupted,
    )


def _make_metric(correct_id: int, incorrect_id: int, position: int) -> MetricFn:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, correct_id, incorrect_id, position=position)
    return metric
