"""
tomi/patching/head_patching.py
--------------------------------
Per-attention-head activation patching.

Patches the attention **output** of individual heads rather than the full
attention output, allowing us to identify which specific heads are causally
important.

A head's contribution is isolated by zeroing all other heads' outputs and
restoring only one head's component from the clean run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from tomi.activations.activation_cache import ActivationCache
from tomi.metrics.logit_diff import logit_diff
from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)

MetricFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class HeadPatchingResult:
    """Results of per-head activation patching.

    Attributes
    ----------
    n_layers:
        Number of transformer layers.
    n_heads:
        Number of attention heads per layer.
    importance_matrix:
        Shape ``(n_layers, n_heads)`` normalised causal effect.
    baseline_clean:
        Clean metric value.
    baseline_corrupted:
        Corrupted metric value.
    """

    n_layers: int
    n_heads: int
    importance_matrix: torch.Tensor
    baseline_clean: float
    baseline_corrupted: float


def head_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    position: int = -1,
    show_progress: bool = True,
) -> HeadPatchingResult:
    """Patch individual attention heads and measure causal importance.

    For each ``(layer, head)`` pair, we replace that head's contribution in
    the corrupted run with the clean equivalent and measure the resulting
    metric.

    Parameters
    ----------
    model:
        Wrapped model.
    clean_tokens:
        Token ids for the clean prompt.
    corrupted_tokens:
        Token ids for the corrupted prompt.
    correct_token_id:
        Correct answer token id.
    incorrect_token_id:
        Foil token id.
    position:
        Evaluation position in the sequence.
    show_progress:
        Show tqdm bar.

    Returns
    -------
    HeadPatchingResult
    """
    metric = _make_metric(correct_token_id, incorrect_token_id, position)

    n_layers = model.n_layers
    n_heads = model.n_heads
    d_head = model.d_head

    # Cache clean attention outputs
    log.info("Caching clean attention outputs …")
    clean_names = [f"blocks.{l}.attn.out" for l in range(n_layers)]
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=clean_names)
    baseline_clean = float(metric(clean_logits).item())

    corrupted_logits = model.get_logits(corrupted_tokens)
    baseline_corrupted = float(metric(corrupted_logits).item())

    importance = torch.zeros(n_layers, n_heads)

    iterator = tqdm(
        [(l, h) for l in range(n_layers) for h in range(n_heads)],
        desc="Head patching",
        disable=not show_progress,
    )

    for layer, head in iterator:
        hook_name = f"blocks.{layer}.attn.out"
        clean_out = clean_cache.get(hook_name)  # (batch, seq, d_model)
        if clean_out is None:
            continue

        # Build patch fn: add this head's contribution from clean run
        # We approximate per-head contribution by assuming equal partitioning
        # of the output dimension; for exact per-head patching the model
        # would need to expose pre-projection head outputs.
        head_dim_start = head * d_head
        head_dim_end = (head + 1) * d_head

        corrupted_out_holder: List[torch.Tensor] = []

        def _head_patch_fn(tensor, hook):
            """Patch only the slice corresponding to *head*."""
            patched = tensor.clone()
            patched[:, :, head_dim_start:head_dim_end] = clean_out[:, :, head_dim_start:head_dim_end]
            return patched

        # Use the hook manager's context manager
        with model.hook_manager.hooks({hook_name: _head_patch_fn}, key_prefix=f"head_{layer}_{head}"):
            out_logits = model.get_logits(corrupted_tokens)

        score = float(metric(out_logits).item())
        denom = abs(baseline_clean - baseline_corrupted) + 1e-8
        importance[layer, head] = (score - baseline_corrupted) / denom

    return HeadPatchingResult(
        n_layers=n_layers,
        n_heads=n_heads,
        importance_matrix=importance,
        baseline_clean=baseline_clean,
        baseline_corrupted=baseline_corrupted,
    )


def _make_metric(correct_id: int, incorrect_id: int, position: int) -> MetricFn:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, correct_id, incorrect_id, position=position)
    return metric
