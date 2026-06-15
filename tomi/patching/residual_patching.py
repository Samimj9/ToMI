"""
tomi/patching/residual_patching.py
------------------------------------
Residual-stream patching across all layers.

Patches the residual stream at a single **token position** and **layer**,
effectively asking: "If the residual stream at position *p* in layer *l*
were replaced with its clean equivalent, how much of the metric do we recover?"

This is the core of the original ROME / causal tracing methodology.
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
class ResidualPatchingResult:
    """Results of residual-stream patching.

    Attributes
    ----------
    n_layers:
        Number of layers.
    n_positions:
        Number of sequence positions.
    importance_matrix:
        Shape ``(n_layers, n_positions)`` normalised causal effect.
    baseline_clean:
        Clean metric.
    baseline_corrupted:
        Corrupted metric.
    token_labels:
        Optional string labels for token positions.
    """

    n_layers: int
    n_positions: int
    importance_matrix: torch.Tensor
    baseline_clean: float
    baseline_corrupted: float
    token_labels: Optional[List[str]] = None


def residual_patching(
    model: ToMModel,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    slot: str = "post",
    metric_position: int = -1,
    show_progress: bool = True,
) -> ResidualPatchingResult:
    """Patch residual stream at each (layer, token-position) and measure recovery.

    Parameters
    ----------
    model:
        Wrapped model.
    clean_tokens:
        Shape ``(1, seq_len)``.
    corrupted_tokens:
        Shape ``(1, seq_len)``.
    correct_token_id:
        Correct answer token.
    incorrect_token_id:
        Foil token.
    slot:
        Which residual slot to patch: ``"pre"``, ``"mid"``, or ``"post"``.
    metric_position:
        Sequence position for metric evaluation.
    show_progress:
        Display tqdm bar.

    Returns
    -------
    ResidualPatchingResult
    """
    metric = _make_metric(correct_token_id, incorrect_token_id, metric_position)
    n_layers = model.n_layers
    seq_len = clean_tokens.shape[1]

    # Cache clean residual stream
    names = [f"blocks.{l}.resid.{slot}" for l in range(n_layers)]
    log.info("Caching clean residual stream (%s) …", slot)
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=names)
    baseline_clean = float(metric(clean_logits).item())

    corrupted_logits = model.get_logits(corrupted_tokens)
    baseline_corrupted = float(metric(corrupted_logits).item())

    importance = torch.zeros(n_layers, seq_len)

    iterator = tqdm(
        [(l, p) for l in range(n_layers) for p in range(seq_len)],
        desc="Residual patching",
        disable=not show_progress,
    )

    for layer, pos in iterator:
        hook_name = f"blocks.{layer}.resid.{slot}"
        clean_resid = clean_cache.get(hook_name)
        if clean_resid is None:
            continue

        def _patch_fn(tensor, hook, _clean=clean_resid, _pos=pos):
            patched = tensor.clone()
            patched[:, _pos, :] = _clean[:, _pos, :]
            return patched

        with model.hook_manager.hooks({hook_name: _patch_fn}, key_prefix=f"resid_{layer}_{pos}"):
            out_logits = model.get_logits(corrupted_tokens)

        score = float(metric(out_logits).item())
        denom = abs(baseline_clean - baseline_corrupted) + 1e-8
        importance[layer, pos] = (score - baseline_corrupted) / denom

    return ResidualPatchingResult(
        n_layers=n_layers,
        n_positions=seq_len,
        importance_matrix=importance,
        baseline_clean=baseline_clean,
        baseline_corrupted=baseline_corrupted,
    )


def _make_metric(correct_id: int, incorrect_id: int, position: int) -> MetricFn:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, correct_id, incorrect_id, position=position)
    return metric
