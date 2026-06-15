"""
tomi/metrics/causal_effect.py
------------------------------
Causal-effect metrics used to interpret activation patching results.

The key quantities:

* **Normalised Causal Effect (NCE)**: how much restoring a clean activation
  at a given hook recovers the clean performance, relative to a fully-clean
  baseline.  Values near 1.0 indicate that a component is causally important.

* **Direct Indirect Effect (DIE)**: used in path patching / causal mediation
  analysis to decompose total effects.
"""

from __future__ import annotations

from typing import Optional

import torch


def causal_effect(
    baseline_score: torch.Tensor,
    patched_score: torch.Tensor,
    corrupted_score: Optional[torch.Tensor] = None,
    normalise: bool = True,
) -> torch.Tensor:
    """Compute the causal effect of a patch.

    Parameters
    ----------
    baseline_score:
        Metric value on the clean (unpatched) run.
    patched_score:
        Metric value after patching the activation.
    corrupted_score:
        Metric value on the fully-corrupted run.  Required when
        *normalise* is ``True``.
    normalise:
        Whether to normalise relative to the clean–corrupted range.

    Returns
    -------
    torch.Tensor
        If *normalise*: a value in [0, 1] where 1 means full recovery.
        Otherwise, the raw difference ``patched_score - corrupted_score``.

    Raises
    ------
    ValueError
        If *normalise* is ``True`` but *corrupted_score* is ``None``.
    """
    if normalise:
        if corrupted_score is None:
            raise ValueError(
                "corrupted_score is required when normalise=True."
            )
        denom = (baseline_score - corrupted_score).abs()
        if denom.abs() < 1e-8:
            return torch.zeros_like(baseline_score)
        return (patched_score - corrupted_score) / (denom + 1e-8)
    else:
        return patched_score - (corrupted_score if corrupted_score is not None else 0.0)


def indirect_effect(
    total_effect: torch.Tensor,
    direct_effect: torch.Tensor,
) -> torch.Tensor:
    """Compute the indirect effect.

    Parameters
    ----------
    total_effect:
        Total causal effect of an intervention.
    direct_effect:
        Direct path effect (holding mediators constant).

    Returns
    -------
    torch.Tensor
        ``total_effect - direct_effect``.
    """
    return total_effect - direct_effect


def average_total_effect(
    clean_scores: torch.Tensor,
    corrupted_scores: torch.Tensor,
) -> torch.Tensor:
    """Compute the average total effect across a batch.

    Parameters
    ----------
    clean_scores:
        Metric values on clean inputs ``(n,)``.
    corrupted_scores:
        Metric values on corrupted inputs ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar mean total effect.
    """
    return (clean_scores - corrupted_scores).mean()
