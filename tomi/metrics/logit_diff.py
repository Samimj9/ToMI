"""
tomi/metrics/logit_diff.py
---------------------------
Logit-difference metric — the primary scalar metric for activation patching.

``logit_diff(logits, correct_token_id, incorrect_token_id)`` measures how
much the model prefers the *correct* token over an *incorrect* baseline.

It is positive when the model ranks the correct token above the incorrect one.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from tomi.utils.logging import get_logger

log = get_logger(__name__)


def logit_diff(
    logits: torch.Tensor,
    correct_token_id: Union[int, torch.Tensor],
    incorrect_token_id: Union[int, torch.Tensor],
    position: int = -1,
    mean_over_batch: bool = True,
) -> torch.Tensor:
    """Compute logit difference at a specific sequence position.

    Parameters
    ----------
    logits:
        Model logits of shape ``(batch, seq_len, vocab_size)``.
    correct_token_id:
        Token id of the correct / expected next token.
    incorrect_token_id:
        Token id of the incorrect / foil token.
    position:
        Sequence position to evaluate at. Defaults to ``-1`` (last token).
    mean_over_batch:
        Average over the batch dimension.

    Returns
    -------
    torch.Tensor
        Scalar (if *mean_over_batch*) or shape ``(batch,)`` tensor.
    """
    # Extract logits at the target position: (batch, vocab_size)
    pos_logits = logits[:, position, :]

    correct_logit = pos_logits[:, correct_token_id]    # (batch,)
    incorrect_logit = pos_logits[:, incorrect_token_id]  # (batch,)

    diff = correct_logit - incorrect_logit  # (batch,)

    if mean_over_batch:
        return diff.mean()
    return diff


def logit_diff_from_cache(
    logits: torch.Tensor,
    token_pairs: List[tuple[int, int]],
    positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """Compute logit differences for multiple (correct, incorrect) pairs.

    Parameters
    ----------
    logits:
        Shape ``(batch, seq_len, vocab_size)``.
    token_pairs:
        List of ``(correct_id, incorrect_id)`` tuples.
    positions:
        Sequence positions to evaluate (one per pair).
        Defaults to ``-1`` for all pairs.

    Returns
    -------
    torch.Tensor
        Shape ``(len(token_pairs),)`` with one scalar per pair.
    """
    if positions is None:
        positions = [-1] * len(token_pairs)

    results = []
    for (correct_id, incorrect_id), pos in zip(token_pairs, positions):
        diff = logit_diff(logits, correct_id, incorrect_id, position=pos)
        results.append(diff)
    return torch.stack(results)


def log_prob_diff(
    logits: torch.Tensor,
    correct_token_id: int,
    incorrect_token_id: int,
    position: int = -1,
) -> torch.Tensor:
    """Compute log-probability difference at a sequence position.

    Parameters
    ----------
    logits:
        Shape ``(batch, seq_len, vocab_size)``.
    correct_token_id:
        Token id of the correct token.
    incorrect_token_id:
        Foil token id.
    position:
        Sequence position to evaluate.

    Returns
    -------
    torch.Tensor
        Scalar log-prob difference.
    """
    log_probs = torch.nn.functional.log_softmax(logits[:, position, :], dim=-1)
    correct_lp = log_probs[:, correct_token_id].mean()
    incorrect_lp = log_probs[:, incorrect_token_id].mean()
    return correct_lp - incorrect_lp


def correct_token_prob(
    logits: torch.Tensor,
    correct_token_id: int,
    position: int = -1,
) -> torch.Tensor:
    """Return the probability assigned to *correct_token_id*.

    Parameters
    ----------
    logits:
        Shape ``(batch, seq_len, vocab_size)``.
    correct_token_id:
        Target token id.
    position:
        Sequence position.

    Returns
    -------
    torch.Tensor
        Scalar probability.
    """
    probs = torch.nn.functional.softmax(logits[:, position, :], dim=-1)
    return probs[:, correct_token_id].mean()
