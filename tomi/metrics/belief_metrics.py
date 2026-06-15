"""
tomi/metrics/belief_metrics.py
--------------------------------
Theory-of-Mind specific metrics.

These metrics evaluate how well a model tracks agent beliefs:

* **belief_score** — probability assigned to the agent's believed location
  vs the true location (when those differ).
* **false_belief_accuracy** — whether the model correctly predicts the
  agent will act on their *false* belief, not the updated world-state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BeliefScoreResult:
    """Result of a belief score evaluation.

    Attributes
    ----------
    belief_logit_diff:
        ``logit(belief_answer) - logit(reality_answer)``.
        Positive means the model correctly weights the agent's belief.
    belief_prob:
        Probability assigned to the belief answer.
    reality_prob:
        Probability assigned to the reality answer.
    is_correct:
        Whether the model's top prediction matches the belief answer.
    """

    belief_logit_diff: float
    belief_prob: float
    reality_prob: float
    is_correct: bool


def belief_score(
    logits: torch.Tensor,
    belief_token_id: int,
    reality_token_id: int,
    position: int = -1,
) -> BeliefScoreResult:
    """Evaluate belief-tracking at a sequence position.

    For a false-belief task:

    * *belief_token_id* is the answer consistent with the agent's (false) belief.
    * *reality_token_id* is the answer consistent with the true world-state.

    A well-calibrated model should assign higher probability to the
    belief answer (reflecting the agent's perspective).

    Parameters
    ----------
    logits:
        Model logits ``(batch, seq_len, vocab_size)``.
    belief_token_id:
        Token id for the agent's believed answer.
    reality_token_id:
        Token id for the actual world-state answer.
    position:
        Sequence position to evaluate (default ``-1``).

    Returns
    -------
    BeliefScoreResult
    """
    # Mean over batch
    pos_logits = logits[:, position, :].mean(dim=0)  # (vocab_size,)
    probs = F.softmax(pos_logits, dim=-1)

    b_logit = pos_logits[belief_token_id].item()
    r_logit = pos_logits[reality_token_id].item()
    b_prob = probs[belief_token_id].item()
    r_prob = probs[reality_token_id].item()

    top_pred = int(pos_logits.argmax().item())
    is_correct = top_pred == belief_token_id

    return BeliefScoreResult(
        belief_logit_diff=b_logit - r_logit,
        belief_prob=b_prob,
        reality_prob=r_prob,
        is_correct=is_correct,
    )


def false_belief_accuracy(
    results: List[BeliefScoreResult],
) -> float:
    """Compute accuracy over a list of :class:`BeliefScoreResult` objects.

    Parameters
    ----------
    results:
        One result per false-belief task instance.

    Returns
    -------
    float
        Fraction of instances where the model correctly predicted the
        agent's (false) belief.
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.is_correct)
    return correct / len(results)


def perspective_taking_score(
    logits_own: torch.Tensor,
    logits_other: torch.Tensor,
    target_token_id: int,
    position: int = -1,
) -> float:
    """Measure how much the model distinguishes between self and other perspectives.

    Computes the difference in predicted probability for *target_token_id*
    between two logit sets: *logits_own* (from the self-perspective prompt)
    and *logits_other* (from the other-agent-perspective prompt).

    Parameters
    ----------
    logits_own:
        Logits for the self-perspective.
    logits_other:
        Logits for the other-agent perspective.
    target_token_id:
        The token whose probability is compared.
    position:
        Sequence position.

    Returns
    -------
    float
        ``P(target | other) - P(target | own)``.
        Positive values suggest the model adapts its prediction to the
        other agent's perspective.
    """
    own_prob = float(F.softmax(logits_own[:, position, :], dim=-1).mean(0)[target_token_id])
    other_prob = float(F.softmax(logits_other[:, position, :], dim=-1).mean(0)[target_token_id])
    return other_prob - own_prob
