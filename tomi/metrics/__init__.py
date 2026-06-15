"""tomi.metrics — scalar evaluation metrics for interpretability experiments."""

from tomi.metrics.belief_metrics import (
    BeliefScoreResult,
    belief_score,
    false_belief_accuracy,
    perspective_taking_score,
)
from tomi.metrics.causal_effect import (
    average_total_effect,
    causal_effect,
    indirect_effect,
)
from tomi.metrics.logit_diff import (
    correct_token_prob,
    log_prob_diff,
    logit_diff,
    logit_diff_from_cache,
)

__all__ = [
    "logit_diff",
    "logit_diff_from_cache",
    "log_prob_diff",
    "correct_token_prob",
    "causal_effect",
    "indirect_effect",
    "average_total_effect",
    "belief_score",
    "BeliefScoreResult",
    "false_belief_accuracy",
    "perspective_taking_score",
]
