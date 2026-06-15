"""
tests/test_metrics.py
----------------------
Unit tests for metrics: logit_diff, causal_effect, belief_score.
"""

from __future__ import annotations

import pytest
import torch

from tomi.metrics.logit_diff import logit_diff, log_prob_diff, correct_token_prob
from tomi.metrics.causal_effect import causal_effect, indirect_effect, average_total_effect
from tomi.metrics.belief_metrics import belief_score, false_belief_accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logits(vocab_size: int = 100, batch: int = 1, seq: int = 5) -> torch.Tensor:
    """Make deterministic logits with token 0 highest, token 1 second."""
    logits = torch.zeros(batch, seq, vocab_size)
    logits[:, -1, 0] = 10.0   # correct token = 0
    logits[:, -1, 1] = 5.0    # incorrect token = 1
    return logits


# ---------------------------------------------------------------------------
# logit_diff
# ---------------------------------------------------------------------------

class TestLogitDiff:
    def test_positive_when_correct_higher(self):
        logits = make_logits()
        diff = logit_diff(logits, correct_token_id=0, incorrect_token_id=1)
        assert float(diff.item()) == pytest.approx(5.0)

    def test_negative_when_incorrect_higher(self):
        logits = make_logits()
        diff = logit_diff(logits, correct_token_id=1, incorrect_token_id=0)
        assert float(diff.item()) == pytest.approx(-5.0)

    def test_shape_batch_false(self):
        logits = make_logits(batch=4)
        diff = logit_diff(logits, 0, 1, mean_over_batch=False)
        assert diff.shape == (4,)

    def test_custom_position(self):
        logits = torch.zeros(1, 10, 50)
        logits[0, 3, 5] = 8.0
        logits[0, 3, 7] = 2.0
        diff = logit_diff(logits, correct_token_id=5, incorrect_token_id=7, position=3)
        assert float(diff.item()) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# log_prob_diff
# ---------------------------------------------------------------------------

class TestLogProbDiff:
    def test_positive_when_correct_higher(self):
        logits = make_logits()
        diff = log_prob_diff(logits, correct_token_id=0, incorrect_token_id=1)
        assert float(diff.item()) > 0.0


# ---------------------------------------------------------------------------
# correct_token_prob
# ---------------------------------------------------------------------------

class TestCorrectTokenProb:
    def test_prob_between_0_and_1(self):
        logits = make_logits()
        prob = correct_token_prob(logits, 0)
        assert 0.0 < float(prob.item()) < 1.0

    def test_highest_logit_has_highest_prob(self):
        logits = make_logits()
        prob_0 = correct_token_prob(logits, 0)
        prob_1 = correct_token_prob(logits, 1)
        assert float(prob_0.item()) > float(prob_1.item())


# ---------------------------------------------------------------------------
# causal_effect
# ---------------------------------------------------------------------------

class TestCausalEffect:
    def test_full_recovery_gives_one(self):
        """Patched score == clean score → NCE = 1."""
        clean = torch.tensor(5.0)
        corrupted = torch.tensor(1.0)
        patched = torch.tensor(5.0)
        nce = causal_effect(clean, patched, corrupted, normalise=True)
        assert float(nce.item()) == pytest.approx(1.0, abs=1e-5)

    def test_no_recovery_gives_zero(self):
        """Patched score == corrupted score → NCE = 0."""
        clean = torch.tensor(5.0)
        corrupted = torch.tensor(1.0)
        patched = torch.tensor(1.0)
        nce = causal_effect(clean, patched, corrupted, normalise=True)
        assert float(nce.item()) == pytest.approx(0.0, abs=1e-5)

    def test_partial_recovery(self):
        clean = torch.tensor(4.0)
        corrupted = torch.tensor(0.0)
        patched = torch.tensor(2.0)
        nce = causal_effect(clean, patched, corrupted, normalise=True)
        assert float(nce.item()) == pytest.approx(0.5, abs=1e-4)

    def test_no_normalise(self):
        clean = torch.tensor(4.0)
        corrupted = torch.tensor(1.0)
        patched = torch.tensor(3.0)
        diff = causal_effect(clean, patched, corrupted_score=corrupted, normalise=False)
        assert float(diff.item()) == pytest.approx(2.0)

    def test_normalise_requires_corrupted(self):
        with pytest.raises(ValueError):
            causal_effect(torch.tensor(1.0), torch.tensor(1.0), normalise=True)

    def test_indirect_effect(self):
        total = torch.tensor(3.0)
        direct = torch.tensor(1.0)
        ie = indirect_effect(total, direct)
        assert float(ie.item()) == pytest.approx(2.0)

    def test_average_total_effect(self):
        clean = torch.tensor([4.0, 5.0])
        corrupted = torch.tensor([1.0, 2.0])
        ate = average_total_effect(clean, corrupted)
        assert float(ate.item()) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# belief_score
# ---------------------------------------------------------------------------

class TestBeliefScore:
    def test_correct_when_belief_logit_highest(self):
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 10] = 8.0  # belief token
        logits[0, -1, 20] = 3.0  # reality token
        result = belief_score(logits, belief_token_id=10, reality_token_id=20)
        assert result.is_correct
        assert result.belief_logit_diff == pytest.approx(5.0, abs=1e-4)
        assert result.belief_prob > result.reality_prob

    def test_incorrect_when_reality_logit_highest(self):
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 10] = 2.0  # belief
        logits[0, -1, 20] = 9.0  # reality (higher)
        result = belief_score(logits, belief_token_id=10, reality_token_id=20)
        assert not result.is_correct
        assert result.belief_logit_diff < 0.0


class TestFalseBeliefAccuracy:
    def test_empty(self):
        from tomi.metrics.belief_metrics import false_belief_accuracy
        assert false_belief_accuracy([]) == 0.0

    def test_all_correct(self):
        from tomi.metrics.belief_metrics import BeliefScoreResult, false_belief_accuracy
        results = [
            BeliefScoreResult(1.0, 0.8, 0.2, True),
            BeliefScoreResult(2.0, 0.9, 0.1, True),
        ]
        assert false_belief_accuracy(results) == pytest.approx(1.0)

    def test_half_correct(self):
        from tomi.metrics.belief_metrics import BeliefScoreResult, false_belief_accuracy
        results = [
            BeliefScoreResult(1.0, 0.8, 0.2, True),
            BeliefScoreResult(-1.0, 0.2, 0.8, False),
        ]
        assert false_belief_accuracy(results) == pytest.approx(0.5)
