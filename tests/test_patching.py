"""
tests/test_patching.py
-----------------------
Unit tests for patching result dataclasses and utility functions
(no model loading required).
"""

from __future__ import annotations

import pytest
import torch

from tomi.patching.activation_patching import PatchingResult
from tomi.patching.head_patching import HeadPatchingResult


class TestPatchingResult:
    @pytest.fixture
    def result(self) -> PatchingResult:
        return PatchingResult(
            hook_names=["blocks.0.attn.out", "blocks.1.attn.out", "blocks.0.mlp.post"],
            metric_values=[4.0, 3.5, 2.0],
            baseline_clean=5.0,
            baseline_corrupted=1.0,
        )

    def test_importance_normalised(self, result):
        """Importance scores should be in [0, 1] when metric is between clean and corrupted."""
        for score in result.importance_matrix:
            assert 0.0 <= score <= 1.0

    def test_full_recovery_score_is_one(self, result):
        """A metric equal to baseline_clean should give importance 1.0."""
        r = PatchingResult(
            hook_names=["blocks.0.resid.post"],
            metric_values=[5.0],  # == baseline_clean
            baseline_clean=5.0,
            baseline_corrupted=1.0,
        )
        assert r.importance_matrix[0] == pytest.approx(1.0)

    def test_no_recovery_score_is_zero(self, result):
        r = PatchingResult(
            hook_names=["blocks.0.resid.post"],
            metric_values=[1.0],  # == baseline_corrupted
            baseline_clean=5.0,
            baseline_corrupted=1.0,
        )
        assert r.importance_matrix[0] == pytest.approx(0.0)

    def test_as_layer_matrix_shape(self, result):
        mat = result.as_layer_matrix(n_layers=2, components=("attn.out", "mlp.post"))
        assert mat.shape == (2, 2)

    def test_as_layer_matrix_values(self, result):
        mat = result.as_layer_matrix(n_layers=2, components=("attn.out",))
        # blocks.0.attn.out importance ≈ (4.0 - 1.0) / (5.0 - 1.0) = 0.75
        assert mat[0, 0] == pytest.approx(0.75, abs=1e-4)
        # blocks.1.attn.out ≈ (3.5 - 1.0) / 4.0 = 0.625
        assert mat[1, 0] == pytest.approx(0.625, abs=1e-4)


class TestHeadPatchingResult:
    @pytest.fixture
    def head_result(self) -> HeadPatchingResult:
        importance = torch.zeros(4, 8)
        importance[2, 3] = 0.9  # strong head at layer 2, head 3
        importance[1, 0] = 0.4
        return HeadPatchingResult(
            n_layers=4,
            n_heads=8,
            importance_matrix=importance,
            baseline_clean=5.0,
            baseline_corrupted=1.0,
        )

    def test_shape(self, head_result):
        assert head_result.importance_matrix.shape == (4, 8)

    def test_max_value(self, head_result):
        assert head_result.importance_matrix.max().item() == pytest.approx(0.9)

    def test_top_head_position(self, head_result):
        mat = head_result.importance_matrix
        idx = mat.argmax()
        layer = idx.item() // 8
        head = idx.item() % 8
        assert layer == 2
        assert head == 3
