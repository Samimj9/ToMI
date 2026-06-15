"""
tests/test_activation_cache.py
--------------------------------
Unit tests for ActivationCache.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tomi.activations.activation_cache import ActivationCache


@pytest.fixture
def sample_cache() -> ActivationCache:
    """An ActivationCache with 4 entries across 2 layers."""
    return ActivationCache({
        "blocks.0.attn.out": torch.randn(1, 5, 32),
        "blocks.0.resid.post": torch.randn(1, 5, 32),
        "blocks.1.attn.out": torch.randn(1, 5, 32),
        "blocks.1.resid.post": torch.randn(1, 5, 32),
    })


class TestActivationCache:
    def test_len(self, sample_cache):
        assert len(sample_cache) == 4

    def test_keys_sorted(self, sample_cache):
        keys = sample_cache.keys()
        assert keys == sorted(keys)

    def test_getitem(self, sample_cache):
        t = sample_cache["blocks.0.attn.out"]
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 5, 32)

    def test_getitem_missing_raises(self, sample_cache):
        with pytest.raises(KeyError):
            _ = sample_cache["blocks.99.attn.out"]

    def test_contains(self, sample_cache):
        assert "blocks.0.attn.out" in sample_cache
        assert "blocks.0.mlp.post" not in sample_cache

    def test_get_default(self, sample_cache):
        assert sample_cache.get("missing") is None

    def test_to_numpy(self, sample_cache):
        np_cache = sample_cache.to_numpy()
        assert isinstance(np_cache, dict)
        for v in np_cache.values():
            assert isinstance(v, np.ndarray)

    def test_to_cpu(self, sample_cache):
        cpu_cache = sample_cache.to_cpu()
        for v in cpu_cache.values():
            assert v.device.type == "cpu"

    def test_filter(self, sample_cache):
        subset = sample_cache.filter("blocks.0")
        assert len(subset) == 2
        assert all(k.startswith("blocks.0") for k in subset.keys())

    def test_filter_by_component(self, sample_cache):
        attn_cache = sample_cache.filter_by_component("attn")
        assert len(attn_cache) == 2
        assert all(".attn." in k for k in attn_cache.keys())

    def test_get_residual_stream(self, sample_cache):
        """Stacking post-residual activations should give (n_layers, B, S, D)."""
        resid = sample_cache.get_residual_stream(slot="post")
        assert resid.shape == (2, 1, 5, 32)

    def test_get_residual_stream_missing_slot_raises(self, sample_cache):
        with pytest.raises(ValueError):
            sample_cache.get_residual_stream(slot="mid")  # not in this cache

    def test_accumulate(self, sample_cache):
        extra = ActivationCache({"blocks.2.attn.out": torch.randn(1, 5, 32)})
        merged = sample_cache.accumulate(extra)
        assert len(merged) == 5
        assert "blocks.2.attn.out" in merged

    def test_detach(self, sample_cache):
        # Add a tensor that requires grad to test detach
        t = torch.randn(1, 3, 16, requires_grad=True)
        cache = ActivationCache({"blocks.0.attn.out": t})
        detached = cache.detach()
        assert not detached["blocks.0.attn.out"].requires_grad

    def test_repr(self, sample_cache):
        r = repr(sample_cache)
        assert "ActivationCache" in r
        assert "n_keys=4" in r
