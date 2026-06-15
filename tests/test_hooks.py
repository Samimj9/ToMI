"""
tests/test_hooks.py
--------------------
Unit tests for the HookPoint and HookManager infrastructure.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tomi.hooks.hook_point import HookPoint
from tomi.hooks.hook_manager import HookManager
from tomi.hooks.naming import (
    attn_hook,
    mlp_hook,
    resid_hook,
    parse_hook_name,
    validate_hook_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_hook_point() -> HookPoint:
    return HookPoint(name="blocks.0.attn.out")


@pytest.fixture
def two_point_manager() -> HookManager:
    points = {
        "blocks.0.attn.out": HookPoint("blocks.0.attn.out"),
        "blocks.1.mlp.post": HookPoint("blocks.1.mlp.post"),
    }
    return HookManager(points)


# ---------------------------------------------------------------------------
# HookPoint tests
# ---------------------------------------------------------------------------

class TestHookPoint:
    def test_name(self, simple_hook_point):
        assert simple_hook_point.name == "blocks.0.attn.out"

    def test_no_hooks_initially(self, simple_hook_point):
        assert not simple_hook_point.has_hooks()
        assert simple_hook_point.hook_keys() == []

    def test_add_hook(self, simple_hook_point):
        simple_hook_point.add_hook("test", lambda t, h: t)
        assert simple_hook_point.has_hooks()
        assert "test" in simple_hook_point.hook_keys()

    def test_remove_hook(self, simple_hook_point):
        simple_hook_point.add_hook("a", lambda t, h: t)
        simple_hook_point.add_hook("b", lambda t, h: t)
        simple_hook_point.remove_hook("a")
        assert "a" not in simple_hook_point.hook_keys()
        assert "b" in simple_hook_point.hook_keys()

    def test_remove_all_hooks(self, simple_hook_point):
        simple_hook_point.add_hook("x", lambda t, h: t)
        simple_hook_point.add_hook("y", lambda t, h: t)
        simple_hook_point.remove_hooks()
        assert not simple_hook_point.has_hooks()

    def test_forward_passthrough(self, simple_hook_point):
        """Without hooks, forward is identity."""
        t = torch.randn(2, 10, 64)
        out = simple_hook_point.forward(t)
        assert torch.allclose(out, t)

    def test_forward_caches_output(self, simple_hook_point):
        t = torch.randn(1, 5, 32)
        simple_hook_point.forward(t)
        assert simple_hook_point.output is not None
        assert simple_hook_point.output.shape == t.shape

    def test_hook_modifies_tensor(self, simple_hook_point):
        """Hook that doubles the tensor."""
        simple_hook_point.add_hook("double", lambda t, h: t * 2)
        t = torch.ones(1, 3, 8)
        out = simple_hook_point.forward(t)
        assert torch.allclose(out, t * 2)

    def test_multiple_hooks_apply_in_order(self, simple_hook_point):
        """Two hooks: +1 then *2 → (x+1)*2."""
        simple_hook_point.add_hook("plus1", lambda t, h: t + 1)
        simple_hook_point.add_hook("times2", lambda t, h: t * 2)
        t = torch.zeros(1)
        out = simple_hook_point.forward(t)
        assert out.item() == pytest.approx(2.0)

    def test_clear_cache(self, simple_hook_point):
        t = torch.randn(1, 4)
        simple_hook_point.forward(t)
        simple_hook_point.clear_cache()
        assert simple_hook_point.output is None


# ---------------------------------------------------------------------------
# HookManager tests
# ---------------------------------------------------------------------------

class TestHookManager:
    def test_len(self, two_point_manager):
        assert len(two_point_manager) == 2

    def test_contains(self, two_point_manager):
        assert "blocks.0.attn.out" in two_point_manager
        assert "blocks.99.resid.pre" not in two_point_manager

    def test_getitem(self, two_point_manager):
        hp = two_point_manager["blocks.0.attn.out"]
        assert isinstance(hp, HookPoint)

    def test_getitem_missing_raises(self, two_point_manager):
        with pytest.raises(KeyError):
            _ = two_point_manager["nonexistent"]

    def test_add_and_remove_hooks(self, two_point_manager):
        fn = lambda t, h: t
        two_point_manager.add_hook("blocks.0.attn.out", "cache", fn)
        assert two_point_manager["blocks.0.attn.out"].has_hooks()
        two_point_manager.remove_all_hooks()
        assert not two_point_manager["blocks.0.attn.out"].has_hooks()

    def test_hooks_context_manager(self, two_point_manager):
        """Hooks added in context manager should be removed after exit."""
        fn = lambda t, h: t
        with two_point_manager.hooks({"blocks.0.attn.out": fn}):
            assert two_point_manager["blocks.0.attn.out"].has_hooks()
        assert not two_point_manager["blocks.0.attn.out"].has_hooks()

    def test_run_with_cache(self, two_point_manager):
        """run_with_cache should return an ActivationCache with the right keys."""
        tensor_0 = torch.randn(1, 5, 32)
        tensor_1 = torch.randn(1, 5, 64)

        def _forward():
            two_point_manager["blocks.0.attn.out"].forward(tensor_0)
            two_point_manager["blocks.1.mlp.post"].forward(tensor_1)

        cache = two_point_manager.run_with_cache(_forward)
        assert "blocks.0.attn.out" in cache
        assert "blocks.1.mlp.post" in cache
        assert torch.allclose(cache["blocks.0.attn.out"], tensor_0)


# ---------------------------------------------------------------------------
# Naming tests
# ---------------------------------------------------------------------------

class TestNaming:
    def test_attn_hook(self):
        assert attn_hook(3, "out") == "blocks.3.attn.out"

    def test_mlp_hook(self):
        assert mlp_hook(5, "post") == "blocks.5.mlp.post"

    def test_resid_hook(self):
        assert resid_hook(0, "pre") == "blocks.0.resid.pre"

    def test_parse_block_hook(self):
        parsed = parse_hook_name("blocks.7.attn.q")
        assert parsed.layer == 7
        assert parsed.component == "attn"
        assert parsed.slot == "q"
        assert parsed.is_attn

    def test_parse_embed_hook(self):
        parsed = parse_hook_name("embed.out")
        assert parsed.layer is None
        assert parsed.is_embed

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_hook_name("this.is.not.valid.at.all")

    def test_validate_valid(self):
        assert validate_hook_name("blocks.3.attn.out")
        assert validate_hook_name("blocks.0.mlp.post")
        assert validate_hook_name("embed.out")
        assert validate_hook_name("unembed.pre")

    def test_validate_invalid_slot(self):
        assert not validate_hook_name("blocks.0.attn.xyz")
        assert not validate_hook_name("blocks.0.mlp.wrong")
