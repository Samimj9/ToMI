"""
tomi/models/qwen.py
--------------------
``QwenAdapter`` — wraps Qwen-family models (Qwen2, Qwen2.5, QwenMoE, …)
to expose the universal ToMI hook interface.

Qwen uses the same architecture block ordering as LLaMA-2:

    embed → [RMSNorm → SelfAttention → residual
              RMSNorm → MLP           → residual] × n_layers → RMSNorm → lm_head

The adapter registers hook points for:

* ``embed.out``                 — token embeddings
* ``blocks.L.resid.pre``        — residual entering layer L
* ``blocks.L.attn.q/k/v``       — Q/K/V projections (post-split)
* ``blocks.L.attn.out``         — attention output (after o_proj)
* ``blocks.L.resid.mid``        — residual after attention
* ``blocks.L.mlp.pre``          — input to MLP (gate/up proj input)
* ``blocks.L.mlp.post``         — output of MLP (down_proj output)
* ``blocks.L.resid.post``       — residual exiting layer L
* ``unembed.pre``               — final layer-norm output
* ``unembed.out``               — logits
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tomi.hooks.hook_manager import HookManager
from tomi.hooks.hook_point import HookPoint
from tomi.hooks.naming import attn_hook, embed_hook, mlp_hook, resid_hook, unembed_hook
from tomi.models.base_model import ToMModel
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class QwenAdapter(ToMModel):
    """ToMI wrapper for Qwen-family models.

    Parameters
    ----------
    model:
        A loaded Qwen ``PreTrainedModel`` (e.g. ``Qwen2ForCausalLM``).
    tokenizer:
        Corresponding tokenizer.
    device:
        Device the model resides on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        # super().__init__ calls _build_hook_manager internally
        super().__init__(model, tokenizer, device)

    # ------------------------------------------------------------------
    # Architecture introspection
    # ------------------------------------------------------------------

    def _get_layers(self) -> nn.ModuleList:
        """Return the list of transformer layer modules."""
        # Qwen2 stores layers at model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(
            "Could not find transformer layers in Qwen model. "
            "Expected `model.model.layers`."
        )

    def _get_embed(self) -> nn.Embedding:
        """Return the token embedding module."""
        return self.model.model.embed_tokens

    def _get_lm_head(self) -> nn.Linear:
        """Return the LM-head (unembedding) linear layer."""
        return self.model.lm_head

    def _get_final_norm(self) -> nn.Module:
        """Return the final layer norm."""
        return self.model.model.norm

    # ------------------------------------------------------------------
    # ToMModel properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self._get_layers())

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_heads(self) -> int:
        return self.model.config.num_attention_heads

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    # ------------------------------------------------------------------
    # Hook-point construction
    # ------------------------------------------------------------------

    def _build_hook_manager(self) -> HookManager:
        """Build hook points for the Qwen architecture."""
        hook_points: Dict[str, HookPoint] = {}
        layers = self._get_layers()
        n = len(layers)

        # ---- embedding ----
        embed_pt = HookPoint(embed_hook("out"))
        hook_points[embed_pt.name] = embed_pt
        self._get_embed().register_forward_hook(
            _make_output_hook(embed_pt)
        )

        # ---- per-layer hooks ----
        for layer_idx in range(n):
            layer = layers[layer_idx]
            self._register_layer_hooks(layer, layer_idx, hook_points)

        # ---- unembedding ----
        norm_pt = HookPoint(unembed_hook("pre"))
        hook_points[norm_pt.name] = norm_pt
        self._get_final_norm().register_forward_hook(
            _make_output_hook(norm_pt)
        )

        unembed_pt = HookPoint(unembed_hook("out"))
        hook_points[unembed_pt.name] = unembed_pt
        self._get_lm_head().register_forward_hook(
            _make_output_hook(unembed_pt)
        )

        return HookManager(hook_points)

    def _register_layer_hooks(
        self,
        layer: nn.Module,
        layer_idx: int,
        hook_points: Dict[str, HookPoint],
    ) -> None:
        """Register all hook points for a single transformer layer."""

        # ---- residual pre (input to the layer) ----
        resid_pre = HookPoint(resid_hook(layer_idx, "pre"))
        hook_points[resid_pre.name] = resid_pre
        layer.register_forward_pre_hook(
            _make_first_input_hook(resid_pre)
        )

        # ---- attention Q/K/V ----
        attn = layer.self_attn
        q_pt = HookPoint(attn_hook(layer_idx, "q"))
        k_pt = HookPoint(attn_hook(layer_idx, "k"))
        v_pt = HookPoint(attn_hook(layer_idx, "v"))
        hook_points[q_pt.name] = q_pt
        hook_points[k_pt.name] = k_pt
        hook_points[v_pt.name] = v_pt

        # Q proj
        attn.q_proj.register_forward_hook(_make_output_hook(q_pt))
        # K proj
        attn.k_proj.register_forward_hook(_make_output_hook(k_pt))
        # V proj
        attn.v_proj.register_forward_hook(_make_output_hook(v_pt))

        # ---- attention output (after o_proj) ----
        attn_out = HookPoint(attn_hook(layer_idx, "out"))
        hook_points[attn_out.name] = attn_out
        attn.o_proj.register_forward_hook(_make_output_hook(attn_out))

        # ---- residual mid (after attention, before MLP) ----
        resid_mid = HookPoint(resid_hook(layer_idx, "mid"))
        hook_points[resid_mid.name] = resid_mid
        # We hook the MLP's input norm to approximate residual mid
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm.register_forward_hook(
                _make_input_hook(resid_mid)
            )

        # ---- MLP pre / post ----
        mlp = layer.mlp
        mlp_pre = HookPoint(mlp_hook(layer_idx, "pre"))
        hook_points[mlp_pre.name] = mlp_pre

        # Gate proj input = MLP input
        if hasattr(mlp, "gate_proj"):
            mlp.gate_proj.register_forward_pre_hook(
                _make_first_input_hook(mlp_pre)
            )
        elif hasattr(mlp, "up_proj"):
            mlp.up_proj.register_forward_pre_hook(
                _make_first_input_hook(mlp_pre)
            )

        mlp_post = HookPoint(mlp_hook(layer_idx, "post"))
        hook_points[mlp_post.name] = mlp_post
        mlp.down_proj.register_forward_hook(_make_output_hook(mlp_post))

        # ---- residual post (exit of the layer) ----
        resid_post = HookPoint(resid_hook(layer_idx, "post"))
        hook_points[resid_post.name] = resid_post
        layer.register_forward_hook(_make_output_hook(resid_post))


# ---------------------------------------------------------------------------
# Internal hook-factory helpers
# ---------------------------------------------------------------------------

def _make_output_hook(point: HookPoint):
    """Create a PyTorch forward hook that routes the module *output* through *point*."""
    def _hook(module: nn.Module, inp, output):
        if isinstance(output, torch.Tensor):
            return point.forward(output)
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            patched = point.forward(output[0])
            return (patched,) + output[1:]
        return output
    return _hook


def _make_input_hook(point: HookPoint):
    """Create a hook that captures the module *output* but the *input* to a later op."""
    # We abuse output hook to get "what is about to be processed"
    def _hook(module: nn.Module, inp, output):
        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            point.forward(inp[0])
        return output
    return _hook


def _make_first_input_hook(point: HookPoint):
    """Create a PyTorch forward *pre*-hook that routes the first input through *point*."""
    def _pre_hook(module: nn.Module, inp):
        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            patched = point.forward(inp[0])
            return (patched,) + inp[1:]
        return inp
    return _pre_hook
