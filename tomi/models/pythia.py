"""
tomi/models/pythia.py
----------------------
``PythiaAdapter`` — wraps EleutherAI Pythia models (GPT-NeoX family) to
expose the universal ToMI hook interface.

Pythia architecture outline::

    embed_in → [ln1 → Attention → residual
                ln2 → MLP       → residual] × n_layers → final_ln → embed_out

Hook points registered:

* ``embed.out``                 — token + positional embeddings
* ``blocks.L.resid.pre``        — residual entering layer L
* ``blocks.L.attn.q/k/v``       — Q/K/V projections
* ``blocks.L.attn.out``         — attention output (dense/o_proj)
* ``blocks.L.resid.mid``        — residual after attention
* ``blocks.L.mlp.pre``          — MLP input
* ``blocks.L.mlp.post``         — MLP output
* ``blocks.L.resid.post``       — residual exiting layer L
* ``unembed.pre``               — final layer-norm output
* ``unembed.out``               — logits
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tomi.hooks.hook_manager import HookManager
from tomi.hooks.hook_point import HookPoint
from tomi.hooks.naming import attn_hook, embed_hook, mlp_hook, resid_hook, unembed_hook
from tomi.models.base_model import ToMModel
from tomi.models.qwen import (
    _make_first_input_hook,
    _make_input_hook,
    _make_output_hook,
)
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class PythiaAdapter(ToMModel):
    """ToMI wrapper for Pythia (GPT-NeoX) models.

    Parameters
    ----------
    model:
        A loaded ``GPTNeoXForCausalLM``.
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
        super().__init__(model, tokenizer, device)

    # ------------------------------------------------------------------
    # Architecture introspection
    # ------------------------------------------------------------------

    def _get_layers(self) -> nn.ModuleList:
        # GPT-NeoX: model.gpt_neox.layers
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        # Fallback for some HF model classes
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError(
            "Could not find transformer layers in Pythia model. "
            "Expected `model.gpt_neox.layers`."
        )

    def _get_embed(self) -> nn.Embedding:
        return self.model.gpt_neox.embed_in

    def _get_final_norm(self) -> nn.Module:
        return self.model.gpt_neox.final_layer_norm

    def _get_lm_head(self) -> nn.Linear:
        return self.model.embed_out

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
        """Build hook points for the Pythia (GPT-NeoX) architecture."""
        hook_points: Dict[str, HookPoint] = {}
        layers = self._get_layers()
        n = len(layers)

        # ---- embedding ----
        embed_pt = HookPoint(embed_hook("out"))
        hook_points[embed_pt.name] = embed_pt
        self._get_embed().register_forward_hook(_make_output_hook(embed_pt))

        # ---- per-layer ----
        for layer_idx in range(n):
            layer = layers[layer_idx]
            self._register_layer_hooks(layer, layer_idx, hook_points)

        # ---- unembedding ----
        norm_pt = HookPoint(unembed_hook("pre"))
        hook_points[norm_pt.name] = norm_pt
        self._get_final_norm().register_forward_hook(_make_output_hook(norm_pt))

        unembed_pt = HookPoint(unembed_hook("out"))
        hook_points[unembed_pt.name] = unembed_pt
        self._get_lm_head().register_forward_hook(_make_output_hook(unembed_pt))

        return HookManager(hook_points)

    def _register_layer_hooks(
        self,
        layer: nn.Module,
        layer_idx: int,
        hook_points: Dict[str, HookPoint],
    ) -> None:
        """Register all hook points for a single GPT-NeoX layer."""

        # resid pre
        resid_pre = HookPoint(resid_hook(layer_idx, "pre"))
        hook_points[resid_pre.name] = resid_pre
        layer.register_forward_pre_hook(_make_first_input_hook(resid_pre))

        # attention module
        attn = layer.attention
        q_pt = HookPoint(attn_hook(layer_idx, "q"))
        k_pt = HookPoint(attn_hook(layer_idx, "k"))
        v_pt = HookPoint(attn_hook(layer_idx, "v"))
        hook_points[q_pt.name] = q_pt
        hook_points[k_pt.name] = k_pt
        hook_points[v_pt.name] = v_pt

        # GPT-NeoX fuses QKV into a single projection; intercept after split
        # We hook the dense (o_proj equivalent) output for attn.out
        if hasattr(attn, "query_key_value"):
            # Register a combined hook; splitting is done inside the module
            # so we approximate q/k/v from the combined projection
            attn.query_key_value.register_forward_hook(
                _make_qkv_split_hook(q_pt, k_pt, v_pt, self.n_heads, self.d_head)
            )
        else:
            if hasattr(attn, "q_proj"):
                attn.q_proj.register_forward_hook(_make_output_hook(q_pt))
            if hasattr(attn, "k_proj"):
                attn.k_proj.register_forward_hook(_make_output_hook(k_pt))
            if hasattr(attn, "v_proj"):
                attn.v_proj.register_forward_hook(_make_output_hook(v_pt))

        attn_out_pt = HookPoint(attn_hook(layer_idx, "out"))
        hook_points[attn_out_pt.name] = attn_out_pt
        if hasattr(attn, "dense"):
            attn.dense.register_forward_hook(_make_output_hook(attn_out_pt))
        elif hasattr(attn, "out_proj"):
            attn.out_proj.register_forward_hook(_make_output_hook(attn_out_pt))

        # resid mid — before MLP
        resid_mid = HookPoint(resid_hook(layer_idx, "mid"))
        hook_points[resid_mid.name] = resid_mid
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm.register_forward_hook(
                _make_input_hook(resid_mid)
            )
        elif hasattr(layer, "ln_2"):
            layer.ln_2.register_forward_hook(_make_input_hook(resid_mid))

        # MLP pre / post
        mlp = layer.mlp
        mlp_pre_pt = HookPoint(mlp_hook(layer_idx, "pre"))
        hook_points[mlp_pre_pt.name] = mlp_pre_pt
        mlp.register_forward_pre_hook(_make_first_input_hook(mlp_pre_pt))

        mlp_post_pt = HookPoint(mlp_hook(layer_idx, "post"))
        hook_points[mlp_post_pt.name] = mlp_post_pt
        mlp.register_forward_hook(_make_output_hook(mlp_post_pt))

        # resid post — exit of layer
        resid_post = HookPoint(resid_hook(layer_idx, "post"))
        hook_points[resid_post.name] = resid_post
        layer.register_forward_hook(_make_output_hook(resid_post))


# ---------------------------------------------------------------------------
# Pythia-specific helper: QKV split
# ---------------------------------------------------------------------------

def _make_qkv_split_hook(
    q_pt: HookPoint,
    k_pt: HookPoint,
    v_pt: HookPoint,
    n_heads: int,
    d_head: int,
):
    """Forward hook for fused QKV projection (GPT-NeoX style)."""
    def _hook(module: nn.Module, inp, output):
        if not isinstance(output, torch.Tensor):
            return output
        # output shape: (batch, seq, 3 * n_heads * d_head)
        batch, seq, _ = output.shape
        qkv = output.reshape(batch, seq, 3, n_heads, d_head)
        q_pt.forward(qkv[:, :, 0].reshape(batch, seq, -1))
        k_pt.forward(qkv[:, :, 1].reshape(batch, seq, -1))
        v_pt.forward(qkv[:, :, 2].reshape(batch, seq, -1))
        return output  # do not modify
    return _hook
