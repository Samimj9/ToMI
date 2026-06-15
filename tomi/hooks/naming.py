"""
tomi/hooks/naming.py
--------------------
Universal hook name schema for ToMI.

All hook names follow the pattern::

    blocks.<layer>.<component>.<slot>

where:

* ``<layer>``     — zero-based transformer layer index (int)
* ``<component>`` — ``attn`` | ``mlp`` | ``resid``
* ``<slot>``      — component-specific slot (see below)

Attention slots
~~~~~~~~~~~~~~~
* ``q``   — query projections after per-head split
* ``k``   — key projections after per-head split
* ``v``   — value projections after per-head split
* ``pattern`` — softmax attention pattern (B, H, S, S)
* ``z``   — weighted-sum of values (B, S, H, d_head)
* ``out`` — attention output after output projection (B, S, d_model)

MLP slots
~~~~~~~~~
* ``pre``  — MLP input (before first linear / activation)
* ``gate`` — gate activations (SwiGLU / GeGLU models)
* ``post`` — MLP output (after second linear)

Residual slots
~~~~~~~~~~~~~~
* ``pre``  — residual stream entering the layer
* ``mid``  — residual stream after attention (before MLP)
* ``post`` — residual stream after MLP (exiting the layer)

Embedding slots
~~~~~~~~~~~~~~~
* ``embed.token``   — token embedding (B, S, d_model)
* ``embed.pos``     — positional embedding (if applicable)
* ``embed.out``     — combined embedding output

Unembedding slots
~~~~~~~~~~~~~~~~~
* ``unembed.pre``  — final layer-norm input
* ``unembed.out``  — logits (B, S, V)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Name construction helpers
# ---------------------------------------------------------------------------

def attn_hook(layer: int, slot: str) -> str:
    """Return the attention hook name for *layer* and *slot*.

    Parameters
    ----------
    layer:
        Zero-based layer index.
    slot:
        One of ``q``, ``k``, ``v``, ``pattern``, ``z``, ``out``.

    Returns
    -------
    str
        e.g. ``"blocks.3.attn.out"``
    """
    return f"blocks.{layer}.attn.{slot}"


def mlp_hook(layer: int, slot: str) -> str:
    """Return the MLP hook name for *layer* and *slot*.

    Parameters
    ----------
    layer:
        Zero-based layer index.
    slot:
        One of ``pre``, ``gate``, ``post``.

    Returns
    -------
    str
        e.g. ``"blocks.5.mlp.post"``
    """
    return f"blocks.{layer}.mlp.{slot}"


def resid_hook(layer: int, slot: str) -> str:
    """Return the residual stream hook name for *layer* and *slot*.

    Parameters
    ----------
    layer:
        Zero-based layer index.
    slot:
        One of ``pre``, ``mid``, ``post``.

    Returns
    -------
    str
        e.g. ``"blocks.0.resid.pre"``
    """
    return f"blocks.{layer}.resid.{slot}"


def embed_hook(slot: str) -> str:
    """Return an embedding hook name.

    Parameters
    ----------
    slot:
        One of ``token``, ``pos``, ``out``.

    Returns
    -------
    str
        e.g. ``"embed.out"``
    """
    return f"embed.{slot}"


def unembed_hook(slot: str) -> str:
    """Return an unembedding hook name.

    Parameters
    ----------
    slot:
        One of ``pre``, ``out``.

    Returns
    -------
    str
        e.g. ``"unembed.out"``
    """
    return f"unembed.{slot}"


# ---------------------------------------------------------------------------
# Name parsing helpers
# ---------------------------------------------------------------------------

@dataclass
class ParsedHookName:
    """Structured representation of a parsed hook name."""

    raw: str
    layer: Optional[int]
    component: str
    slot: str

    @property
    def is_attn(self) -> bool:
        return self.component == "attn"

    @property
    def is_mlp(self) -> bool:
        return self.component == "mlp"

    @property
    def is_resid(self) -> bool:
        return self.component == "resid"

    @property
    def is_embed(self) -> bool:
        return self.component == "embed"

    @property
    def is_unembed(self) -> bool:
        return self.component == "unembed"


def parse_hook_name(name: str) -> ParsedHookName:
    """Parse a ToMI hook name into its constituent parts.

    Parameters
    ----------
    name:
        A hook name string, e.g. ``"blocks.3.attn.out"``.

    Returns
    -------
    ParsedHookName

    Raises
    ------
    ValueError
        If *name* does not match the expected schema.
    """
    parts = name.split(".")

    if parts[0] == "blocks" and len(parts) == 4:
        try:
            layer = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid layer index in hook name: '{name}'")
        return ParsedHookName(raw=name, layer=layer, component=parts[2], slot=parts[3])

    if parts[0] in {"embed", "unembed"} and len(parts) == 2:
        return ParsedHookName(raw=name, layer=None, component=parts[0], slot=parts[1])

    raise ValueError(
        f"Cannot parse hook name '{name}'. "
        "Expected 'blocks.<layer>.<component>.<slot>' or '<component>.<slot>'."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_ATTN_SLOTS = {"q", "k", "v", "pattern", "z", "out"}
_VALID_MLP_SLOTS = {"pre", "gate", "post"}
_VALID_RESID_SLOTS = {"pre", "mid", "post"}
_VALID_EMBED_SLOTS = {"token", "pos", "out"}
_VALID_UNEMBED_SLOTS = {"pre", "out"}


def validate_hook_name(name: str) -> bool:
    """Return ``True`` if *name* is a valid ToMI hook name.

    Parameters
    ----------
    name:
        Hook name to validate.

    Returns
    -------
    bool
    """
    try:
        parsed = parse_hook_name(name)
    except ValueError:
        return False

    slot_map = {
        "attn": _VALID_ATTN_SLOTS,
        "mlp": _VALID_MLP_SLOTS,
        "resid": _VALID_RESID_SLOTS,
        "embed": _VALID_EMBED_SLOTS,
        "unembed": _VALID_UNEMBED_SLOTS,
    }
    valid_slots = slot_map.get(parsed.component)
    if valid_slots is None:
        return False
    return parsed.slot in valid_slots
