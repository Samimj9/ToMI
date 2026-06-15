"""
tomi/circuits/node.py
----------------------
``CircuitNode`` — a node in a mechanistic interpretability circuit graph.

A node represents a specific computational unit:

* A residual stream at a given layer and position.
* An attention head ``(layer, head)``.
* An MLP neuron ``(layer, neuron_index)``.
* An embedding or unembedding operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """Type of computational unit represented by a circuit node."""

    EMBED = "embed"
    ATTENTION_HEAD = "attn_head"
    MLP_NEURON = "mlp_neuron"
    MLP_LAYER = "mlp_layer"
    RESIDUAL = "residual"
    UNEMBED = "unembed"


@dataclass
class CircuitNode:
    """A node in a ToMI circuit graph.

    Parameters
    ----------
    node_type:
        The type of computational unit.
    layer:
        Transformer layer index (``None`` for embed/unembed nodes).
    head:
        Attention head index (only for ``ATTENTION_HEAD`` nodes).
    neuron:
        MLP neuron index (only for ``MLP_NEURON`` nodes).
    position:
        Sequence token position this node corresponds to.
    score:
        Optional importance / attribution score.
    label:
        Optional human-readable label.
    """

    node_type: NodeType
    layer: Optional[int] = None
    head: Optional[int] = None
    neuron: Optional[int] = None
    position: Optional[int] = None
    score: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self._auto_label()

    def _auto_label(self) -> str:
        """Generate a compact label from the node's attributes."""
        if self.node_type == NodeType.EMBED:
            return f"embed[{self.position}]"
        if self.node_type == NodeType.UNEMBED:
            return f"unembed[{self.position}]"
        if self.node_type == NodeType.ATTENTION_HEAD:
            return f"L{self.layer}H{self.head}[{self.position}]"
        if self.node_type == NodeType.MLP_NEURON:
            return f"L{self.layer}N{self.neuron}[{self.position}]"
        if self.node_type == NodeType.MLP_LAYER:
            return f"L{self.layer}.mlp[{self.position}]"
        if self.node_type == NodeType.RESIDUAL:
            return f"L{self.layer}.resid[{self.position}]"
        return "node"

    @property
    def node_id(self) -> str:
        """Unique string identifier for use in graph libraries."""
        parts = [self.node_type.value]
        for attr in ("layer", "head", "neuron", "position"):
            val = getattr(self, attr)
            if val is not None:
                parts.append(str(val))
        return "_".join(parts)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"CircuitNode({self.label}, score={self.score:.4f})"
