"""
tomi/circuits/edge.py
----------------------
``CircuitEdge`` — a directed edge in a mechanistic interpretability circuit.

An edge represents a causal information-flow path between two
:class:`~tomi.circuits.node.CircuitNode` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from tomi.circuits.node import CircuitNode


@dataclass
class CircuitEdge:
    """A directed, weighted edge in a circuit graph.

    Parameters
    ----------
    src:
        Source (upstream) node.
    dst:
        Destination (downstream) node.
    weight:
        Edge weight, typically the causal effect or attribution score.
    label:
        Optional human-readable label for the edge.
    """

    src: CircuitNode
    dst: CircuitNode
    weight: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.src.label} → {self.dst.label}"

    @property
    def edge_id(self) -> str:
        """Unique string identifier."""
        return f"{self.src.node_id}__to__{self.dst.node_id}"

    def __hash__(self) -> int:
        return hash(self.edge_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitEdge):
            return NotImplemented
        return self.edge_id == other.edge_id

    def __repr__(self) -> str:
        return f"CircuitEdge({self.label}, w={self.weight:.4f})"
