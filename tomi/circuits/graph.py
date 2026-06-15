"""
tomi/circuits/graph.py
-----------------------
``CircuitGraph`` — a directed graph of :class:`~tomi.circuits.node.CircuitNode`
objects connected by :class:`~tomi.circuits.edge.CircuitEdge` objects.

Provides both a pure-Python representation and an optional NetworkX export.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Set

from tomi.circuits.edge import CircuitEdge
from tomi.circuits.node import CircuitNode, NodeType
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class CircuitGraph:
    """Directed graph of circuit nodes and edges.

    Parameters
    ----------
    name:
        Optional human-readable name for this circuit.
    """

    def __init__(self, name: str = "circuit") -> None:
        self.name = name
        self._nodes: Dict[str, CircuitNode] = {}
        self._edges: Dict[str, CircuitEdge] = {}
        self._adj: Dict[str, List[str]] = {}  # node_id → [dst edge_id, …]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: CircuitNode) -> None:
        """Add *node* to the graph (idempotent).

        Parameters
        ----------
        node:
            The node to add.
        """
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._adj[node.node_id] = []

    def add_edge(self, edge: CircuitEdge) -> None:
        """Add *edge* to the graph, auto-adding its endpoints if needed.

        Parameters
        ----------
        edge:
            The directed edge to add.
        """
        self.add_node(edge.src)
        self.add_node(edge.dst)
        if edge.edge_id not in self._edges:
            self._edges[edge.edge_id] = edge
            self._adj[edge.src.node_id].append(edge.edge_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges.

        Parameters
        ----------
        node_id:
            The identifier of the node to remove.
        """
        if node_id not in self._nodes:
            return
        # Remove incident edges
        incident = [
            eid for eid, e in self._edges.items()
            if e.src.node_id == node_id or e.dst.node_id == node_id
        ]
        for eid in incident:
            del self._edges[eid]
        del self._nodes[node_id]
        del self._adj[node_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[CircuitNode]:
        """All nodes, sorted by node_id."""
        return [self._nodes[k] for k in sorted(self._nodes)]

    @property
    def edges(self) -> List[CircuitEdge]:
        """All edges, sorted by edge_id."""
        return [self._edges[k] for k in sorted(self._edges)]

    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        """Return a node by id, or ``None``."""
        return self._nodes.get(node_id)

    def successors(self, node: CircuitNode) -> List[CircuitNode]:
        """Return the direct successors of *node*.

        Parameters
        ----------
        node:
            Source node.

        Returns
        -------
        List[CircuitNode]
        """
        edge_ids = self._adj.get(node.node_id, [])
        return [self._edges[eid].dst for eid in edge_ids]

    def predecessors(self, node: CircuitNode) -> List[CircuitNode]:
        """Return the direct predecessors of *node*.

        Parameters
        ----------
        node:
            Target node.

        Returns
        -------
        List[CircuitNode]
        """
        return [
            e.src for e in self._edges.values()
            if e.dst.node_id == node.node_id
        ]

    def top_nodes(self, k: int = 10) -> List[CircuitNode]:
        """Return the *k* highest-scoring nodes.

        Parameters
        ----------
        k:
            Number of nodes to return.

        Returns
        -------
        List[CircuitNode]
        """
        return sorted(self._nodes.values(), key=lambda n: abs(n.score), reverse=True)[:k]

    def filter_by_type(self, node_type: NodeType) -> List[CircuitNode]:
        """Return nodes of a specific type.

        Parameters
        ----------
        node_type:
            The type to filter on.

        Returns
        -------
        List[CircuitNode]
        """
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def prune(self, min_score: float = 0.0) -> "CircuitGraph":
        """Return a subgraph containing only nodes with ``|score| >= min_score``.

        Parameters
        ----------
        min_score:
            Minimum absolute score for inclusion.

        Returns
        -------
        CircuitGraph
        """
        pruned = CircuitGraph(name=f"{self.name}_pruned")
        keep_ids: Set[str] = {
            nid for nid, n in self._nodes.items()
            if abs(n.score) >= min_score
        }
        # First add all qualifying nodes
        for nid in keep_ids:
            pruned.add_node(self._nodes[nid])
        # Then add edges where both endpoints qualify
        for eid, edge in self._edges.items():
            if edge.src.node_id in keep_ids and edge.dst.node_id in keep_ids:
                pruned.add_edge(edge)
        return pruned

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_networkx(self):
        """Export to a ``networkx.DiGraph``.

        Returns
        -------
        networkx.DiGraph

        Raises
        ------
        ImportError
            If NetworkX is not installed.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("Install networkx: pip install networkx") from e

        g = nx.DiGraph(name=self.name)
        for node in self.nodes:
            g.add_node(node.node_id, label=node.label, score=node.score, type=node.node_type.value)
        for edge in self.edges:
            g.add_edge(edge.src.node_id, edge.dst.node_id, weight=edge.weight, label=edge.label)
        return g

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"CircuitGraph(name='{self.name}', "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)})"
        )
