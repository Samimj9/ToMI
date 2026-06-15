"""
tomi/circuits/circuit_finder.py
---------------------------------
``CircuitFinder`` — automated circuit discovery via activation patching.

Builds a :class:`~tomi.circuits.graph.CircuitGraph` from patching results
by treating high-importance hooks as nodes and inferring edges from the
layer ordering.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from tomi.circuits.edge import CircuitEdge
from tomi.circuits.graph import CircuitGraph
from tomi.circuits.node import CircuitNode, NodeType
from tomi.models.base_model import ToMModel
from tomi.patching.activation_patching import PatchingResult, activation_patching
from tomi.patching.head_patching import HeadPatchingResult, head_patching
from tomi.utils.logging import get_logger

log = get_logger(__name__)


class CircuitFinder:
    """Discovers circuits from activation patching experiments.

    Parameters
    ----------
    model:
        Wrapped model.
    importance_threshold:
        Minimum normalised causal effect for a component to be included
        in the circuit.
    """

    def __init__(
        self,
        model: ToMModel,
        importance_threshold: float = 0.1,
    ) -> None:
        self.model = model
        self.importance_threshold = importance_threshold

    def find_circuit(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        correct_token_id: int,
        incorrect_token_id: int,
        position: int = -1,
        include_heads: bool = True,
        circuit_name: str = "circuit",
    ) -> CircuitGraph:
        """Run patching experiments and build a circuit graph.

        Parameters
        ----------
        clean_tokens:
            Clean prompt tokens.
        corrupted_tokens:
            Corrupted prompt tokens.
        correct_token_id:
            Correct answer token id.
        incorrect_token_id:
            Foil token id.
        position:
            Metric evaluation position.
        include_heads:
            Whether to run per-head patching and add head nodes.
        circuit_name:
            Name for the resulting :class:`CircuitGraph`.

        Returns
        -------
        CircuitGraph
        """
        log.info("Running activation patching for circuit discovery …")
        patching_result = activation_patching(
            model=self.model,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            correct_token_id=correct_token_id,
            incorrect_token_id=incorrect_token_id,
            position=position,
            show_progress=True,
        )

        graph = CircuitGraph(name=circuit_name)
        self._add_nodes_from_patching(graph, patching_result)

        if include_heads:
            log.info("Running head patching for circuit discovery …")
            head_result = head_patching(
                model=self.model,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                position=position,
                show_progress=True,
            )
            self._add_head_nodes(graph, head_result)

        self._add_layer_order_edges(graph)

        log.info(
            "Circuit found: %d nodes, %d edges",
            len(graph.nodes),
            len(graph.edges),
        )
        return graph

    # ------------------------------------------------------------------
    # Internal graph-building helpers
    # ------------------------------------------------------------------

    def _add_nodes_from_patching(
        self,
        graph: CircuitGraph,
        result: PatchingResult,
    ) -> None:
        """Add nodes from full activation-patching results."""
        score_map = dict(zip(result.hook_names, result.importance_matrix))

        for name, score in score_map.items():
            if abs(score) < self.importance_threshold:
                continue
            node = _hook_name_to_node(name, score=score)
            if node is not None:
                graph.add_node(node)

    def _add_head_nodes(
        self,
        graph: CircuitGraph,
        result: HeadPatchingResult,
    ) -> None:
        """Add per-head nodes from head-patching results."""
        imp = result.importance_matrix  # (n_layers, n_heads)
        for layer in range(result.n_layers):
            for head in range(result.n_heads):
                score = float(imp[layer, head].item())
                if abs(score) < self.importance_threshold:
                    continue
                node = CircuitNode(
                    node_type=NodeType.ATTENTION_HEAD,
                    layer=layer,
                    head=head,
                    score=score,
                )
                graph.add_node(node)

    def _add_layer_order_edges(self, graph: CircuitGraph) -> None:
        """Connect nodes across consecutive layers with directed edges."""
        nodes_by_layer: dict[int, list[CircuitNode]] = {}
        for node in graph.nodes:
            if node.layer is not None:
                nodes_by_layer.setdefault(node.layer, []).append(node)

        layers = sorted(nodes_by_layer.keys())
        for i in range(len(layers) - 1):
            src_layer = layers[i]
            dst_layer = layers[i + 1]
            for src in nodes_by_layer[src_layer]:
                for dst in nodes_by_layer[dst_layer]:
                    # Only connect if both scores are positive (causal direction)
                    if src.score > 0 and dst.score > 0:
                        edge = CircuitEdge(
                            src=src,
                            dst=dst,
                            weight=min(src.score, dst.score),
                        )
                        graph.add_edge(edge)


# ---------------------------------------------------------------------------
# Helper: convert hook name → CircuitNode
# ---------------------------------------------------------------------------

def _hook_name_to_node(
    hook_name: str,
    score: float = 0.0,
    position: Optional[int] = None,
) -> Optional[CircuitNode]:
    """Parse a hook name into a :class:`CircuitNode`, or return ``None``.

    Parameters
    ----------
    hook_name:
        A ToMI hook name like ``"blocks.3.attn.out"``.
    score:
        Importance score to assign.
    position:
        Optional sequence position.

    Returns
    -------
    Optional[CircuitNode]
    """
    parts = hook_name.split(".")
    if parts[0] == "embed":
        return CircuitNode(
            node_type=NodeType.EMBED,
            position=position,
            score=score,
        )
    if parts[0] == "unembed":
        return CircuitNode(
            node_type=NodeType.UNEMBED,
            position=position,
            score=score,
        )
    if parts[0] == "blocks" and len(parts) >= 3:
        try:
            layer = int(parts[1])
        except ValueError:
            return None
        component = parts[2]
        slot = parts[3] if len(parts) > 3 else ""
        if component == "attn":
            return CircuitNode(
                node_type=NodeType.ATTENTION_HEAD,
                layer=layer,
                position=position,
                score=score,
            )
        if component == "mlp":
            return CircuitNode(
                node_type=NodeType.MLP_LAYER,
                layer=layer,
                position=position,
                score=score,
            )
        if component == "resid":
            return CircuitNode(
                node_type=NodeType.RESIDUAL,
                layer=layer,
                position=position,
                score=score,
            )
    return None
