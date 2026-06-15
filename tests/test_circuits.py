"""
tests/test_circuits.py
-----------------------
Unit tests for circuit graph data structures.
"""

from __future__ import annotations

import pytest

from tomi.circuits.node import CircuitNode, NodeType
from tomi.circuits.edge import CircuitEdge
from tomi.circuits.graph import CircuitGraph


@pytest.fixture
def attn_node() -> CircuitNode:
    return CircuitNode(node_type=NodeType.ATTENTION_HEAD, layer=3, head=5, score=0.8)


@pytest.fixture
def mlp_node() -> CircuitNode:
    return CircuitNode(node_type=NodeType.MLP_LAYER, layer=4, score=0.6)


@pytest.fixture
def simple_graph(attn_node, mlp_node) -> CircuitGraph:
    g = CircuitGraph(name="test_circuit")
    edge = CircuitEdge(src=attn_node, dst=mlp_node, weight=0.7)
    g.add_edge(edge)
    return g


class TestCircuitNode:
    def test_label_auto_generated(self, attn_node):
        assert "L3" in attn_node.label
        assert "H5" in attn_node.label

    def test_node_id_unique(self, attn_node, mlp_node):
        assert attn_node.node_id != mlp_node.node_id

    def test_equality(self):
        a = CircuitNode(node_type=NodeType.RESIDUAL, layer=2)
        b = CircuitNode(node_type=NodeType.RESIDUAL, layer=2)
        assert a == b

    def test_hash_consistent_with_eq(self):
        a = CircuitNode(node_type=NodeType.RESIDUAL, layer=2)
        b = CircuitNode(node_type=NodeType.RESIDUAL, layer=2)
        assert hash(a) == hash(b)

    def test_embed_node_label(self):
        n = CircuitNode(node_type=NodeType.EMBED, position=3)
        assert "embed" in n.label

    def test_unembed_node(self):
        n = CircuitNode(node_type=NodeType.UNEMBED)
        assert "unembed" in n.label


class TestCircuitEdge:
    def test_label_auto(self, attn_node, mlp_node):
        e = CircuitEdge(src=attn_node, dst=mlp_node, weight=0.5)
        assert attn_node.label in e.label
        assert mlp_node.label in e.label

    def test_edge_id(self, attn_node, mlp_node):
        e = CircuitEdge(src=attn_node, dst=mlp_node)
        assert "to" in e.edge_id


class TestCircuitGraph:
    def test_add_node(self, attn_node):
        g = CircuitGraph()
        g.add_node(attn_node)
        assert len(g) == 1
        assert attn_node in g.nodes

    def test_add_edge_adds_nodes(self, attn_node, mlp_node):
        g = CircuitGraph()
        edge = CircuitEdge(src=attn_node, dst=mlp_node)
        g.add_edge(edge)
        assert len(g) == 2
        assert len(g.edges) == 1

    def test_successors(self, simple_graph, attn_node, mlp_node):
        succs = simple_graph.successors(attn_node)
        assert mlp_node in succs

    def test_predecessors(self, simple_graph, attn_node, mlp_node):
        preds = simple_graph.predecessors(mlp_node)
        assert attn_node in preds

    def test_top_nodes(self, simple_graph):
        top = simple_graph.top_nodes(k=1)
        assert len(top) == 1
        # attn_node has higher score (0.8 > 0.6)
        assert top[0].score == pytest.approx(0.8)

    def test_filter_by_type(self, simple_graph):
        attn_nodes = simple_graph.filter_by_type(NodeType.ATTENTION_HEAD)
        assert len(attn_nodes) == 1
        mlp_nodes = simple_graph.filter_by_type(NodeType.MLP_LAYER)
        assert len(mlp_nodes) == 1

    def test_prune_removes_low_score(self, simple_graph, attn_node, mlp_node):
        pruned = simple_graph.prune(min_score=0.7)
        # mlp_node has score 0.6 < 0.7; should be pruned
        assert len(pruned) == 1
        assert attn_node in pruned.nodes
        assert mlp_node not in pruned.nodes

    def test_idempotent_add(self, attn_node):
        g = CircuitGraph()
        g.add_node(attn_node)
        g.add_node(attn_node)  # second add should be no-op
        assert len(g) == 1

    def test_remove_node(self, simple_graph, attn_node):
        simple_graph.remove_node(attn_node.node_id)
        assert attn_node not in simple_graph.nodes
        # Edge should also be removed
        assert len(simple_graph.edges) == 0

    def test_repr(self, simple_graph):
        r = repr(simple_graph)
        assert "CircuitGraph" in r
        assert "test_circuit" in r
