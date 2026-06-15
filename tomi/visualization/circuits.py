"""
tomi/visualization/circuits.py
--------------------------------
Circuit graph visualization using Plotly (interactive) and Matplotlib (static).
"""

from __future__ import annotations

from typing import Optional, Tuple

from tomi.circuits.graph import CircuitGraph
from tomi.utils.logging import get_logger

log = get_logger(__name__)


def plot_circuit_graph(
    graph: CircuitGraph,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    node_size_scale: float = 500.0,
    save_path: Optional[str] = None,
):
    """Plot a :class:`~tomi.circuits.graph.CircuitGraph` using Matplotlib + NetworkX.

    Parameters
    ----------
    graph:
        The circuit to visualise.
    title:
        Plot title (defaults to ``graph.name``).
    figsize:
        Figure size.
    node_size_scale:
        Scale factor for node sizes (proportional to ``|score|``).
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "Install matplotlib and networkx: pip install matplotlib networkx"
        ) from e

    g = graph.to_networkx()
    if len(g) == 0:
        log.warning("Empty circuit graph; nothing to plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty circuit graph", ha="center", va="center")
        return fig

    # Layout
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=42, k=2.0)

    node_scores = [abs(g.nodes[n].get("score", 0.0)) for n in g.nodes()]
    max_score = max(node_scores) if node_scores else 1.0
    node_sizes = [max(200, (s / (max_score + 1e-8)) * node_size_scale) for s in node_scores]

    type_colors = {
        "attn_head": "#4e79a7",
        "mlp_layer": "#f28e2b",
        "mlp_neuron": "#e15759",
        "residual": "#76b7b2",
        "embed": "#59a14f",
        "unembed": "#edc948",
    }
    node_colors = [
        type_colors.get(g.nodes[n].get("type", "residual"), "#aec7e8")
        for n in g.nodes()
    ]

    edge_weights = [g[u][v].get("weight", 0.1) for u, v in g.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [max(0.5, (w / (max_w + 1e-8)) * 3) for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(g, pos, width=edge_widths, ax=ax, arrows=True,
                           arrowstyle="->", arrowsize=15, alpha=0.6,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(
        g, pos,
        labels={n: g.nodes[n].get("label", n) for n in g.nodes()},
        font_size=8, ax=ax,
    )

    ax.set_title(title or graph.name, fontsize=13)
    ax.axis("off")

    # Legend
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=t)
        for t, c in type_colors.items()
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved circuit graph to '%s'", save_path)
    return fig


def plot_circuit_interactive(graph: CircuitGraph, title: Optional[str] = None):
    """Create an interactive Plotly visualisation of a circuit graph.

    Parameters
    ----------
    graph:
        The circuit to visualise.
    title:
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "Install plotly and networkx: pip install plotly networkx"
        ) from e

    g = graph.to_networkx()
    if len(g) == 0:
        return go.Figure(layout=go.Layout(title="Empty circuit graph"))

    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=42)

    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[n][0] for n in g.nodes()]
    node_y = [pos[n][1] for n in g.nodes()]
    node_labels = [g.nodes[n].get("label", str(n)) for n in g.nodes()]
    node_scores = [g.nodes[n].get("score", 0.0) for n in g.nodes()]
    node_types = [g.nodes[n].get("type", "?") for n in g.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_labels,
        textposition="top center",
        hovertext=[
            f"{label}<br>type: {t}<br>score: {s:.4f}"
            for label, t, s in zip(node_labels, node_types, node_scores)
        ],
        marker=dict(
            size=[max(10, abs(s) * 30 + 8) for s in node_scores],
            color=node_scores,
            colorscale="RdBu_r",
            colorbar=dict(title="Score"),
            line=dict(width=2, color="white"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title or graph.name,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=40, r=40, t=60, b=40),
        ),
    )
    return fig
