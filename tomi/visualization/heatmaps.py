"""
tomi/visualization/heatmaps.py
--------------------------------
Matplotlib / Plotly heatmap utilities for activation patching results.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from tomi.utils.logging import get_logger

log = get_logger(__name__)


def plot_patching_heatmap(
    importance_matrix: Union["np.ndarray", "torch.Tensor"],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Activation Patching Importance",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = False,
    save_path: Optional[str] = None,
):
    """Plot a 2-D heatmap of patching importance scores.

    Parameters
    ----------
    importance_matrix:
        2-D array of shape ``(n_rows, n_cols)``.
    x_labels:
        Column labels (e.g. component names).
    y_labels:
        Row labels (e.g. layer indices as strings).
    title:
        Plot title.
    figsize:
        Matplotlib figure size.
    cmap:
        Colormap name.
    vmin:
        Color-scale minimum.
    vmax:
        Color-scale maximum.
    annotate:
        Whether to annotate each cell with its value.
    save_path:
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib: pip install matplotlib") from e

    # Convert to numpy
    if hasattr(importance_matrix, "numpy"):
        mat = importance_matrix.detach().cpu().float().numpy()
    else:
        mat = np.array(importance_matrix, dtype=float)

    n_rows, n_cols = mat.shape

    if y_labels is None:
        y_labels = [str(i) for i in range(n_rows)]
    if x_labels is None:
        x_labels = [str(j) for j in range(n_cols)]

    abs_max = np.abs(mat).max()
    if vmin is None:
        vmin = -abs_max
    if vmax is None:
        vmax = abs_max

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if abs(mat[i, j]) < abs_max * 0.6 else "white",
                )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved heatmap to '%s'", save_path)

    return fig


def plot_layer_importance(
    scores: Union[List[float], "np.ndarray"],
    layer_labels: Optional[List[str]] = None,
    title: str = "Layer Importance",
    figsize: Tuple[int, int] = (10, 4),
    color: str = "steelblue",
    save_path: Optional[str] = None,
):
    """Plot a bar chart of per-layer importance scores.

    Parameters
    ----------
    scores:
        1-D sequence of importance values (one per layer).
    layer_labels:
        X-axis labels.  Defaults to ``["L0", "L1", …]``.
    title:
        Plot title.
    figsize:
        Figure size.
    color:
        Bar colour.
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib: pip install matplotlib") from e

    arr = np.array(scores, dtype=float)
    n = len(arr)
    if layer_labels is None:
        layer_labels = [f"L{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(layer_labels, arr, color=color, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Importance (NCE)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
