"""
tomi/visualization/attention.py
---------------------------------
Attention pattern visualization utilities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from tomi.utils.logging import get_logger

log = get_logger(__name__)


def plot_attention_pattern(
    pattern: Union["np.ndarray", "torch.Tensor"],
    token_labels: Optional[List[str]] = None,
    head: Optional[int] = None,
    layer: Optional[int] = None,
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
):
    """Plot a single attention pattern as a heatmap.

    Parameters
    ----------
    pattern:
        Attention weights of shape ``(seq_len, seq_len)`` or
        ``(batch, seq_len, seq_len)`` (first batch element is used).
    token_labels:
        Token strings for axis labels.
    head:
        Head index (for title).
    layer:
        Layer index (for title).
    figsize:
        Figure size.
    cmap:
        Colormap.
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

    if hasattr(pattern, "numpy"):
        pat = pattern.detach().cpu().float().numpy()
    else:
        pat = np.array(pattern, dtype=float)

    # Squeeze batch dimension
    if pat.ndim == 3:
        pat = pat[0]
    if pat.ndim != 2:
        raise ValueError(f"Expected 2-D attention pattern, got shape {pat.shape}.")

    n = pat.shape[0]
    if token_labels is None:
        token_labels = [str(i) for i in range(n)]

    title_parts = []
    if layer is not None:
        title_parts.append(f"Layer {layer}")
    if head is not None:
        title_parts.append(f"Head {head}")
    title = "Attention Pattern" + (f" — {', '.join(title_parts)}" if title_parts else "")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_xlabel("Key (attends to)")
    ax.set_ylabel("Query (token)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_head_importance(
    importance_matrix: Union["np.ndarray", "torch.Tensor"],
    title: str = "Head Importance",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """Plot a (n_layers × n_heads) head-importance heatmap.

    Parameters
    ----------
    importance_matrix:
        Shape ``(n_layers, n_heads)``.
    title:
        Plot title.
    figsize:
        Figure size.
    cmap:
        Colormap.
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

    if hasattr(importance_matrix, "numpy"):
        mat = importance_matrix.detach().cpu().float().numpy()
    else:
        mat = np.array(importance_matrix, dtype=float)

    n_layers, n_heads = mat.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=mat.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised Causal Effect")

    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_heads(
    patterns: Union["np.ndarray", "torch.Tensor"],
    token_labels: Optional[List[str]] = None,
    layer: int = 0,
    ncols: int = 4,
    figsize_per_head: Tuple[int, int] = (3, 3),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
):
    """Plot all attention heads in a single figure with subplots.

    Parameters
    ----------
    patterns:
        Attention weights of shape ``(n_heads, seq_len, seq_len)``.
    token_labels:
        Token labels.
    layer:
        Layer index (for title).
    ncols:
        Number of subplot columns.
    figsize_per_head:
        Size per head subplot.
    cmap:
        Colormap.
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

    if hasattr(patterns, "numpy"):
        pats = patterns.detach().cpu().float().numpy()
    else:
        pats = np.array(patterns, dtype=float)

    if pats.ndim == 4:
        pats = pats[0]  # take first batch element

    n_heads = pats.shape[0]
    n = pats.shape[1]
    nrows = (n_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_head[0] * ncols, figsize_per_head[1] * nrows),
    )
    axes = np.array(axes).flatten()

    if token_labels is None:
        token_labels = [str(i) for i in range(n)]

    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(pats[h], cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"L{layer}H{h}", fontsize=9)
        ax.axis("off")

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Layer {layer} — All Attention Heads", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
