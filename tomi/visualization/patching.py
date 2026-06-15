"""
tomi/visualization/patching.py
--------------------------------
Visualisation helpers specifically designed for patching result objects.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from tomi.patching.activation_patching import PatchingResult
from tomi.patching.head_patching import HeadPatchingResult
from tomi.patching.residual_patching import ResidualPatchingResult
from tomi.utils.logging import get_logger

log = get_logger(__name__)


def plot_activation_patching_results(
    result: PatchingResult,
    n_layers: Optional[int] = None,
    components: Tuple[str, ...] = ("resid.post", "attn.out", "mlp.post"),
    title: str = "Activation Patching Results",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot patching importance as a layer × component heatmap.

    Parameters
    ----------
    result:
        :class:`~tomi.patching.activation_patching.PatchingResult`.
    n_layers:
        Number of layers.  Auto-detected from the hook names if ``None``.
    components:
        Component slots to display.
    title:
        Plot title.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from tomi.visualization.heatmaps import plot_patching_heatmap

    if n_layers is None:
        layers = set()
        for name in result.hook_names:
            parts = name.split(".")
            if parts[0] == "blocks":
                try:
                    layers.add(int(parts[1]))
                except ValueError:
                    pass
        n_layers = max(layers) + 1 if layers else 1

    mat = result.as_layer_matrix(n_layers, components)
    x_labels = list(components)
    y_labels = [f"L{i}" for i in range(n_layers)]

    return plot_patching_heatmap(
        mat.numpy(),
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        figsize=figsize,
        cmap="RdBu_r",
        save_path=save_path,
    )


def plot_head_patching_results(
    result: HeadPatchingResult,
    title: str = "Head Patching Importance",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
):
    """Plot head-patching importance as a (layer × head) heatmap.

    Parameters
    ----------
    result:
        :class:`~tomi.patching.head_patching.HeadPatchingResult`.
    title:
        Plot title.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from tomi.visualization.attention import plot_head_importance

    return plot_head_importance(
        result.importance_matrix,
        title=title,
        figsize=figsize,
        save_path=save_path,
    )


def plot_residual_patching_results(
    result: ResidualPatchingResult,
    title: str = "Residual Stream Patching",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot residual-patching importance as (layer × token position).

    Parameters
    ----------
    result:
        :class:`~tomi.patching.residual_patching.ResidualPatchingResult`.
    title:
        Plot title.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from tomi.visualization.heatmaps import plot_patching_heatmap

    x_labels = (
        result.token_labels
        if result.token_labels is not None
        else [str(p) for p in range(result.n_positions)]
    )
    y_labels = [f"L{i}" for i in range(result.n_layers)]

    return plot_patching_heatmap(
        result.importance_matrix.numpy(),
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        figsize=figsize,
        cmap="RdBu_r",
        save_path=save_path,
    )
