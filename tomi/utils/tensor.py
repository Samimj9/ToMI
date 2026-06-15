"""
tomi/utils/tensor.py
--------------------
Common tensor manipulation utilities used throughout ToMI.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (possibly GPU) tensor to a NumPy array.

    Parameters
    ----------
    tensor:
        Input tensor.

    Returns
    -------
    np.ndarray
    """
    return tensor.detach().cpu().float().numpy()


def batch_index_select(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Index *tensor* along *dim* using *indices* with broadcasting.

    Parameters
    ----------
    tensor:
        Source tensor of shape ``(..., D, ...)``.
    indices:
        1-D or 2-D index tensor.
    dim:
        Dimension to index.

    Returns
    -------
    torch.Tensor
    """
    return tensor.index_select(dim, indices)


def get_token_position(
    tokens: torch.Tensor,
    token_id: int,
    occurrence: int = 0,
) -> int:
    """Return the position of a token id's *n*-th occurrence.

    Parameters
    ----------
    tokens:
        1-D token tensor.
    token_id:
        The token id to search for.
    occurrence:
        Zero-based index of the occurrence to return.

    Returns
    -------
    int
        Position of the token.

    Raises
    ------
    ValueError
        If the token is not found the requested number of times.
    """
    positions = (tokens == token_id).nonzero(as_tuple=True)[0]
    if len(positions) <= occurrence:
        raise ValueError(
            f"Token id {token_id} found only {len(positions)} time(s), "
            f"but occurrence {occurrence} was requested."
        )
    return int(positions[occurrence].item())


def residual_diff(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
) -> torch.Tensor:
    """Compute element-wise absolute difference between two residual stream tensors.

    Parameters
    ----------
    clean:
        Clean run activations.
    corrupted:
        Corrupted run activations.

    Returns
    -------
    torch.Tensor
        Absolute difference tensor with same shape as inputs.
    """
    return (clean - corrupted).abs()


def normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize *tensor* along *dim*.

    Parameters
    ----------
    tensor:
        Input tensor.
    dim:
        Dimension along which to normalise.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
    """
    return tensor / (tensor.norm(dim=dim, keepdim=True) + eps)


def einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Thin wrapper around ``torch.einsum`` for convenience.

    Parameters
    ----------
    equation:
        Einstein summation string.
    *operands:
        Tensors to contract.

    Returns
    -------
    torch.Tensor
    """
    return torch.einsum(equation, *operands)
