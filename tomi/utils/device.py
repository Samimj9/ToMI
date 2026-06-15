"""
tomi/utils/device.py
--------------------
Device selection and tensor-movement helpers for ToMI.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tomi.utils.logging import get_logger

log = get_logger(__name__)


def get_best_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.debug("Selected device: %s", device)
    return device


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve an optional device specification to a ``torch.device``.

    Parameters
    ----------
    device:
        ``None`` (auto-select), a string like ``"cuda:0"`` or ``"cpu"``,
        or an existing ``torch.device``.

    Returns
    -------
    torch.device
    """
    if device is None:
        return get_best_device()
    return torch.device(device)


def move_to_device(
    obj: Union[torch.Tensor, dict, list],
    device: torch.device,
) -> Union[torch.Tensor, dict, list]:
    """Recursively move tensors to *device*.

    Supports plain tensors, dicts of tensors, and lists of tensors.

    Parameters
    ----------
    obj:
        Object to move.
    device:
        Target device.

    Returns
    -------
    Same type as *obj* with all tensors on *device*.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj
