"""
tomi/utils/logging.py
---------------------
Centralised logging for the ToMI library.

All modules should obtain their logger via ``get_logger(__name__)`` so that
end-users can control verbosity with a single call to ``set_log_level``.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_TOMI_ROOT_LOGGER = "tomi"

# Default format: time  level  name  message
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def _configure_root_logger() -> logging.Logger:
    """Create and configure the root ToMI logger (called once at import time)."""
    logger = logging.getLogger(_TOMI_ROOT_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # silent by default
    return logger


_root = _configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger namespaced under 'tomi'.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    # Ensure the name is rooted under 'tomi'
    if not name.startswith(_TOMI_ROOT_LOGGER):
        name = f"{_TOMI_ROOT_LOGGER}.{name}"
    return logging.getLogger(name)


def set_log_level(level: int | str = logging.INFO) -> None:
    """Globally set the verbosity of the ToMI library.

    Parameters
    ----------
    level:
        An integer log level (e.g. ``logging.DEBUG``) or a string
        (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Examples
    --------
    >>> import tomi
    >>> tomi.set_log_level("DEBUG")
    """
    _root.setLevel(level)
    for handler in _root.handlers:
        handler.setLevel(level)


def enable_debug() -> None:
    """Convenience wrapper: set level to DEBUG."""
    set_log_level(logging.DEBUG)


def enable_info() -> None:
    """Convenience wrapper: set level to INFO."""
    set_log_level(logging.INFO)


def silence() -> None:
    """Suppress all ToMI logging output."""
    set_log_level(logging.CRITICAL + 1)
