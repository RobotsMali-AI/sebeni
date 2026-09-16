"""Sebeni — self-aware morphotactic generation for extremely low-resource languages."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("sebeni")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
