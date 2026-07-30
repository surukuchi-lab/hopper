"""
Module: hopper

Developer: ehtkarim
Date: April 29, 2026

Initializes the Hopper package and exposes package metadata for installed and editable builds.
"""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

try:
    __version__ = metadata.version("hopper")
except metadata.PackageNotFoundError:  # pragma: no cover - local editable import
    __version__ = "0.1.0"

__build__ = os.environ.get("HOPPER_BUILD_VERSION", "dev")
__version_info__ = f"{__version__}+{__build__}"


def run_from_config(*args: Any, **kwargs: Any):
    """Lazy import wrapper for :func:`hopper.nodes.pipeline.run_from_config`."""
    from .nodes.pipeline import run_from_config as _run_from_config

    return _run_from_config(*args, **kwargs)


__all__ = ["__version__", "__build__", "__version_info__", "run_from_config"]
