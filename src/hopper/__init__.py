"""
hopper

Synthetic CRES signal + track simulator.

Primary entry points:
- hopper.nodes.pipeline.run_from_config
- CLI: `hopper-sim <config.yaml>`
"""
from __future__ import annotations

import os
from importlib import metadata

from .nodes.pipeline import run_from_config  # noqa: E402

try:
    __version__ = metadata.version("hopper")
except metadata.PackageNotFoundError:  # pragma: no cover - local editable import
    __version__ = "0.1.0"

__build__ = os.environ.get("HOPPER_BUILD_VERSION", "dev")
__version_info__ = f"{__version__}+{__build__}"

__all__ = ["__version__", "__build__", "__version_info__", "run_from_config"]