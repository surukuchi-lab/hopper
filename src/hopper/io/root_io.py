from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def write_track_root(path: str | Path, arrays: Dict[str, np.ndarray], tree_name: str = "track") -> None:
    """
    Write a ROOT file containing a single TTree with the provided 1D arrays.

    Requires `uproot` (and its pure-python dependencies). This does NOT require
    a full ROOT installation.

    Parameters
    ----------
    arrays:
      dict of branch_name -> 1D numpy array
    """
    try:
        import uproot  # type: ignore
    except Exception as e:
        raise ImportError("uproot is required for ROOT output (pip install uproot or pip install .[root])") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lengths = {k: np.asarray(v).shape[0] for k, v in arrays.items()}
    if not lengths:
        raise ValueError("No arrays provided for ROOT output")
    n = next(iter(lengths.values()))
    for k, n_k in lengths.items():
        if n_k != n:
            raise ValueError(f"Length mismatch for branch {k}: {n_k} vs {n}")

    arrays_np = {k: np.asarray(v) for k, v in arrays.items()}

    with uproot.recreate(str(path)) as f:
        f[tree_name] = arrays_np
