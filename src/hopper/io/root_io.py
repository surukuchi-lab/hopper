from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


def _validate_array_chunk(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arrays_np = {k: np.asarray(v) for k, v in arrays.items()}
    if not arrays_np:
        raise ValueError("No arrays provided for ROOT output")

    lengths = {k: v.shape[0] for k, v in arrays_np.items()}
    n = next(iter(lengths.values()))
    for k, n_k in lengths.items():
        if n_k != n:
            raise ValueError(f"Length mismatch for branch {k}: {n_k} vs {n}")
    return arrays_np


def write_track_root(path: str | Path, arrays: Dict[str, np.ndarray], tree_name: str = "track") -> None:
    """
    Write a ROOT file containing a single TTree with the provided 1D arrays.

    Requires `uproot` (and its pure-python dependencies). This does NOT require
    a full ROOT installation.
    """
    arrays_np = _validate_array_chunk(arrays)
    write_track_root_chunks(path, [arrays_np], tree_name=tree_name)


def write_track_root_chunks(
    path: str | Path,
    chunks: Iterable[Mapping[str, np.ndarray]],
    tree_name: str = "track",
) -> None:
    """
    Stream a ROOT TTree from branch-array chunks.

    This is used for RF-sampled tracks, where phase_uniform may request many
    samples per cyclotron turn. Only one chunk of branch arrays stays in
    memory at a time.
    """
    try:
        import uproot  # type: ignore
    except Exception as e:
        raise ImportError("uproot is required for ROOT output (pip install uproot or pip install .[root])") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    iterator = iter(chunks)
    try:
        first = _validate_array_chunk(next(iterator))
    except StopIteration as exc:
        raise ValueError("No chunks provided for ROOT output") from exc

    branch_types = {name: arr.dtype for name, arr in first.items()}

    with uproot.recreate(str(path)) as f:
        f.mktree(tree_name, branch_types)
        f[tree_name].extend(first)
        for chunk in iterator:
            arrays_np = _validate_array_chunk(chunk)
            if set(arrays_np) != set(branch_types):
                missing = set(branch_types) - set(arrays_np)
                extra = set(arrays_np) - set(branch_types)
                raise ValueError(f"ROOT chunk branch mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
            f[tree_name].extend(arrays_np)
