from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def write_iq_npz(path: str | Path, t_s: np.ndarray, iq: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Write a compressed NPZ file with raw complex IQ samples.

    Stored keys:
      - t_s: time array [s]
      - iq: complex IQ array (Voltage-like units)
      - meta_json: JSON-encoded metadata (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "t_s": np.asarray(t_s, float),
        "iq": np.asarray(iq, np.complex128),
    }
    if meta is not None:
        out["meta_json"] = json.dumps(meta, sort_keys=True)

    np.savez_compressed(path, **out)
