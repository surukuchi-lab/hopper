"""
Module: hopper.io.npz_io

Developer: ehtkarim
Date: April 29, 2026

Writes IQ time streams, metadata, and optional track arrays to compressed NPZ outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def write_iq_npz(
    path: str | Path,
    t_s: np.ndarray,
    iq: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    *,
    adc_iq: Optional[np.ndarray] = None,
    iq_fast: Optional[np.ndarray] = None,
    t_fast_s: Optional[np.ndarray] = None,
) -> None:
    """
    Write a compressed NPZ file with raw complex IQ samples.

    Stored keys:
      - t_s: time array [s]
      - iq: complex IQ array (Voltage-like units)
      - meta_json: JSON-encoded metadata (optional)
      - adc_iq: optional interleaved integer I/Q ADC codes
      - iq_fast, t_fast_s: optional pre-decimation baseband diagnostic arrays
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "t_s": np.asarray(t_s, float),
        "iq": np.asarray(iq, np.complex128),
    }
    if adc_iq is not None:
        out["adc_iq"] = np.asarray(adc_iq)
    if iq_fast is not None:
        out["iq_fast"] = np.asarray(iq_fast, np.complex128)
    if t_fast_s is not None:
        out["t_fast_s"] = np.asarray(t_fast_s, float)
    if meta is not None:
        out["meta_json"] = json.dumps(meta, sort_keys=True)

    np.savez_compressed(path, **out)
