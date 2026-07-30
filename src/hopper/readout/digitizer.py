"""
Module: hopper.readout.digitizer

Developer: ehtkarim
Date: April 29, 2026

Quantizes complex IQ samples and reports ADC saturation counts for configured readout ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DigitizeResult:
    adc_iq: np.ndarray
    saturation_count: int


def digitize_iq(
    iq: np.ndarray,
    *,
    v_range: float,
    v_offset: float,
    bit_depth: int = 8,
    signed: bool = False,
    strict_range: bool = True,
) -> DigitizeResult:
    """Digitize complex IQ into interleaved [I0,Q0,I1,Q1,...] integer codes."""
    x = np.asarray(iq, dtype=np.complex128)
    if float(v_range) <= 0.0:
        raise ValueError("v_range must be positive")
    n_codes = int(2 ** int(bit_depth))
    if n_codes < 2:
        raise ValueError("bit_depth must be positive")

    vals = np.empty(2 * x.size, dtype=float)
    vals[0::2] = np.real(x)
    vals[1::2] = np.imag(x)
    raw = np.floor((n_codes - 1) * (vals - float(v_offset)) / float(v_range) + 0.5)
    saturation = int(np.count_nonzero((raw <= 0.0) | (raw >= n_codes - 1)))
    if strict_range and saturation > 0:
        raise ValueError(f"digitizer input reached endpoint codes for {saturation} I/Q values")
    clipped = np.clip(raw, 0, n_codes - 1).astype(np.int64)
    if signed:
        clipped = clipped - (n_codes // 2)
    if bit_depth <= 8:
        dtype = np.int8 if signed else np.uint8
    elif bit_depth <= 16:
        dtype = np.int16 if signed else np.uint16
    elif bit_depth <= 32:
        dtype = np.int32 if signed else np.uint32
    else:
        dtype = np.int64 if signed else np.uint64
    return DigitizeResult(adc_iq=clipped.astype(dtype), saturation_count=saturation)
