"""
Module: hopper.readout.noise

Developer: ehtkarim
Date: April 29, 2026

Generates complex white-noise samples with PSD and impedance normalization.
"""

from __future__ import annotations

import numpy as np


def complex_white_noise(
    n: int,
    *,
    fs_hz: float,
    noise_floor_psd_W_per_Hz: float,
    impedance_ohm: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return complex WGN with Locust-like per-quadrature voltage RMS."""
    if rng is None:
        rng = np.random.default_rng()
    sigma_q = np.sqrt(max(float(noise_floor_psd_W_per_Hz), 0.0) * float(impedance_ohm) * float(fs_hz) / 2.0)
    return rng.normal(0.0, sigma_q, int(n)) + 1j * rng.normal(0.0, sigma_q, int(n))
