from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


def resample_linear(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation wrapper that supports real or complex y.
    Assumes t_src is strictly increasing.
    """
    t_src = np.asarray(t_src, float)
    t_tgt = np.asarray(t_tgt, float)
    y_src = np.asarray(y_src)

    if y_src.ndim != 1:
        raise ValueError("resample_linear expects 1D y_src")
    if t_src.ndim != 1 or t_tgt.ndim != 1:
        raise ValueError("resample_linear expects 1D time arrays")
    if t_src.size != y_src.size:
        raise ValueError("t_src and y_src size mismatch")

    if np.iscomplexobj(y_src):
        re = np.interp(t_tgt, t_src, np.real(y_src))
        im = np.interp(t_tgt, t_src, np.imag(y_src))
        return re + 1j * im
    return np.interp(t_tgt, t_src, y_src.astype(float))


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """
    Cumulative trapezoidal integral of y(x) over x.
    Returns array same length as x with out[0]=initial.
    Supports real or complex y.
    """
    x = np.asarray(x, float)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("cumulative_trapezoid expects 1D arrays")
    if x.size != y.size:
        raise ValueError("x and y size mismatch")

    dx = np.diff(x)
    if dx.size == 0:
        return np.asarray([initial], dtype=y.dtype if np.iscomplexobj(y) else float)

    # trap areas
    areas = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(y, dtype=areas.dtype)
    out[0] = initial
    out[1:] = initial + np.cumsum(areas)
    return out


def unwrap_angle(theta: np.ndarray) -> np.ndarray:
    """Unwrap a phase/angle array (radians)."""
    return np.unwrap(np.asarray(theta, float))


def robust_percentile(x: np.ndarray, p: float, default: float = 0.0) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float(default)
    try:
        return float(np.nanpercentile(x, p))
    except Exception:
        return float(default)


@dataclass(frozen=True)
class TimeGrid:
    t0: float
    dt: float
    n: int

    @property
    def t(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.n, dtype=float)
