from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cavity import Cavity

CHI01P: float = 3.8317059702075125  # first root of J1


def _j1(x: np.ndarray) -> np.ndarray:
    """
    Bessel J1 with a SciPy fallback.
    The fallback is a low-order series approximation adequate for small x.
    """
    try:
        from scipy.special import j1 as scipy_j1
        return scipy_j1(x)
    except Exception:
        x = np.asarray(x, float)
        # J1(x) ≈ x/2 - x^3/16 + x^5/384  (good for small x)
        return 0.5 * x - (x**3) / 16.0 + (x**5) / 384.0


class ModeMap(Protocol):
    def __call__(self, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AnalyticTE011ModeMap:
    """
    Analytic magnitude-only coupling proxy used in the notebook:

      |J1(χ r/a)| * |cos(π z/L)|

    where χ is the first root of J1.
    """
    cavity: Cavity

    def __call__(self, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        r = np.asarray(r_m, float)
        z = np.asarray(z_m, float)
        Cr = np.abs(_j1(CHI01P * r / self.cavity.radius_m))
        Cz = np.abs(np.cos(np.pi * z / self.cavity.length_m))
        return Cr * Cz
