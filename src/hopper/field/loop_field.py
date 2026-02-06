from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import ellipe, ellipk

from ..constants import MU0


def loop_field_br_bz_cylindrical(
    r: np.ndarray,
    z: np.ndarray,
    a: float,
    I: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnetic field of a single circular loop of radius a (m), current I (A),
    lying in the plane z=0, centered on the symmetry axis.

    Returns (Br, Bz) in cylindrical coordinates at points (r,z).

    Uses the standard elliptic-integral expression.

    Notes:
      - r and z can be scalars or arrays; broadcasting is supported.
      - r must be >= 0 (cylindrical radius).
      - At r=0, Br is identically zero and Bz reduces to on-axis formula.
    """
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    a = float(a)
    I = float(I)

    r_b, z_b = np.broadcast_arrays(r, z)
    rr = np.abs(r_b)
    zz = z_b

    on_axis = rr < 1e-14

    Br = np.zeros_like(rr, dtype=float)
    Bz = np.zeros_like(rr, dtype=float)

    # On-axis: Bz = μ0 I a^2 / (2 (a^2 + z^2)^(3/2))
    if np.any(on_axis):
        zax = zz[on_axis]
        Bz[on_axis] = MU0 * I * a * a / (2.0 * np.power(a * a + zax * zax, 1.5))
        Br[on_axis] = 0.0

    # Off-axis: elliptic integral formulas
    if np.any(~on_axis):
        r1 = rr[~on_axis]
        z1 = zz[~on_axis]

        k2 = (4.0 * a * r1) / ((a + r1) ** 2 + z1**2)
        k2 = np.clip(k2, 0.0, 1.0 - 1e-15)

        K = ellipk(k2)
        E = ellipe(k2)

        denom = np.sqrt((a + r1) ** 2 + z1**2)
        rho2 = (a - r1) ** 2 + z1**2

        pref = MU0 * I / (2.0 * np.pi * denom)

        Bz_off = pref * (K + ((a * a - r1 * r1 - z1 * z1) / (rho2 + 1e-300)) * E)
        Br_off = pref * (z1 / (r1 + 1e-300)) * (-K + ((a * a + r1 * r1 + z1 * z1) / (rho2 + 1e-300)) * E)

        Br[~on_axis] = Br_off
        Bz[~on_axis] = Bz_off

    return Br, Bz
