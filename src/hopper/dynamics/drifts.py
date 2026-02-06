from __future__ import annotations

import numpy as np

from ..constants import E_CHARGE
from ..utils.math import cumulative_trapezoid


def gradB_drift_vphi(
    mu_J_per_T: float,
    q_C: float,
    Bmag_T: np.ndarray,
    Br_T: np.ndarray | None,
    Bz_T: np.ndarray | None,
    dBdr_T_per_m: np.ndarray,
    dBdz_T_per_m: np.ndarray,
) -> np.ndarray:
    """
    Compute grad-B drift velocity in the azimuthal direction (phi-hat), assuming axisymmetry.

    General guiding-center formula:
      v_{∇B} = (µ / (q B^2)) (B × ∇B)

    In cylindrical basis (r,phi,z) with ∇B = (∂B/∂r) rhat + (∂B/∂z) zhat and B = Br rhat + Bz zhat:
      (B × ∇B)_phi = Bz * (∂B/∂r) - Br * (∂B/∂z)

    so:
      v_phi = µ / (q B^2) * ( Bz dBdr - Br dBdz )

    If Br/Bz components are unavailable, we fall back to Br=0, Bz=Bmag which yields:
      v_phi ≈ µ/(q B) * dBdr

    Note: for an electron, q is negative, so the drift direction is reversed vs positive charge.
    """
    Bmag = np.asarray(Bmag_T, float)
    dBdr = np.asarray(dBdr_T_per_m, float)
    dBdz = np.asarray(dBdz_T_per_m, float)

    if Br_T is None or Bz_T is None:
        Br = np.zeros_like(Bmag)
        Bz = Bmag
    else:
        Br = np.asarray(Br_T, float)
        Bz = np.asarray(Bz_T, float)

    cross_phi = Bz * dBdr - Br * dBdz
    vphi = float(mu_J_per_T) * cross_phi / (float(q_C) * (Bmag * Bmag + 1e-300))
    return vphi


def integrate_phi_from_vphi(t_s: np.ndarray, vphi_m_per_s: np.ndarray, r0_m: float, phi0_rad: float) -> np.ndarray:
    """
    Integrate azimuthal angle phi(t) from azimuthal speed vphi and radius r0:
      dphi/dt = vphi / r0

    If r0==0, returns constant phi0.
    """
    t = np.asarray(t_s, float)
    vphi = np.asarray(vphi_m_per_s, float)
    if t.size != vphi.size:
        raise ValueError("t and vphi size mismatch")
    if abs(r0_m) < 1e-14:
        return np.full_like(t, float(phi0_rad), dtype=float)
    dphi_dt = vphi / float(r0_m)
    return cumulative_trapezoid(dphi_dt, t, initial=float(phi0_rad))
