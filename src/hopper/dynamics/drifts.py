from __future__ import annotations

import numpy as np

from ..constants import M_E
from ..utils.math import cumulative_trapezoid


def gradB_drift_vphi(
    mu_J_per_T: float | np.ndarray,
    q_C: float,
    Bmag_T: np.ndarray,
    Br_T: np.ndarray | None,
    Bz_T: np.ndarray | None,
    dBdr_T_per_m: np.ndarray,
    dBdz_T_per_m: np.ndarray,
    *,
    gamma: float | np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the azimuthal grad-B drift velocity for an axisymmetric field.

    This repo uses the adiabatic invariant convention

        μ = γ m_e v_perp^2 / (2 B)

    With this convention, the relativistic correction is already absorbed into μ,
    so the guiding-center grad-B drift becomes,

        v_{∇B,φ} = μ / (q B^2) * (Bz dB/dr - Br dB/dz)
    """
    mu = np.asarray(mu_J_per_T, float)
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
    vphi = mu * cross_phi / (float(q_C) * (Bmag * Bmag + 1e-300))
    return np.asarray(vphi, dtype=float)


def curvature_drift_vphi(
    gamma: float | np.ndarray,
    vpar_m_per_s: float | np.ndarray,
    q_C: float,
    Bmag_T: float | np.ndarray,
    b_cross_kappa_phi_per_m: float | np.ndarray,
) -> np.ndarray:
    """
    Compute the azimuthal curvature drift velocity for the cached field line.

    Relativistic guiding-center curvature drift:

        v_{curv} = γ m_e v_parallel^2 / (q B) * b × κ,
        κ = (b·∇)b.

    The field-line cache supplies the phi component of ``b × κ``.  This drift is
    zero for a straight axial field and is typically the leading correction missing
    when comparing a grad-B-only guiding-center model against a high-fidelity
    Lorentz/RK track in a magnetic bottle.
    """
    g = np.maximum(np.asarray(gamma, dtype=float), 1.0)
    vpar = np.asarray(vpar_m_per_s, dtype=float)
    B = np.asarray(Bmag_T, dtype=float)
    b_cross_kappa_phi = np.asarray(b_cross_kappa_phi_per_m, dtype=float)
    return np.asarray(g * M_E * vpar * vpar * b_cross_kappa_phi / (float(q_C) * (B + 1e-300)), dtype=float)


def integrate_phi_from_vphi(
    t_s: np.ndarray,
    vphi_m_per_s: np.ndarray,
    radius_m: float | np.ndarray,
    phi0_rad: float,
) -> np.ndarray:
    """
    Integrate azimuthal angle phi(t) from azimuthal speed vphi and local radius:
      dphi/dt = vphi / r(t).

    The radius may be scalar or array-valued; array support is used when the guiding center
    follows a precomputed magnetic field line rather than a constant-r cylinder.
    """
    t = np.asarray(t_s, float)
    vphi = np.asarray(vphi_m_per_s, float)
    if t.size != vphi.size:
        raise ValueError("t and vphi size mismatch")

    r = np.asarray(radius_m, dtype=float)
    if r.ndim == 0:
        r = np.full_like(t, float(r), dtype=float)
    if r.size != t.size:
        raise ValueError("radius_m must be scalar or have the same size as t_s")

    dphi_dt = np.zeros_like(t, dtype=float)
    np.divide(vphi, r, out=dphi_dt, where=np.abs(r) >= 1.0e-14)
    return cumulative_trapezoid(dphi_dt, t, initial=float(phi0_rad))
