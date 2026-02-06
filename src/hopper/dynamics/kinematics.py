from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import C0, E_CHARGE, EPS0, M_E

# NOTE:
# - E_CHARGE is assumed to be the *magnitude* of the electron charge (Coulombs).
# - Where a charge sign matters (e.g. drifts), code should pass q_C=-E_CHARGE for an electron.
# - For cyclotron frequency and Larmor radius magnitudes, we use |q| so the results are >= 0.


def gamma_beta_v_from_kinetic(E_kin_eV: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relativistic factors from kinetic energy (eV).
    Matches the notebook conventions: gamma = 1 + E/511keV.

    Returns
    -------
    gamma : ndarray
    beta  : ndarray
    v     : ndarray (m/s)
    """
    Ek = np.asarray(E_kin_eV, dtype=float)
    gamma = 1.0 + Ek / 511_000.0
    beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
    v = beta * C0
    return gamma, beta, v


def cyclotron_frequency_hz(
    B_T: float | np.ndarray,
    gamma: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """
    Cyclotron frequency (Hz): f_c = |q| B / (2π γ m_e).
    """
    B = np.asarray(B_T, dtype=float)
    g = np.asarray(gamma, dtype=float)
    q = float(np.abs(q_C))
    return (q * B) / (2.0 * np.pi * np.maximum(g, 1e-30) * M_E)


def mu_from_pitch(
    energy_eV: float | np.ndarray,
    pitch_angle_deg: float,
    B_ref_T: float,
) -> float | np.ndarray:
    """
    Magnetic moment µ inferred from pitch angle at a reference field B_ref:

        µ = γ m v^2 sin^2(θ) / (2 B_ref)
    """
    gamma, _, v = gamma_beta_v_from_kinetic(energy_eV)
    sin2 = float(np.sin(np.deg2rad(pitch_angle_deg)) ** 2)
    mu = gamma * M_E * v * v * sin2 / (2.0 * float(B_ref_T))
    if np.ndim(mu) == 0:
        return float(mu)
    return np.asarray(mu, dtype=float)


def critical_B_from_mu(
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> float | np.ndarray:
    """
    Mirror (turning-point) field Bc implied by (E, µ):

        Bc = (γ m v^2) / (2 µ)

    With µ computed from a fixed pitch at B0, this reduces to Bc = B0 / sin^2(theta), independent of energy.
    """
    gamma, _, v = gamma_beta_v_from_kinetic(energy_eV)
    mu = np.asarray(mu_J_per_T, dtype=float)
    Bc = (np.asarray(gamma, dtype=float) * M_E * v * v) / (2.0 * np.maximum(mu, 1e-30))
    if np.ndim(Bc) == 0:
        return float(Bc)
    return np.asarray(Bc, dtype=float)


def vpar_m_per_s_from_B(
    B_T: float | np.ndarray,
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> np.ndarray:
    """
    Parallel speed magnitude from (B, E, µ) under the adiabatic approximation:

        v_par(B) = v * sqrt(max(0, 1 - B/Bc))

    where Bc is the critical mirror field from (E, µ).
    """
    B = np.asarray(B_T, dtype=float)
    gamma, _, v = gamma_beta_v_from_kinetic(energy_eV)
    Bc = np.asarray(critical_B_from_mu(energy_eV, mu_J_per_T), dtype=float)
    arg = 1.0 - B / np.maximum(Bc, 1e-30)
    return np.asarray(v, dtype=float) * np.sqrt(np.maximum(arg, 0.0))


def larmor_radius_m_array(
    B_T: float | np.ndarray,
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """
    Relativistic Larmor radius (vectorized):

    ρ = sqrt(2 µ γ m / B) / |q|
    """
    B = np.asarray(B_T, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    gamma = np.asarray(gamma_beta_v_from_kinetic(energy_eV)[0], dtype=float)
    q = float(np.abs(q_C))
    return np.sqrt(2.0 * mu * gamma * M_E / np.maximum(B, 1e-30)) / max(q, 1e-30)


def larmor_radius_m(
    mu_J_per_T: float,
    gamma: float,
    B_T: float,
    *,
    q_C: float = -E_CHARGE,
) -> float:
    
    q = float(np.abs(q_C))
    return float(np.sqrt(2.0 * float(mu_J_per_T) * float(gamma) * M_E / max(float(B_T), 1e-30)) / max(q, 1e-30))


def larmor_power_W_array(
    B_T: float | np.ndarray,
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """
    Radiated power proxy (vectorized):

    P ∝ q^4 B^2 γ^2 β_perp^2 / (6π ε0 m_e^2 c)

    with the adiabatic approximation:
    
    β_perp^2 ≈ β^2 * (B/Bc), clipped to [0, β^2].
    """
    B = np.asarray(B_T, dtype=float)
    gamma, beta, _ = gamma_beta_v_from_kinetic(energy_eV)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)

    # Bc = (γ m v^2) / (2 µ)
    _, _, v = gamma_beta_v_from_kinetic(energy_eV)
    Bc = (gamma * M_E * np.asarray(v, dtype=float) ** 2) / (2.0 * np.maximum(mu, 1e-30))

    beta_perp2 = beta * beta * (B / np.maximum(Bc, 1e-30))
    beta_perp2 = np.clip(beta_perp2, 0.0, beta * beta)

    q = float(np.abs(q_C))
    pref = (q**4) / (6.0 * np.pi * EPS0 * (M_E**2) * C0)
    return pref * (B**2) * (gamma**2) * beta_perp2


def larmor_power_W(
    B_T: float,
    energy_eV: float,
    mu_J_per_T: float,
    *,
    q_C: float = -E_CHARGE,
) -> float:
    """
    Scalar wrapper for the vectorized radiated power proxy.
    """
    return float(larmor_power_W_array(float(B_T), float(energy_eV), float(mu_J_per_T), q_C=q_C))
