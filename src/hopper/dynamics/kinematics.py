from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import C0, E_CHARGE, EPS0, M_E


def gamma_beta_v_from_kinetic(E_kin_eV: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relativistic factors from kinetic energy (eV).
    Matches the notebook conventions: gamma = 1 + E/511keV.
    """
    Ek = np.asarray(E_kin_eV, float)
    gamma = 1.0 + Ek / 511_000.0
    beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
    v = beta * C0
    return gamma, beta, v


def mu_from_pitch(E_kin_eV: float, pitch_angle_deg: float, B_T: float) -> float:
    """
    Magnetic moment µ (adiabatic invariant) at a point with local field magnitude B.

      µ = γ m v^2 sin^2(θ) / (2 B)
    """
    gamma, beta, v = gamma_beta_v_from_kinetic(E_kin_eV)
    sin2 = float(np.sin(np.deg2rad(pitch_angle_deg)) ** 2)
    return float(gamma * M_E * (v * v) * sin2 / (2.0 * float(B_T)))


def critical_B_from_mu(E_kin_eV: float, mu_J_per_T: float) -> float:
    """
    Turning-point field Bc:

      Bc = γ m v^2 / (2 µ)
    """
    gamma, beta, v = gamma_beta_v_from_kinetic(E_kin_eV)
    return float(gamma * M_E * (v * v) / (2.0 * float(mu_J_per_T)))


def vpar_from_E_mu_B(E_kin_eV: float, mu_J_per_T: float, B_T: float) -> float:
    """
    Parallel speed magnitude:

      v_parallel = v * sqrt(max(0, 1 - B/Bc))
    """
    gamma, beta, v = gamma_beta_v_from_kinetic(E_kin_eV)
    Bc = critical_B_from_mu(E_kin_eV, mu_J_per_T)
    arg = 1.0 - float(B_T) / (Bc + 1e-300)
    if arg <= 0.0:
        return 0.0
    return float(v * np.sqrt(arg))


def vperp_from_mu_B(gamma: float, mu_J_per_T: float, B_T: float) -> float:
    """
    Perpendicular speed magnitude from mu invariance:

      µ = γ m v_perp^2 / (2B)  =>  v_perp = sqrt(2 µ B / (γ m))
    """
    return float(np.sqrt(np.maximum(0.0, 2.0 * mu_J_per_T * float(B_T) / (float(gamma) * M_E))))


def cyclotron_frequency_hz(B_T: float | np.ndarray, gamma: float | np.ndarray, q_C: float = -E_CHARGE) -> np.ndarray:
    """
    Cyclotron frequency (Hz):

      f_c = |q| B / (2π γ m)
    """
    B = np.asarray(B_T, float)
    g = np.asarray(gamma, float)
    return np.abs(q_C) * B / (2.0 * np.pi * g * M_E)


def larmor_radius_m(mu_J_per_T: float, gamma: float, B_T: float, q_C: float = -E_CHARGE) -> float:
    """
    Relativistic Larmor radius from mu invariance:

      ρ = sqrt(2 µ γ m / B) / |q|
    """
    return float(np.sqrt(np.maximum(0.0, 2.0 * mu_J_per_T * float(gamma) * M_E / (float(B_T) + 1e-300))) / np.abs(q_C))


def larmor_power_W(B_T: float, E_kin_eV: float, mu_J_per_T: float, q_C: float = -E_CHARGE) -> float:
    """
    Larmor / synchrotron-like power proxy (Watts):

      P ∝ q^4 B^2 γ^2 β_perp^2 / (6π ε0 m^2 c)

    with β_perp^2 approximated via mu invariance:
      β_perp^2 = β^2 * (B/Bc)
    """
    gamma, beta, v = gamma_beta_v_from_kinetic(E_kin_eV)
    Bc = critical_B_from_mu(E_kin_eV, mu_J_per_T)
    beta_perp2 = float(beta * beta) * (float(B_T) / (Bc + 1e-300))
    P = (np.abs(q_C) ** 4 * (float(B_T) ** 2) * (float(gamma) ** 2) * beta_perp2) / (
        6.0 * np.pi * EPS0 * (M_E**2) * C0
    )
    return float(P)

def mu_from_pitch(energy_eV: float | np.ndarray, pitch_angle_deg: float, B0_T: float) -> float | np.ndarray:
    """
    Magnetic moment inferred by enforcing a fixed pitch angle at the reference field B0.

    NOTE: Under radiation, true μ is not strictly conserved; this is a modeling choice.
    Keeping pitch fixed makes B_crit (=B0/sin^2θ) independent of energy, which is what
    enables fast template time-scaling without recomputing turning points.
    """
    gamma, beta, v = gamma_beta_v_from_kinetic(energy_eV)
    sin2 = float(np.sin(np.deg2rad(pitch_angle_deg)) ** 2)
    return gamma * M_E * v * v * sin2 / (2.0 * B0_T)


def larmor_radius_m_array(B_T: np.ndarray, energy_eV: float | np.ndarray, mu_SI: float | np.ndarray) -> np.ndarray:
    B = np.asarray(B_T, dtype=float)
    gamma, _, _ = gamma_beta_v_from_kinetic(energy_eV)
    return np.sqrt(2.0 * np.asarray(mu_SI, dtype=float) * np.asarray(gamma, dtype=float) * M_E / np.maximum(B, 1e-30)) / E_CHARGE


def larmor_power_W_array(B_T: np.ndarray, energy_eV: float | np.ndarray, mu_SI: float | np.ndarray) -> np.ndarray:
    """
    Vectorized radiated power proxy. Broadcasts over inputs.
    """
    B = np.asarray(B_T, dtype=float)
    gamma, beta, v = gamma_beta_v_from_kinetic(energy_eV)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    v = np.asarray(v, dtype=float)
    mu = np.asarray(mu_SI, dtype=float)

    # Critical mirror field from μ definition:
    # Bc = (γ m v^2) / (2 μ)
    Bc = (gamma * M_E * v * v) / (2.0 * np.maximum(mu, 1e-30))

    # Approx β_perp^2 ≈ β^2 * B/Bc, clamped
    beta_perp2 = beta * beta * (B / np.maximum(Bc, 1e-30))
    beta_perp2 = np.clip(beta_perp2, 0.0, beta * beta)

    # Power proxy (same structure as scalar version)
    q = E_CHARGE
    pref = (q ** 4) / (6.0 * np.pi * EPS0 * (M_E ** 2) * C0)
    return pref * (B ** 2) * (gamma ** 2) * beta_perp2

