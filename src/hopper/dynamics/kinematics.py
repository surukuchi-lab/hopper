from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import C0, E_CHARGE, EPS0, M_E, MEC2_EV

# NOTE:
# - E_CHARGE is the magnitude of the electron charge (Coulombs).
# - Where charge sign matters (e.g. drifts), callers should pass q_C=-E_CHARGE.
# - Frequency, radius, and radiated-power magnitudes use |q|.


def gamma_beta_v_from_kinetic(E_kin_eV: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relativistic factors from kinetic energy (eV).

        gamma = 1 + E / (m_e c^2)
        beta  = sqrt(1 - gamma^-2)
        v     = beta c
    """
    Ek = np.asarray(E_kin_eV, dtype=float)
    gamma = 1.0 + Ek / MEC2_EV
    beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
    v = beta * C0
    return gamma, beta, v


def kinetic_energy_eV_from_gamma(gamma: float | np.ndarray) -> np.ndarray:
    """Kinetic energy (eV) from relativistic gamma."""
    g = np.asarray(gamma, dtype=float)
    return np.maximum((g - 1.0) * MEC2_EV, 0.0)


def cyclotron_frequency_hz(
    B_T: float | np.ndarray,
    gamma: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """Cyclotron frequency magnitude (Hz): f_c = |q| B / (2π γ m_e)."""
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

        µ = γ m_e v_perp^2 / (2 B_ref)
          = γ m_e v^2 sin^2(theta) / (2 B_ref)
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
    Mirror field implied by (E, µ):

        Bc = (γ m_e v^2) / (2 µ)
    """
    gamma, _, v = gamma_beta_v_from_kinetic(energy_eV)
    mu = np.asarray(mu_J_per_T, dtype=float)
    Bc = (np.asarray(gamma, dtype=float) * M_E * np.asarray(v, dtype=float) ** 2) / (2.0 * np.maximum(mu, 1e-30))
    if np.ndim(Bc) == 0:
        return float(Bc)
    return np.asarray(Bc, dtype=float)


def beta_parallel2_from_B_gamma_mu(
    B_T: float | np.ndarray,
    gamma: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> np.ndarray:
    """
    Parallel velocity fraction squared under the guiding-center adiabatic model:

        β_parallel^2 = 1 - γ^-2 - β_perp^2,
        β_perp^2     = 2 µ B / (γ m_e c^2).

    The returned value is *not* clipped. Negative values indicate that the state is beyond the mirror condition for the supplied (B, γ, µ).
    """
    B = np.asarray(B_T, dtype=float)
    g = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    g_safe = np.maximum(g, 1.0)
    beta2 = 1.0 - 1.0 / (g_safe * g_safe)
    beta_perp2 = (2.0 * mu * B) / (g_safe * M_E * (C0**2))
    return beta2 - beta_perp2


def parallel_u2_from_B_gamma_mu(
    B_T: float | np.ndarray,
    gamma: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> np.ndarray:
    """
    Squared dimensionless parallel 4-speed,

        u_parallel^2 = (γ β_parallel)^2 = γ^2 - 1 - 2 µ B γ / (m_e c^2).

    This quantity is convenient for radiation-reaction updates because, in the local gyro-averaged loss model, u_parallel is held fixed during the short radiative substep while γ and µ evolve.
    """
    B = np.asarray(B_T, dtype=float)
    g = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    u2 = g * g - 1.0 - (2.0 * mu * B * g) / (M_E * (C0**2))
    return np.maximum(u2, 0.0)


def pitch_angle_deg_from_B_gamma_mu(
    B_T: float | np.ndarray,
    gamma: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> np.ndarray:
    """Local pitch angle (degrees) implied by (B, γ, µ)."""
    g = np.asarray(gamma, dtype=float)
    beta2 = np.maximum(0.0, 1.0 - 1.0 / np.maximum(g, 1.0) ** 2)
    beta_perp2 = np.clip(
        (2.0 * np.asarray(mu_J_per_T, dtype=float) * np.asarray(B_T, dtype=float)) / (np.maximum(g, 1.0) * M_E * (C0**2)),
        0.0,
        None,
    )
    sin2 = np.divide(beta_perp2, np.maximum(beta2, 1e-30))
    sin2 = np.clip(sin2, 0.0, 1.0)
    return np.rad2deg(np.arcsin(np.sqrt(sin2)))


def vpar_m_per_s_from_B(
    B_T: float | np.ndarray,
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
) -> np.ndarray:
    """
    Parallel speed magnitude from (B, E, µ):

        v_parallel(B) = c * sqrt(max(β_parallel^2, 0)).
    """
    gamma = np.asarray(gamma_beta_v_from_kinetic(energy_eV)[0], dtype=float)
    beta_par2 = beta_parallel2_from_B_gamma_mu(B_T, gamma, mu_J_per_T)
    return C0 * np.sqrt(np.maximum(beta_par2, 0.0))


def larmor_radius_m_array(
    B_T: float | np.ndarray,
    energy_eV: float | np.ndarray,
    mu_J_per_T: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """
    Relativistic Larmor radius (vectorized):

        ρ = sqrt(2 µ γ m_e / B) / |q|.
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
    """Scalar wrapper for the relativistic Larmor radius."""
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
    Free-space cyclotron/synchrotron radiated power (vectorized):
        P = q^4 B^2 γ^2 β_perp^2 / (6π ε0 m_e^2 c)
    with β_perp obtained from the supplied µ through
        v_perp^2 = 2 µ B / (γ m_e).
    """
    B = np.asarray(B_T, dtype=float)
    gamma, beta, _ = gamma_beta_v_from_kinetic(energy_eV)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)

    beta_perp2 = (2.0 * mu * B) / (np.maximum(gamma, 1e-30) * M_E * (C0**2))
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
    """Scalar wrapper for the vectorized radiated-power expression."""
    return float(larmor_power_W_array(float(B_T), float(energy_eV), float(mu_J_per_T), q_C=q_C))

def radiative_drag_eta_per_s(
    B_T: float | np.ndarray,
    *,
    q_C: float = -E_CHARGE,
) -> np.ndarray:
    """
    Gyro-averaged synchrotron drag coefficient η(B) in the exact local loss law

        dγ/dt = -η(B) (γ^2 - γ_parallel^2),

    where γ_parallel = sqrt(1 + u_parallel^2) and u_parallel is held fixed during the local radiation substep. In SI units,

        η(B) = |q|^4 B^2 / (6π ε0 m_e^3 c^3).
    """
    B = np.asarray(B_T, dtype=float)
    q = float(np.abs(q_C))
    pref = (q**4) / (6.0 * np.pi * EPS0 * (M_E**3) * (C0**3))
    return pref * (B**2)


def gamma_mu_after_radiation_step_fixed_upar(
    gamma0: float | np.ndarray,
    mu0_J_per_T: float | np.ndarray,
    B_T: float | np.ndarray,
    dt_s: float | np.ndarray,
    *,
    energy_loss_scale: float = 1.0,
    gamma_floor: float = 1.0,
    q_C: float = -E_CHARGE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact local radiation-reaction update at fixed magnetic field and fixed u_parallel.

    The short radiative substep follows the pitch-angle-damping picture found in the synchrotron-radiation literature:
    the component parallel to the magnetic field is unchanged during the local loss step, while the perpendicular component is damped.
    In terms of u_parallel = γ β_parallel, this gives the exact scalar ODE

        dγ/dt = -η(B) (γ^2 - γ_parallel^2),
        γ_parallel = sqrt(1 + u_parallel^2),

    with solution

        r0 = (γ0 - γ_parallel) / (γ0 + γ_parallel),
        r1 = r0 exp(-2 η γ_parallel Δt),
        γ1 = γ_parallel (1 + r1) / (1 - r1).

    The updated magnetic moment is then reconstructed from the same fixed-u_parallel
    relation at the same local B:

        μ1 = (m_e c^2 / (2 B)) (γ1^2 - γ_parallel^2) / γ1.

    Notes
    -----
    - This is exact for a uniform-B local radiation step.
    - In the full track integrator, B is taken at the segment midpoint, giving a midpoint-accurate update without sacrificing the compact adaptive axial grid.
    """
    g0 = np.asarray(gamma0, dtype=float)
    mu0 = np.asarray(mu0_J_per_T, dtype=float)
    B = np.asarray(B_T, dtype=float)
    dt = np.asarray(dt_s, dtype=float)

    g0 = np.maximum(g0, 1.0)
    mu0 = np.maximum(mu0, 0.0)
    B_safe = np.maximum(B, 0.0)
    dt_pos = np.maximum(dt, 0.0)

    upar2 = parallel_u2_from_B_gamma_mu(B_safe, g0, mu0)
    gamma_par = np.sqrt(1.0 + upar2)

    eta = float(energy_loss_scale) * radiative_drag_eta_per_s(B_safe, q_C=q_C)
    r0 = np.divide(g0 - gamma_par, np.maximum(g0 + gamma_par, 1e-30))
    r0 = np.clip(r0, 0.0, 1.0 - 1.0e-15)
    r1 = r0 * np.exp(-2.0 * eta * gamma_par * dt_pos)
    r1 = np.clip(r1, 0.0, 1.0 - 1.0e-15)

    g1 = gamma_par * np.divide(1.0 + r1, np.maximum(1.0 - r1, 1e-30))
    g1 = np.maximum(g1, np.maximum(float(gamma_floor), gamma_par))

    uperp2 = np.maximum(g1 * g1 - gamma_par * gamma_par, 0.0)
    mu1 = np.where(
        B_safe > 0.0,
        0.5 * M_E * (C0**2) * np.divide(uperp2, np.maximum(B_safe * g1, 1e-30)),
        mu0,
    )
    mu1 = np.maximum(mu1, 0.0)

    return np.asarray(g1, dtype=float), np.asarray(mu1, dtype=float)

