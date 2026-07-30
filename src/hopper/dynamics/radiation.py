"""
Module: hopper.dynamics.radiation

Developer: ehtkarim
Date: April 29, 2026

Advances coupled electron and cavity state with compact or vectorized radiative back-reaction paths.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .. import constants as const
from ..cavity.interaction import CavityInteraction
from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..field.field_map import FieldMap
from .axial_profile import AxialFieldProfile
from .axial_solver import AxialSolver
from .drifts import curvature_drift_vphi, gradB_drift_vphi
from .kinematics import (
    beta_parallel2_from_B_gamma_mu,
    cyclotron_frequency_hz,
    gamma_mu_after_power_loss_step_fixed_upar,
    gamma_mu_after_radiation_step_fixed_upar,
    gamma_mu_after_signed_power_work_step_fixed_upar,
    kinetic_energy_eV_from_gamma,
)
from .template import build_bounce_template

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scalar / local helpers
# -----------------------------------------------------------------------------


def _gamma_from_energy_eV(energy_eV: float) -> float:
    return float(1.0 + float(energy_eV) / const.MEC2_EV)


def _energy_eV_from_gamma(gamma: float) -> float:
    return float(np.asarray(kinetic_energy_eV_from_gamma(float(gamma))).reshape(()))


def _vpar_abs_from_B_gamma_mu(B_T: float, gamma: float, mu_J_per_T: float) -> float:
    beta_par2 = float(np.asarray(beta_parallel2_from_B_gamma_mu(float(B_T), float(gamma), float(mu_J_per_T))).reshape(()))
    return const.C0 * float(np.sqrt(max(beta_par2, 0.0)))


def _gamma_floor_from_energy_floor(energy_floor_eV: float) -> float:
    return max(1.0, _gamma_from_energy_eV(max(float(energy_floor_eV), 0.0)))


def _B_along(field: FieldMap, axial_profile: Optional[AxialFieldProfile], r0_m: float, z_m: np.ndarray | float) -> np.ndarray:
    if axial_profile is not None:
        return axial_profile.B(z_m)
    return field.B(float(r0_m), z_m)


def _bz_over_B(axial_profile: Optional[AxialFieldProfile], z_m: np.ndarray | float) -> np.ndarray:
    if axial_profile is not None:
        return axial_profile.bz_over_B(z_m)
    return np.ones_like(np.asarray(z_m, dtype=float), dtype=float)


def _r_along(axial_profile: Optional[AxialFieldProfile], r0_m: float, z_m: np.ndarray | float) -> np.ndarray:
    if axial_profile is not None:
        return axial_profile.r_at_z(z_m)
    return np.full_like(np.asarray(z_m, dtype=float), float(r0_m), dtype=float)


def _local_perp_basis_phi0(Br_T: float, Bphi_T: float, Bz_T: float) -> tuple[np.ndarray, np.ndarray]:
    """Local perpendicular basis at phi=0 used by compact scalar radiation updates."""
    e_r = np.asarray([1.0, 0.0, 0.0])
    e_phi = np.asarray([0.0, 1.0, 0.0])
    e_z = np.asarray([0.0, 0.0, 1.0])
    B_vec = float(Br_T) * e_r + float(Bphi_T) * e_phi + float(Bz_T) * e_z
    B_mag = max(float(np.linalg.norm(B_vec)), 1.0e-300)
    b = B_vec / B_mag
    u1 = np.cross(e_phi, b)
    norm = float(np.linalg.norm(u1))
    if norm < 1.0e-14:
        u1 = e_r.copy()
        norm = 1.0
    u1 = u1 / norm
    u2 = np.cross(b, u1)
    u2 = u2 / max(float(np.linalg.norm(u2)), 1.0e-300)
    return u1, u2


def _local_perp_basis_cylindrical(
    phi_rad: np.ndarray | float,
    Br_T: np.ndarray | float,
    Bphi_T: np.ndarray | float,
    Bz_T: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized local perpendicular basis matching track reconstruction."""
    phi = np.asarray(phi_rad, dtype=float)
    Br = np.asarray(Br_T, dtype=float)
    Bphi = np.asarray(Bphi_T, dtype=float)
    Bz = np.asarray(Bz_T, dtype=float)
    shape = np.broadcast_shapes(phi.shape, Br.shape, Bphi.shape, Bz.shape)
    phi = np.broadcast_to(phi, shape)
    Br = np.broadcast_to(Br, shape)
    Bphi = np.broadcast_to(Bphi, shape)
    Bz = np.broadcast_to(Bz, shape)

    c = np.cos(phi)
    s = np.sin(phi)
    zeros = np.zeros_like(c)
    ones = np.ones_like(c)
    e_r = np.stack([c, s, zeros], axis=-1)
    e_phi = np.stack([-s, c, zeros], axis=-1)
    e_z = np.stack([zeros, zeros, ones], axis=-1)
    B_vec = Br[..., None] * e_r + Bphi[..., None] * e_phi + Bz[..., None] * e_z
    b = B_vec / np.maximum(np.linalg.norm(B_vec, axis=-1)[..., None], 1.0e-300)
    u1 = np.cross(e_phi, b)
    u1_norm = np.linalg.norm(u1, axis=-1)
    fallback = np.cross(e_z, b)
    fallback_norm = np.linalg.norm(fallback, axis=-1)
    use_fallback = u1_norm < 1.0e-14
    u1 = np.where(use_fallback[..., None], fallback, u1)
    u1_norm = np.where(use_fallback, fallback_norm, u1_norm)
    use_er = u1_norm < 1.0e-14
    u1 = np.where(use_er[..., None], e_r, u1)
    u1_norm = np.where(use_er, np.linalg.norm(e_r, axis=-1), u1_norm)
    u1 = u1 / np.maximum(u1_norm[..., None], 1.0e-300)
    u2 = np.cross(b, u1)
    u2 = u2 / np.maximum(np.linalg.norm(u2, axis=-1)[..., None], 1.0e-300)
    return u1, u2


def _local_dphi_dt(
    *,
    z_m: float,
    gamma: float,
    mu_J_per_T: float,
    parallel_sign: float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    include_gradB: bool,
    include_curvature_drift: bool,
) -> float:
    if axial_profile is None:
        return 0.0
    r_val = float(np.asarray(axial_profile.r_at_z(float(z_m))).reshape(()))
    if abs(r_val) < 1.0e-14:
        return 0.0
    B_val = float(np.asarray(axial_profile.B(float(z_m))).reshape(()))
    Br_val, Bphi_val, Bz_val = axial_profile.components(float(z_m))
    Br_val = float(np.asarray(Br_val).reshape(()))
    Bphi_val = float(np.asarray(Bphi_val).reshape(()))
    Bz_val = float(np.asarray(Bz_val).reshape(()))
    vpar_abs = _vpar_abs_from_B_gamma_mu(B_val, float(gamma), float(mu_J_per_T))
    vphi = float(parallel_sign) * vpar_abs * Bphi_val / max(B_val, 1.0e-300)
    if bool(include_gradB):
        dBdr_val, dBdz_val = axial_profile.gradB(float(z_m))
        vphi += float(np.asarray(gradB_drift_vphi(
            float(mu_J_per_T),
            q_C=-const.E_CHARGE,
            Bmag_T=B_val,
            Br_T=Br_val,
            Bz_T=Bz_val,
            dBdr_T_per_m=float(np.asarray(dBdr_val).reshape(())),
            dBdz_T_per_m=float(np.asarray(dBdz_val).reshape(())),
            gamma=float(gamma),
        )).reshape(()))
    if bool(include_curvature_drift):
        kappa_phi = float(np.asarray(axial_profile.b_cross_kappa_phi(float(z_m))).reshape(()))
        vphi += float(np.asarray(curvature_drift_vphi(
            float(gamma),
            vpar_abs,
            q_C=-const.E_CHARGE,
            Bmag_T=B_val,
            b_cross_kappa_phi_per_m=kappa_phi,
        )).reshape(()))
    return vphi / r_val


def _cavity_source_power_and_response(
    *,
    B_T: float,
    gamma: float,
    mu_J_per_T: float,
    z_m: float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    mode_map: Optional[ModeMap],
    resonance: Optional[ResonanceCurve],
    cavity_interaction: CavityInteraction,
) -> tuple[float, float]:
    """Return source power and resonance response for compact cavity evolution."""
    r_mid = float(np.asarray(_r_along(axial_profile, float(r0_m), float(z_m))).reshape(()))
    fc_mid = float(np.asarray(cyclotron_frequency_hz(float(B_T), float(gamma))).reshape(()))
    response = 1.0 if resonance is None else float(np.asarray(resonance(fc_mid)).reshape(()))

    if mode_map is not None and bool(getattr(mode_map, "is_vector_e_field", False)):
        if axial_profile is not None:
            Br_mid, Bphi_mid, Bz_mid = axial_profile.components(float(z_m))
            Br_mid = float(np.asarray(Br_mid).reshape(()))
            Bphi_mid = float(np.asarray(Bphi_mid).reshape(()))
            Bz_mid = float(np.asarray(Bz_mid).reshape(()))
        else:
            Br_mid, Bphi_mid, Bz_mid = (0.0, 0.0, float(B_T))
        u1, u2 = _local_perp_basis_phi0(Br_mid, Bphi_mid, Bz_mid)
        drive = mode_map.gyro_drive_coupling_W_per_sqrt_J(  # type: ignore[attr-defined]
            r_gc_m=np.asarray([r_mid], dtype=float),
            phi_gc_rad=np.asarray([0.0], dtype=float),
            z_gc_m=np.asarray([float(z_m)], dtype=float),
            B_T=np.asarray([float(B_T)], dtype=float),
            gamma=np.asarray([float(gamma)], dtype=float),
            mu_J_per_T=np.asarray([float(mu_J_per_T)], dtype=float),
            u1=np.asarray([u1], dtype=float),
            u2=np.asarray([u2], dtype=float),
        )
        source_power = cavity_interaction.source_power_from_drive_W(drive, response)
        return float(np.asarray(source_power).reshape(())), float(response)

    # Analytic/scalar mode maps are not used for compact stimulated back-reaction.
    # They can still drive the coherent signal fallback in signal.synth, but using a scalar
    # power proxy here would reintroduce the old Larmor/Purcell approximation.
    return 0.0, float(response)



def _complex_back_reaction_available(
    cavity_interaction: Optional[CavityInteraction],
    mode_map: Optional[ModeMap],
) -> bool:
    return bool(
        cavity_interaction is not None
        and cavity_interaction.coherent_back_reaction_enabled
        and mode_map is not None
        and bool(getattr(mode_map, "is_vector_e_field", False))
    )


def _cavity_amplitude_from_energy(
    cavity_interaction: Optional[CavityInteraction],
    cavity_energy_J: float,
) -> complex:
    if cavity_interaction is not None:
        if float(cavity_energy_J) == float(cavity_interaction.initial_energy_J):
            return cavity_interaction.initial_amplitude_sqrt_J
        phase = float(getattr(cavity_interaction, "initial_phase_rad", 0.0))
    else:
        phase = 0.0
    return complex(np.sqrt(max(float(cavity_energy_J), 0.0)) * np.exp(1j * phase))


def _one_pole_zoh_update(amplitude: complex, drive: complex, lambda_per_s: complex, dt_s: float) -> complex:
    h = max(float(dt_s), 0.0)
    if h == 0.0:
        return complex(amplitude)
    lam = complex(lambda_per_s)
    if abs(lam) <= 0.0:
        return complex(amplitude + drive * h)
    e = np.exp(-lam * h)
    return complex(e * amplitude + ((1.0 - e) / lam) * drive)


def _complex_cavity_drive_block(
    *,
    B_T: np.ndarray | float,
    gamma: np.ndarray | float,
    mu_J_per_T: np.ndarray | float,
    z_m: np.ndarray | float,
    phi_gc_rad: np.ndarray | float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    mode_map: ModeMap,
    cavity_interaction: CavityInteraction,
    phase_rf_rad: np.ndarray | float,
    t_s: np.ndarray | float,
) -> np.ndarray:
    """Vectorized analytic baseband drive on compact midpoint arrays."""
    z = np.asarray(z_m, dtype=float)
    B = np.asarray(B_T, dtype=float)
    g = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    phi = np.asarray(phi_gc_rad, dtype=float)
    phase_rf = np.asarray(phase_rf_rad, dtype=float)
    t = np.asarray(t_s, dtype=float)
    shape = np.broadcast_shapes(z.shape, B.shape, g.shape, mu.shape, phi.shape, phase_rf.shape, t.shape)
    z = np.broadcast_to(z, shape)
    B = np.broadcast_to(B, shape)
    g = np.broadcast_to(g, shape)
    mu = np.broadcast_to(mu, shape)
    phi = np.broadcast_to(phi, shape)
    phase_rf = np.broadcast_to(phase_rf, shape)
    t = np.broadcast_to(t, shape)

    r_mid = np.asarray(_r_along(axial_profile, float(r0_m), z), dtype=float)
    if axial_profile is not None:
        Br_mid, Bphi_mid, Bz_mid = axial_profile.components(z)
        Br_mid = np.asarray(Br_mid, dtype=float)
        Bphi_mid = np.asarray(Bphi_mid, dtype=float)
        Bz_mid = np.asarray(Bz_mid, dtype=float)
    else:
        Br_mid = np.zeros(shape, dtype=float)
        Bphi_mid = np.zeros(shape, dtype=float)
        Bz_mid = np.asarray(B, dtype=float)
    u1, u2 = _local_perp_basis_cylindrical(phi, Br_mid, Bphi_mid, Bz_mid)
    coeff = mode_map.gyro_drive_coupling_W_per_sqrt_J(  # type: ignore[attr-defined]
        r_gc_m=np.asarray(r_mid, dtype=float),
        phi_gc_rad=np.asarray(phi, dtype=float),
        z_gc_m=np.asarray(z, dtype=float),
        B_T=np.asarray(B, dtype=float),
        gamma=np.asarray(g, dtype=float),
        mu_J_per_T=np.asarray(mu, dtype=float),
        u1=np.asarray(u1, dtype=float),
        u2=np.asarray(u2, dtype=float),
    )
    phase = phase_rf + float(cavity_interaction.cyclotron_phase0_rad) - cavity_interaction.omega_lo_rad_per_s * t
    scale = np.sqrt(max(float(cavity_interaction.source_power_scale), 0.0))
    return np.asarray(scale * coeff * np.exp(1j * phase), dtype=np.complex128)


def _complex_cavity_drive_midpoint(
    *,
    B_T: float,
    gamma: float,
    mu_J_per_T: float,
    z_m: float,
    phi_gc_rad: float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    mode_map: ModeMap,
    cavity_interaction: CavityInteraction,
    phase_rf_rad: float,
    t_s: float,
) -> complex:
    """Vector-map analytic baseband drive at one compact-grid midpoint."""
    drive = _complex_cavity_drive_block(
        B_T=float(B_T),
        gamma=float(gamma),
        mu_J_per_T=float(mu_J_per_T),
        z_m=float(z_m),
        phi_gc_rad=float(phi_gc_rad),
        r0_m=float(r0_m),
        axial_profile=axial_profile,
        mode_map=mode_map,
        cavity_interaction=cavity_interaction,
        phase_rf_rad=float(phase_rf_rad),
        t_s=float(t_s),
    )
    return complex(np.asarray(drive).reshape(()))


def _radiative_state_step(
    *,
    gamma: float,
    mu_J_per_T: float,
    B_T: float,
    dt_s: float,
    z_m: float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    mode_map: Optional[ModeMap],
    resonance: Optional[ResonanceCurve],
    cavity_interaction: Optional[CavityInteraction],
    cavity_amplitude_sqrt_J: complex,
    phase_rf_rad: float,
    t_s: float,
    energy_loss_scale: float,
    gamma_floor: float,
    phi_gc_rad: float = 0.0,
    drive_mid_sqrt_J_per_s: complex | None = None,
    stats: Optional[dict[str, int]] = None,
) -> tuple[float, float, complex, float, float]:
    """Advance electron radiative state plus complex cavity amplitude for one local step.

    The complex back-reaction branch uses the same time-evolution operator as signal
    synthesis.  The cavity drive is evaluated at the segment midpoint on the compact
    field-line grid.  The signed work rate is

        P_work = 2 Re[a_mid^* d_mid],

    positive when the electron transfers energy to the cavity and negative when the
    stored field accelerates the electron.  The electron update holds u_parallel fixed,
    so the work changes the perpendicular action and magnetic moment consistently.
    """
    if stats is not None:
        stats["n_radiative_state_steps"] = int(stats.get("n_radiative_state_steps", 0)) + 1
    if float(dt_s) <= 0.0:
        return float(gamma), float(mu_J_per_T), complex(cavity_amplitude_sqrt_J), float(phase_rf_rad), 0.0

    fc_mid = float(np.asarray(cyclotron_frequency_hz(float(B_T), float(gamma))).reshape(()))
    phase_mid = float(phase_rf_rad) + np.pi * 2.0 * fc_mid * float(dt_s) * 0.5
    phase_next = float(phase_rf_rad) + 2.0 * np.pi * fc_mid * float(dt_s)

    if _complex_back_reaction_available(cavity_interaction, mode_map):
        assert cavity_interaction is not None and mode_map is not None
        if drive_mid_sqrt_J_per_s is None:
            drive_mid = _complex_cavity_drive_midpoint(
                B_T=float(B_T),
                gamma=float(gamma),
                mu_J_per_T=float(mu_J_per_T),
                z_m=float(z_m),
                phi_gc_rad=float(phi_gc_rad),
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                mode_map=mode_map,
                cavity_interaction=cavity_interaction,
                phase_rf_rad=phase_mid,
                t_s=float(t_s) + 0.5 * float(dt_s),
            )
            if stats is not None:
                stats["n_scalar_drive_midpoint_calls"] = int(stats.get("n_scalar_drive_midpoint_calls", 0)) + 1
        else:
            drive_mid = complex(drive_mid_sqrt_J_per_s)
            if stats is not None:
                stats["n_precomputed_drive_midpoints_used"] = int(stats.get("n_precomputed_drive_midpoints_used", 0)) + 1
        a0 = complex(cavity_amplitude_sqrt_J)
        half = 0.5 * float(dt_s)
        a_mid = _one_pole_zoh_update(a0, drive_mid, cavity_interaction.lambda_per_s, half)
        work_power = 2.0 * float(np.real(np.conjugate(a_mid) * drive_mid))
        signed_loss_power = float(energy_loss_scale) * float(cavity_interaction.back_reaction_scale) * work_power
        gamma_trial, mu_trial = gamma_mu_after_signed_power_work_step_fixed_upar(
            float(gamma),
            float(mu_J_per_T),
            float(B_T),
            float(dt_s),
            signed_loss_power,
            gamma_floor=float(gamma_floor),
        )
        a1 = _one_pole_zoh_update(a0, drive_mid, cavity_interaction.lambda_per_s, float(dt_s))
        return (
            float(np.asarray(gamma_trial).reshape(())),
            float(np.asarray(mu_trial).reshape(())),
            complex(a1),
            float(phase_next),
            float(work_power),
        )

    gamma_trial, mu_trial, cavity_energy_trial = _radiative_step_with_optional_cavity(
        gamma=float(gamma),
        mu_J_per_T=float(mu_J_per_T),
        B_T=float(B_T),
        dt_s=float(dt_s),
        z_m=float(z_m),
        r0_m=float(r0_m),
        axial_profile=axial_profile,
        mode_map=mode_map,
        resonance=resonance,
        cavity_interaction=cavity_interaction,
        cavity_energy_J=float(abs(cavity_amplitude_sqrt_J) ** 2),
        energy_loss_scale=float(energy_loss_scale),
        gamma_floor=float(gamma_floor),
    )
    amp = _cavity_amplitude_from_energy(cavity_interaction, cavity_energy_trial)
    return float(gamma_trial), float(mu_trial), complex(amp), float(phase_next), 0.0

def _radiative_step_with_optional_cavity(
    *,
    gamma: float,
    mu_J_per_T: float,
    B_T: float,
    dt_s: float,
    z_m: float,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    mode_map: Optional[ModeMap],
    resonance: Optional[ResonanceCurve],
    cavity_interaction: Optional[CavityInteraction],
    cavity_energy_J: float,
    energy_loss_scale: float,
    gamma_floor: float,
) -> tuple[float, float, float]:
    if float(dt_s) <= 0.0:
        return float(gamma), float(mu_J_per_T), float(cavity_energy_J)

    use_vector_cavity = (
        cavity_interaction is not None
        and cavity_interaction.excitation_enabled
        and mode_map is not None
        and bool(getattr(mode_map, "is_vector_e_field", False))
    )
    if use_vector_cavity:
        source_power, _response = _cavity_source_power_and_response(
            B_T=float(B_T),
            gamma=float(gamma),
            mu_J_per_T=float(mu_J_per_T),
            z_m=float(z_m),
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mode_map=mode_map,
            resonance=resonance,
            cavity_interaction=cavity_interaction,
        )
        field_work_power = cavity_interaction.field_work_power_W(source_power, float(cavity_energy_J))
        back_power = cavity_interaction.back_reaction_power_W(source_power, float(cavity_energy_J))
        gamma_trial, mu_trial = gamma_mu_after_power_loss_step_fixed_upar(
            float(gamma),
            float(mu_J_per_T),
            float(B_T),
            float(dt_s),
            float(energy_loss_scale) * float(np.asarray(back_power).reshape(())),
            gamma_floor=float(gamma_floor),
        )
        cavity_trial = cavity_interaction.advance_stored_energy_J(
            float(cavity_energy_J),
            float(np.asarray(field_work_power).reshape(())),
            float(dt_s),
        )
        return (
            float(np.asarray(gamma_trial).reshape(())),
            float(np.asarray(mu_trial).reshape(())),
            float(np.asarray(cavity_trial).reshape(())),
        )

    gamma_trial, mu_trial = gamma_mu_after_radiation_step_fixed_upar(
        float(gamma),
        float(mu_J_per_T),
        float(B_T),
        float(dt_s),
        energy_loss_scale=float(energy_loss_scale),
        gamma_floor=float(gamma_floor),
    )
    return (
        float(np.asarray(gamma_trial).reshape(())),
        float(np.asarray(mu_trial).reshape(())),
        float(cavity_energy_J),
    )


# -----------------------------------------------------------------------------
# Bounce-return detection
# -----------------------------------------------------------------------------


def _sign_no_zeros(x: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(x, dtype=float)).astype(int)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    if np.all(s == 0):
        s[:] = 1
    return s


def _reflection_indices(vpar: np.ndarray, *, min_dz_m: float, z_m: np.ndarray | None = None) -> np.ndarray:
    s = _sign_no_zeros(vpar)
    flips = np.flatnonzero(s[1:] != s[:-1]) + 1
    if z_m is None or min_dz_m <= 0.0 or len(flips) <= 1:
        return np.asarray(flips, dtype=int)

    z = np.asarray(z_m, dtype=float)
    keep: list[int] = [int(flips[0])]
    last = int(flips[0])
    for f in flips[1:]:
        if abs(float(z[int(f)]) - float(z[last])) >= float(min_dz_m):
            keep.append(int(f))
            last = int(f)
    return np.asarray(keep, dtype=int)


def _truncate_track_at_bounce_return(
    *,
    t_s: np.ndarray,
    z_m: np.ndarray,
    vz_m_per_s: np.ndarray,
    energy_eV: np.ndarray,
    mu_J_per_T: np.ndarray,
    cavity_energy_J: np.ndarray,
    z0_m: float,
    vpar_sign0: int,
    z_tol_m: float,
    min_reflections: int,
    n_bounce_returns: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Truncate a compact radiative track at the first full-bounce return to z0.

    A full bounce is a return to z0 with the original z-velocity sign after at least
    ``min_reflections`` genuine mirror reflections. The endpoint is linearly interpolated
    on the z(t) crossing, which avoids accumulating a phase bias from simply snapping to
    the nearest adaptive node. ``n_bounce_returns`` counts how many such returns to traverse
    before truncating.
    """
    t = np.asarray(t_s, dtype=float)
    z = np.asarray(z_m, dtype=float)
    v = np.asarray(vz_m_per_s, dtype=float)
    E = np.asarray(energy_eV, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    U = np.asarray(cavity_energy_J, dtype=float)
    if not (t.ndim == z.ndim == v.ndim == E.ndim == mu.ndim == U.ndim == 1):
        raise ValueError("Compact radiative track arrays must be 1D.")
    if not (t.size == z.size == v.size == E.size == mu.size == U.size):
        raise ValueError("Compact radiative track arrays must have equal length.")
    if t.size < 3:
        return None
    if int(n_bounce_returns) < 1:
        raise ValueError("n_bounce_returns must be >= 1")

    s = _sign_no_zeros(v)
    s0 = 1 if int(vpar_sign0) >= 0 else -1
    dz0 = z - float(z0_m)
    excursion_needed = max(10.0 * float(z_tol_m), 1.0e-9)
    flips = _reflection_indices(v, min_dz_m=excursion_needed, z_m=z)
    if len(flips) < int(min_reflections):
        return None
    start_idx = int(flips[int(min_reflections) - 1])
    if np.max(np.abs(dz0[: start_idx + 1])) < excursion_needed:
        return None

    tol = max(float(z_tol_m), 1.0e-12)
    n_found = 0
    for i in range(start_idx, t.size - 1):
        left = float(dz0[i])
        right = float(dz0[i + 1])

        if abs(left) <= tol and s[i] == s0:
            n_found += 1
            if n_found >= int(n_bounce_returns):
                end = i + 1
                return t[:end], z[:end], v[:end], E[:end], mu[:end], U[:end]

        crosses = (left == 0.0) or (right == 0.0) or (left * right < 0.0)
        if not crosses:
            continue

        denom = right - left
        if abs(denom) <= 1.0e-30:
            frac = 0.0
        else:
            frac = float(np.clip(-left / denom, 0.0, 1.0))

        v_cross = float(v[i] + frac * (v[i + 1] - v[i]))
        s_cross = s0 if abs(v_cross) <= 1.0e-14 else (1 if v_cross > 0.0 else -1)
        if s_cross != s0:
            continue

        n_found += 1
        if n_found < int(n_bounce_returns):
            continue

        t_cross = float(t[i] + frac * (t[i + 1] - t[i]))
        z_cross = float(z[i] + frac * (z[i + 1] - z[i]))
        E_cross = float(E[i] + frac * (E[i + 1] - E[i]))
        mu_cross = float(mu[i] + frac * (mu[i + 1] - mu[i]))
        U_cross = float(U[i] + frac * (U[i + 1] - U[i]))

        t_out = np.concatenate([t[: i + 1], np.asarray([t_cross], dtype=float)])
        z_out = np.concatenate([z[: i + 1], np.asarray([z_cross], dtype=float)])
        v_out = np.concatenate([v[: i + 1], np.asarray([v_cross], dtype=float)])
        E_out = np.concatenate([E[: i + 1], np.asarray([E_cross], dtype=float)])
        mu_out = np.concatenate([mu[: i + 1], np.asarray([mu_cross], dtype=float)])
        U_out = np.concatenate([U[: i + 1], np.asarray([U_cross], dtype=float)])
        keep = np.concatenate([[True], np.diff(t_out) > 0.0])
        return t_out[keep], z_out[keep], v_out[keep], E_out[keep], mu_out[keep], U_out[keep]

    if abs(float(dz0[-1])) <= tol and s[-1] == s0:
        n_found += 1
    if n_found >= int(n_bounce_returns):
        return t, z, v, E, mu, U
    return None


# -----------------------------------------------------------------------------
# Continuous radiative axial integration
# -----------------------------------------------------------------------------


def integrate_axial_track_energy_analytic(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile] = None,
    mu0_J_per_T: float = 0.0,
    t0_s: float,
    duration_s: float,
    z0_m: float,
    vpar_sign: int,
    energy0_eV: float,
    energy_floor_eV: float,
    energy_loss_scale: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    cavity_amplitude0_sqrt_J: complex | None = None,
    phase_rf0_rad: float = 0.0,
    phi_gc0_rad: float = 0.0,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    stats: Optional[dict[str, int]] = None,
    return_aux: bool = False,
    max_steps: int = 5_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct adaptive axial integration with continuous radiative evolution of γ and μ.

    Unlike the earlier implementation, turning points are now located by bisection in
    *time* for the coupled trial state. The accepted (t, z, γ, μ) state at the mirror is
    therefore self-consistent: the time advance, radiative loss, and mirror location all
    correspond to the same shortened step. This removes the previous bias where the code
    could advance the radiative state for a full step and only then project the spatial
    point back toward the mirror.
    """
    t0 = float(t0_s)
    t_end = t0 + float(duration_s)
    if float(duration_s) <= 0.0:
        return (
            np.asarray([t0], dtype=float),
            np.asarray([float(z0_m)], dtype=float),
            np.asarray([0.0], dtype=float),
            np.asarray([float(energy0_eV)], dtype=float),
            np.asarray([float(mu0_J_per_T)], dtype=float),
            np.asarray([max(float(cavity_energy0_J), 0.0)], dtype=float),
        )

    gamma_floor = _gamma_floor_from_energy_floor(float(energy_floor_eV))
    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), gamma_floor)
    mu = max(float(mu0_J_per_T), 0.0)
    cavity_energy = max(float(cavity_energy0_J), 0.0)
    cavity_amplitude = (
        _cavity_amplitude_from_energy(cavity_interaction, cavity_energy)
        if cavity_amplitude0_sqrt_J is None
        else complex(cavity_amplitude0_sqrt_J)
    )
    phase_rf = float(phase_rf0_rad)
    phi_gc = float(phi_gc0_rad)
    sign = 1.0 if int(vpar_sign) >= 0 else -1.0
    z = float(z0_m)
    t = t0

    z_min = float(field.z[0])
    z_max = float(field.z[-1])
    c_turn = max(float(v_turn_threshold_c), 0.0) * const.C0
    turn_tol = 1.0e-14

    B0 = float(_B_along(field, axial_profile, float(r0_m), z))
    beta0 = float(np.asarray(beta_parallel2_from_B_gamma_mu(B0, gamma, mu)).reshape(()))
    if beta0 < -1.0e-12:
        raise ValueError("Initial state is outside the mirror condition for the supplied (E0, μ0).")

    vpar0_abs = _vpar_abs_from_B_gamma_mu(B0, gamma, mu)
    bz_over_B0 = float(np.asarray(_bz_over_B(axial_profile, z)).reshape(()))

    t_list = [t]
    z_list = [z]
    v_list = [sign * vpar0_abs * bz_over_B0]
    E_list = [_energy_eV_from_gamma(gamma)]
    mu_list = [mu]
    cavity_energy_list = [cavity_energy]
    phase_rf_list = [phase_rf]
    phi_gc_list = [phi_gc]
    cavity_amplitude_list = [cavity_amplitude]

    for _ in range(int(max_steps)):
        if t >= t_end - 1.0e-24:
            break

        B_here = float(_B_along(field, axial_profile, float(r0_m), z))
        beta_par2_here = float(np.asarray(beta_parallel2_from_B_gamma_mu(B_here, gamma, mu)).reshape(()))
        if beta_par2_here < -1.0e-12:
            raise RuntimeError("Radiative axial integrator drifted outside the mirror condition.")

        vpar_abs = const.C0 * float(np.sqrt(max(beta_par2_here, 0.0)))
        if vpar_abs < c_turn and c_turn > 0.0:
            dt = float(safety) * float(dt_max_s) * (vpar_abs / (c_turn + 1.0e-300))
            dt = max(float(dt_min_s), dt)
        else:
            dt = float(dt_max_s)
        dt = min(max(float(dt_min_s), dt), t_end - t)

        z_speed_factor_here = float(np.asarray(_bz_over_B(axial_profile, z)).reshape(()))

        if vpar_abs <= 0.0 or beta_par2_here <= turn_tol:
            # Move an infinitesimal distance inward along the reflected branch so the next
            # accepted step starts from a strictly allowed state.
            nudge = max(1.0e-12, 1.0e-9 * max(abs(z), 1.0))
            z_dir = np.sign(sign * z_speed_factor_here) if abs(z_speed_factor_here) > 0.0 else sign
            z_candidate = float(np.clip(z + z_dir * nudge, z_min, z_max))
            beta_candidate = float(
                np.asarray(beta_parallel2_from_B_gamma_mu(float(_B_along(field, axial_profile, float(r0_m), z_candidate)), gamma, mu)).reshape(())
            )
            if beta_candidate > 0.0:
                z = z_candidate
                continue

        dphi_dt_here = _local_dphi_dt(
            z_m=float(z),
            gamma=float(gamma),
            mu_J_per_T=float(mu),
            parallel_sign=float(sign),
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            include_gradB=bool(include_gradB),
            include_curvature_drift=bool(include_curvature_drift),
        )

        def trial_state(dt_trial: float) -> tuple[float, float, float, complex, float, float, float]:
            z_trial = z + sign * vpar_abs * z_speed_factor_here * float(dt_trial)
            z_mid = 0.5 * (z + z_trial)
            phi_mid = phi_gc + 0.5 * dphi_dt_here * float(dt_trial)
            phi_trial = phi_gc + dphi_dt_here * float(dt_trial)
            B_mid = float(_B_along(field, axial_profile, float(r0_m), z_mid))

            if gamma <= gamma_floor * (1.0 + 1.0e-14) and not _complex_back_reaction_available(cavity_interaction, mode_map):
                gamma_trial = gamma
                mu_trial = mu
                cavity_amplitude_trial = cavity_amplitude
                phase_trial = phase_rf + 2.0 * np.pi * float(np.asarray(cyclotron_frequency_hz(B_mid, gamma)).reshape(())) * float(dt_trial)
            elif float(energy_loss_scale) == 0.0 and not _complex_back_reaction_available(cavity_interaction, mode_map):
                gamma_trial = gamma
                mu_trial = mu
                cavity_amplitude_trial = cavity_amplitude
                phase_trial = phase_rf + 2.0 * np.pi * float(np.asarray(cyclotron_frequency_hz(B_mid, gamma)).reshape(())) * float(dt_trial)
            else:
                gamma_trial, mu_trial, cavity_amplitude_trial, phase_trial, _work = _radiative_state_step(
                    gamma=float(gamma),
                    mu_J_per_T=float(mu),
                    B_T=float(B_mid),
                    dt_s=float(dt_trial),
                    z_m=float(z_mid),
                    r0_m=float(r0_m),
                    axial_profile=axial_profile,
                    mode_map=mode_map,
                    resonance=resonance,
                    cavity_interaction=cavity_interaction,
                    cavity_amplitude_sqrt_J=complex(cavity_amplitude),
                    phase_rf_rad=float(phase_rf),
                    t_s=float(t),
                    energy_loss_scale=float(energy_loss_scale),
                    gamma_floor=float(gamma_floor),
                    phi_gc_rad=float(phi_mid),
                    stats=stats,
                )

            B_trial = float(_B_along(field, axial_profile, float(r0_m), z_trial))
            beta_par2_trial = float(np.asarray(beta_parallel2_from_B_gamma_mu(B_trial, gamma_trial, mu_trial)).reshape(()))
            return z_trial, gamma_trial, mu_trial, cavity_amplitude_trial, phase_trial, beta_par2_trial, phi_trial

        z_trial, gamma_trial, mu_trial, cavity_amplitude_trial, phase_trial, beta_par2_trial, phi_trial = trial_state(dt)

        if beta_par2_trial >= -turn_tol:
            turning_step = beta_par2_trial <= turn_tol
            t_new = t + dt
            z_new = float(z_trial)
            gamma_new = float(gamma_trial)
            mu_new = float(mu_trial)
            cavity_amplitude_new = complex(cavity_amplitude_trial)
            phase_rf_new = float(phase_trial)
            phi_gc_new = float(phi_trial)
            cavity_energy_new = float(abs(cavity_amplitude_new) ** 2)
        else:
            # The mirror lies inside the proposed interval. Locate the turning time with a
            # consistent time/gamma/mu bisection instead of advancing radiation for the full
            # dt and then projecting only z.
            dt_lo = 0.0
            z_lo = z
            gamma_lo = gamma
            mu_lo = mu
            cavity_amplitude_lo = cavity_amplitude
            phase_rf_lo = phase_rf
            phi_gc_lo = phi_gc
            for _ in range(80):
                dt_mid = 0.5 * (dt_lo + dt)
                z_mid, gamma_mid, mu_mid, cavity_amplitude_mid, phase_mid, beta_mid, phi_mid_trial = trial_state(dt_mid)
                if beta_mid >= 0.0:
                    dt_lo = dt_mid
                    z_lo = float(z_mid)
                    gamma_lo = float(gamma_mid)
                    mu_lo = float(mu_mid)
                    cavity_amplitude_lo = complex(cavity_amplitude_mid)
                    phase_rf_lo = float(phase_mid)
                    phi_gc_lo = float(phi_mid_trial)
                else:
                    dt = dt_mid

            if dt_lo <= 1.0e-24:
                sign *= -1.0
                nudge = max(1.0e-12, 1.0e-9 * max(abs(z), 1.0))
                z_dir = np.sign(sign * z_speed_factor_here) if abs(z_speed_factor_here) > 0.0 else sign
                z = float(np.clip(z + z_dir * nudge, z_min, z_max))
                continue

            turning_step = True
            t_new = t + dt_lo
            z_new = float(z_lo)
            gamma_new = float(gamma_lo)
            mu_new = float(mu_lo)
            cavity_amplitude_new = complex(cavity_amplitude_lo)
            phase_rf_new = float(phase_rf_lo)
            phi_gc_new = float(phi_gc_lo)
            cavity_energy_new = float(abs(cavity_amplitude_new) ** 2)

        if (z_new < z_min or z_new > z_max) and not field.clamp_to_grid:
            raise RuntimeError("Radiative axial integrator left the field-map z range.")

        if turning_step:
            sign *= -1.0

        B_new = float(_B_along(field, axial_profile, float(r0_m), z_new))
        v_new_abs = _vpar_abs_from_B_gamma_mu(B_new, gamma_new, mu_new)
        z_speed_factor_new = float(np.asarray(_bz_over_B(axial_profile, z_new)).reshape(()))
        v_new = sign * v_new_abs * z_speed_factor_new

        t = float(t_new)
        z = float(z_new)
        gamma = float(gamma_new)
        mu = float(mu_new)
        cavity_energy = float(cavity_energy_new)
        cavity_amplitude = complex(cavity_amplitude_new)
        phase_rf = float(phase_rf_new)
        phi_gc = float(phi_gc_new)

        t_list.append(t)
        z_list.append(z)
        v_list.append(v_new)
        E_list.append(_energy_eV_from_gamma(gamma))
        mu_list.append(mu)
        cavity_energy_list.append(cavity_energy)
        phase_rf_list.append(phase_rf)
        phi_gc_list.append(phi_gc)
        cavity_amplitude_list.append(cavity_amplitude)

    if t_list[-1] < t_end - 1.0e-18:
        raise RuntimeError(
            "Radiative axial integrator did not reach the requested end time. "
            "Increase max_steps or inspect the mirror condition / field-map bounds."
        )

    result = (
        np.asarray(t_list, dtype=float),
        np.asarray(z_list, dtype=float),
        np.asarray(v_list, dtype=float),
        np.asarray(E_list, dtype=float),
        np.asarray(mu_list, dtype=float),
        np.asarray(cavity_energy_list, dtype=float),
    )
    if return_aux:
        return result + (
            np.asarray(phase_rf_list, dtype=float),
            np.asarray(cavity_amplitude_list, dtype=np.complex128),
            np.asarray(phi_gc_list, dtype=float),
        )
    return result


# -----------------------------------------------------------------------------
# Bouncewise radiative integration
# -----------------------------------------------------------------------------


def _initial_bounce_period_guess(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    z0_m: float,
    vpar_sign: int,
    energy_eV: float,
    mu_J_per_T: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    template_build: str,
    template_return_z_tol_m: float,
    template_max_duration_s: float,
    template_min_reflections: int,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
    mirror_quadrature_min_theta_nodes: int = 513,
    mirror_quadrature_max_theta_nodes: int = 8193,
    mirror_template_max_period_rel_error: float = 1.0e-3,
) -> float:
    solver = AxialSolver(
        field=field,
        r0_m=float(r0_m),
        E0_eV=float(energy_eV),
        mu0_J_per_T=float(mu_J_per_T),
        dt_max_s=float(dt_max_s),
        dt_min_s=float(dt_min_s),
        safety=float(safety),
        v_turn_threshold_c=float(v_turn_threshold_c),
        axial_profile=axial_profile,
    )
    tpl = build_bounce_template(
        solver,
        z0_m=float(z0_m),
        vpar_sign0=int(vpar_sign),
        duration_hint_s=min(float(template_max_duration_s), 5.0e-5),
        max_duration_s=float(template_max_duration_s),
        return_z_tol_m=float(template_return_z_tol_m),
        min_reflections=int(template_min_reflections),
        method=str(template_build),
        mirror_z0_tol_m=float(mirror_z0_tol_m),
        mirror_symmetry_check=bool(mirror_symmetry_check),
        mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
        mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
        mirror_quadrature_min_theta_nodes=int(mirror_quadrature_min_theta_nodes),
        mirror_quadrature_max_theta_nodes=int(mirror_quadrature_max_theta_nodes),
        mirror_template_max_period_rel_error=float(mirror_template_max_period_rel_error),
    )
    return float(tpl.bounce_period_s)


def _integrate_one_radiative_bounce(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    gamma0: float,
    mu0_J_per_T: float,
    t0_s: float,
    z0_m: float,
    vpar_sign: int,
    energy_floor_eV: float,
    energy_loss_scale: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    bounce_period_guess_s: float,
    template_return_z_tol_m: float,
    template_min_reflections: int,
    template_max_duration_s: float,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    phase_rf0_rad: float = 0.0,
    phi_gc0_rad: float = 0.0,
    cavity_amplitude0_sqrt_J: complex | None = None,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    stats: Optional[dict[str, int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    guess = max(float(bounce_period_guess_s), 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))
    factors = (1.10, 1.35, 1.70, 2.20)
    energy0_eV = _energy_eV_from_gamma(float(gamma0))

    for fac in factors:
        duration = min(float(template_max_duration_s), fac * guess)
        seg = integrate_axial_track_energy_analytic(
            field=field,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mu0_J_per_T=float(mu0_J_per_T),
            t0_s=float(t0_s),
            duration_s=float(duration),
            z0_m=float(z0_m),
            vpar_sign=int(vpar_sign),
            energy0_eV=float(energy0_eV),
            energy_floor_eV=float(energy_floor_eV),
            energy_loss_scale=float(energy_loss_scale),
            dt_max_s=float(dt_max_s),
            dt_min_s=float(dt_min_s),
            safety=float(safety),
            v_turn_threshold_c=float(v_turn_threshold_c),
            mode_map=mode_map,
            resonance=resonance,
            cavity_interaction=cavity_interaction,
            cavity_energy0_J=float(cavity_energy0_J),
            cavity_amplitude0_sqrt_J=cavity_amplitude0_sqrt_J,
            phase_rf0_rad=float(phase_rf0_rad),
            phi_gc0_rad=float(phi_gc0_rad),
            include_gradB=bool(include_gradB),
            include_curvature_drift=bool(include_curvature_drift),
            stats=stats,
            return_aux=True,
        )
        t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_aux, amp_aux, phi_aux = seg
        truncated = _truncate_track_at_bounce_return(
            t_s=t_seg,
            z_m=z_seg,
            vz_m_per_s=v_seg,
            energy_eV=E_seg,
            mu_J_per_T=mu_seg,
            cavity_energy_J=U_seg,
            z0_m=float(z0_m),
            vpar_sign0=int(vpar_sign),
            z_tol_m=float(template_return_z_tol_m),
            min_reflections=int(template_min_reflections),
        )
        if truncated is not None:
            t_tr, z_tr, v_tr, E_tr, mu_tr, U_tr = truncated
            phase_tr = np.interp(t_tr, t_seg, phase_aux)
            amp_tr = np.interp(t_tr, t_seg, np.real(amp_aux)) + 1j * np.interp(t_tr, t_seg, np.imag(amp_aux))
            phi_tr = np.interp(t_tr, t_seg, phi_aux)
            return t_tr, z_tr, v_tr, E_tr, mu_tr, U_tr, phase_tr, amp_tr, phi_tr

    raise RuntimeError(
        "Failed to bracket a full radiative bounce return within the configured maximum duration. "
        "Increase dynamics.template_max_duration_s or inspect the mirror condition."
    )


def _integrate_template_periodic_radiative_block(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    gamma0: float,
    mu0_J_per_T: float,
    t0_s: float,
    z0_m: float,
    vpar_sign: int,
    energy_floor_eV: float,
    energy_loss_scale: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    block_bounces: int,
    template_build: str,
    template_return_z_tol_m: float,
    template_max_duration_s: float,
    template_min_reflections: int,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    phase_rf0_rad: float = 0.0,
    phi_gc0_rad: float = 0.0,
    cavity_amplitude0_sqrt_J: complex | None = None,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    back_reaction_block_vectorized: bool = True,
    back_reaction_block_max_rel_state_change: float = 5.0e-3,
    back_reaction_max_updates_per_bounce: int = 96,
    back_reaction_predictor_corrector: bool = True,
    mirror_quadrature_min_theta_nodes: int = 513,
    mirror_quadrature_max_theta_nodes: int = 8193,
    mirror_template_max_period_rel_error: float = 1.0e-3,
    mirror_template_max_phase_slip_rad: float = 5.0e-3,
    stats: Optional[dict[str, int]] = None,
    mirror_tol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Fast periodic-template radiative block for template_tiling/per_bounce runs.

    The spatial path is the symmetric bounce template at the block's initial state.
    The radiative state is advanced segment-by-segment on that template.  If the
    updated state makes a future template point kinematically forbidden, the helper
    returns ``None`` so the caller can fall back to the exact continuous compact
    solver for that block.  This keeps the speed of mirror templating when the state
    evolves slowly, without silently stepping beyond the moving mirror.
    """
    n_block = max(1, int(block_bounces))
    if stats is not None:
        stats["n_template_blocks_attempted"] = int(stats.get("n_template_blocks_attempted", 0)) + 1
    energy0_eV = _energy_eV_from_gamma(float(gamma0))
    solver = AxialSolver(
        field=field,
        r0_m=float(r0_m),
        E0_eV=float(energy0_eV),
        mu0_J_per_T=float(mu0_J_per_T),
        dt_max_s=float(dt_max_s),
        dt_min_s=float(dt_min_s),
        safety=float(safety),
        v_turn_threshold_c=float(v_turn_threshold_c),
        axial_profile=axial_profile,
    )
    try:
        tpl = build_bounce_template(
            solver,
            z0_m=float(z0_m),
            vpar_sign0=int(vpar_sign),
            duration_hint_s=min(float(template_max_duration_s), 5.0e-5),
            max_duration_s=float(template_max_duration_s),
            return_z_tol_m=float(template_return_z_tol_m),
            min_reflections=int(template_min_reflections),
            method=str(template_build),
            mirror_z0_tol_m=float(mirror_z0_tol_m),
            mirror_symmetry_check=bool(mirror_symmetry_check),
            mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
            mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
            mirror_quadrature_min_theta_nodes=int(mirror_quadrature_min_theta_nodes),
            mirror_quadrature_max_theta_nodes=int(mirror_quadrature_max_theta_nodes),
            mirror_template_max_period_rel_error=float(mirror_template_max_period_rel_error),
        )
    except Exception as exc:
        LOGGER.debug("Fast template radiative block unavailable; falling back to compact solver: %s", exc)
        return None

    if stats is not None:
        stats["last_template_method"] = str(getattr(tpl, "method", str(template_build)))
        stats["last_template_period_s"] = float(tpl.bounce_period_s)
        if getattr(tpl, "z_turn_positive_m", None) is not None:
            stats["last_template_z_turn_positive_m"] = float(tpl.z_turn_positive_m)
        if getattr(tpl, "theta_node_count", None) is not None:
            stats["last_template_theta_node_count"] = int(tpl.theta_node_count)
        if getattr(tpl, "period_rel_error_estimate", None) is not None:
            stats["last_template_period_rel_error_estimate"] = float(tpl.period_rel_error_estimate)

    if (
        getattr(tpl, "period_rel_error_estimate", None) is not None
        and float(tpl.period_rel_error_estimate) > float(mirror_template_max_period_rel_error)
    ):
        if stats is not None:
            stats["n_template_blocks_rejected_period_error"] = int(stats.get("n_template_blocks_rejected_period_error", 0)) + 1
        return None

    t_one = np.asarray(tpl.t_rel_s, dtype=float)
    z_one = np.asarray(tpl.z_m, dtype=float)
    v_one = np.asarray(tpl.vpar_ref_m_per_s, dtype=float)
    if t_one.size < 3 or float(tpl.bounce_period_s) <= 0.0:
        return None

    period = float(tpl.bounce_period_s)
    t_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    v_ref_parts: list[np.ndarray] = []
    for ib in range(n_block):
        sl = slice(None) if ib == 0 else slice(1, None)
        t_parts.append(float(t0_s) + ib * period + t_one[sl])
        z_parts.append(z_one[sl])
        v_ref_parts.append(v_one[sl])

    t_arr = np.concatenate(t_parts).astype(float)
    z_arr = np.concatenate(z_parts).astype(float)
    v_ref_arr = np.concatenate(v_ref_parts).astype(float)
    n = t_arr.size
    if n < 2 or not np.all(np.diff(t_arr) > 0.0):
        return None

    gamma_arr = np.empty(n, dtype=float)
    mu_arr = np.empty(n, dtype=float)
    energy_arr = np.empty(n, dtype=float)
    cavity_arr = np.empty(n, dtype=float)
    phase_arr = np.empty(n, dtype=float)
    phi_arr = np.empty(n, dtype=float)
    amplitude_arr = np.empty(n, dtype=np.complex128)
    v_arr = np.empty(n, dtype=float)

    gamma = max(float(gamma0), _gamma_floor_from_energy_floor(float(energy_floor_eV)))
    mu = max(float(mu0_J_per_T), 0.0)
    cavity_energy = max(float(cavity_energy0_J), 0.0)
    cavity_amplitude = (
        _cavity_amplitude_from_energy(cavity_interaction, cavity_energy)
        if cavity_amplitude0_sqrt_J is None
        else complex(cavity_amplitude0_sqrt_J)
    )
    phase_rf = float(phase_rf0_rad)
    phi_gc = float(phi_gc0_rad)
    gamma_floor = _gamma_floor_from_energy_floor(float(energy_floor_eV))

    def _zdot_at(z_val: float, v_ref_val: float, gamma_val: float, mu_val: float) -> float:
        B_val = float(np.asarray(_B_along(field, axial_profile, r0_m, z_val)).reshape(()))
        v_abs = _vpar_abs_from_B_gamma_mu(B_val, gamma_val, mu_val)
        b_z = float(np.asarray(_bz_over_B(axial_profile, z_val)).reshape(()))
        return float(np.sign(v_ref_val if v_ref_val != 0.0 else 1.0) * v_abs * abs(b_z))

    gamma_arr[0] = gamma
    mu_arr[0] = mu
    energy_arr[0] = _energy_eV_from_gamma(gamma)
    cavity_arr[0] = cavity_energy
    phase_arr[0] = phase_rf
    phi_arr[0] = phi_gc
    amplitude_arr[0] = cavity_amplitude
    v_arr[0] = _zdot_at(float(z_arr[0]), float(v_ref_arr[0]), gamma, mu)

    use_vectorized_drive = bool(
        back_reaction_block_vectorized
        and _complex_back_reaction_available(cavity_interaction, mode_map)
        and n > 1
    )

    max_updates_per_bounce = max(1, int(back_reaction_max_updates_per_bounce))
    points_per_bounce = max(1, int(np.ceil((n - 1) / max(float(n_block), 1.0))))
    update_stride = 1
    if use_vectorized_drive:
        update_stride = max(1, int(np.ceil(points_per_bounce / float(max_updates_per_bounce))))
    update_idx = np.arange(0, n, update_stride, dtype=int)
    if update_idx.size == 0 or int(update_idx[0]) != 0:
        update_idx = np.concatenate([np.asarray([0], dtype=int), update_idx])
    if int(update_idx[-1]) != n - 1:
        update_idx = np.concatenate([update_idx, np.asarray([n - 1], dtype=int)])
    update_idx = np.unique(update_idx)

    drive_mid_block: np.ndarray | None = None
    if use_vectorized_drive:
        i0_all = update_idx[:-1]
        i1_all = update_idx[1:]
        dt_block = t_arr[i1_all] - t_arr[i0_all]
        z_mid_all = 0.5 * (z_arr[i0_all] + z_arr[i1_all])
        B_mid_all = np.asarray(_B_along(field, axial_profile, r0_m, z_mid_all), dtype=float)
        phi_pred_edges = np.empty(update_idx.size, dtype=float)
        phase_pred_edges = np.empty(update_idx.size, dtype=float)
        phi_pred_edges[0] = phi_gc
        phase_pred_edges[0] = phase_rf
        for j, (i0, i1) in enumerate(zip(i0_all, i1_all)):
            sgn_j = float(np.sign(v_ref_arr[int(i0)] if v_ref_arr[int(i0)] != 0.0 else 1.0))
            dphi_j = _local_dphi_dt(
                z_m=float(z_arr[int(i0)]),
                gamma=float(gamma0),
                mu_J_per_T=float(mu0_J_per_T),
                parallel_sign=sgn_j,
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                include_gradB=bool(include_gradB),
                include_curvature_drift=bool(include_curvature_drift),
            )
            phi_pred_edges[j + 1] = phi_pred_edges[j] + dphi_j * float(dt_block[j])
            fc_j = float(np.asarray(cyclotron_frequency_hz(float(B_mid_all[j]), float(gamma0))).reshape(()))
            phase_pred_edges[j + 1] = phase_pred_edges[j] + 2.0 * np.pi * fc_j * float(dt_block[j])
        phi_mid_block = 0.5 * (phi_pred_edges[:-1] + phi_pred_edges[1:])
        phase_mid_block = 0.5 * (phase_pred_edges[:-1] + phase_pred_edges[1:])
        t_mid_block = 0.5 * (t_arr[i0_all] + t_arr[i1_all])
        drive_mid_block = _complex_cavity_drive_block(
            B_T=B_mid_all,
            gamma=np.full_like(B_mid_all, float(gamma0), dtype=float),
            mu_J_per_T=np.full_like(B_mid_all, float(mu0_J_per_T), dtype=float),
            z_m=z_mid_all,
            phi_gc_rad=phi_mid_block,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mode_map=mode_map,  # type: ignore[arg-type]
            cavity_interaction=cavity_interaction,  # type: ignore[arg-type]
            phase_rf_rad=phase_mid_block,
            t_s=t_mid_block,
        )
        if stats is not None:
            stats["n_block_vectorized_drive_calls"] = int(stats.get("n_block_vectorized_drive_calls", 0)) + 1
            stats["n_block_vectorized_drive_samples"] = int(stats.get("n_block_vectorized_drive_samples", 0)) + int(drive_mid_block.size)
            stats["n_template_update_intervals"] = int(stats.get("n_template_update_intervals", 0)) + int(update_idx.size - 1)
            stats["template_update_stride"] = max(int(stats.get("template_update_stride", 0)), int(update_stride))

    if use_vectorized_drive and bool(back_reaction_predictor_corrector) and drive_mid_block is not None:
        gamma_edges = np.empty(update_idx.size, dtype=float)
        mu_edges = np.empty(update_idx.size, dtype=float)
        phase_edges = np.empty(update_idx.size, dtype=float)
        phi_edges = np.empty(update_idx.size, dtype=float)
        amp_pred = complex(cavity_amplitude)
        gamma_pred = float(gamma)
        mu_pred = float(mu)
        phase_pred = float(phase_rf)
        phi_pred = float(phi_gc)
        gamma_edges[0] = gamma_pred
        mu_edges[0] = mu_pred
        phase_edges[0] = phase_pred
        phi_edges[0] = phi_pred
        for j, (i0_raw, i1_raw) in enumerate(zip(update_idx[:-1], update_idx[1:])):
            i0 = int(i0_raw)
            i1 = int(i1_raw)
            dt = float(t_arr[i1] - t_arr[i0])
            z_mid = 0.5 * (float(z_arr[i0]) + float(z_arr[i1]))
            sign_i = float(np.sign(v_ref_arr[i0] if v_ref_arr[i0] != 0.0 else 1.0))
            dphi_dt_i = _local_dphi_dt(
                z_m=float(z_arr[i0]),
                gamma=float(gamma_pred),
                mu_J_per_T=float(mu_pred),
                parallel_sign=sign_i,
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                include_gradB=bool(include_gradB),
                include_curvature_drift=bool(include_curvature_drift),
            )
            phi_mid_pred = phi_pred + 0.5 * dphi_dt_i * dt
            gamma_pred, mu_pred, amp_pred, phase_pred, _work = _radiative_state_step(
                gamma=float(gamma_pred),
                mu_J_per_T=float(mu_pred),
                B_T=float(B_mid_all[j]),
                dt_s=float(dt),
                z_m=float(z_mid),
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                mode_map=mode_map,
                resonance=resonance,
                cavity_interaction=cavity_interaction,
                cavity_amplitude_sqrt_J=complex(amp_pred),
                phase_rf_rad=float(phase_pred),
                t_s=float(t_arr[i0]),
                energy_loss_scale=float(energy_loss_scale),
                gamma_floor=float(gamma_floor),
                phi_gc_rad=float(phi_mid_pred),
                drive_mid_sqrt_J_per_s=complex(drive_mid_block[j]),
                stats=None,
            )
            phi_pred = phi_pred + dphi_dt_i * dt
            gamma_edges[j + 1] = float(gamma_pred)
            mu_edges[j + 1] = float(mu_pred)
            phase_edges[j + 1] = float(phase_pred)
            phi_edges[j + 1] = float(phi_pred)

        gamma_mid_corr = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
        mu_mid_corr = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        phase_mid_corr = 0.5 * (phase_edges[:-1] + phase_edges[1:])
        phi_mid_corr = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        phase_slip = float(np.max(np.abs(np.angle(np.exp(1j * (phase_mid_corr - phase_mid_block)))))) if phase_mid_corr.size else 0.0
        if int(n_block) > 1 and phase_slip > float(mirror_template_max_phase_slip_rad):
            if stats is not None:
                stats["n_template_blocks_rejected_phase_slip"] = int(stats.get("n_template_blocks_rejected_phase_slip", 0)) + 1
                stats["last_template_phase_slip_rad"] = float(phase_slip)
            return None
        drive_mid_block = _complex_cavity_drive_block(
            B_T=B_mid_all,
            gamma=gamma_mid_corr,
            mu_J_per_T=mu_mid_corr,
            z_m=z_mid_all,
            phi_gc_rad=phi_mid_corr,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mode_map=mode_map,  # type: ignore[arg-type]
            cavity_interaction=cavity_interaction,  # type: ignore[arg-type]
            phase_rf_rad=phase_mid_corr,
            t_s=t_mid_block,
        )
        if stats is not None:
            stats["n_block_predictor_corrector_refreshes"] = int(stats.get("n_block_predictor_corrector_refreshes", 0)) + 1
            stats["last_template_phase_slip_rad"] = float(phase_slip)
            stats["n_block_vectorized_drive_calls"] = int(stats.get("n_block_vectorized_drive_calls", 0)) + 1
            stats["n_block_vectorized_drive_samples"] = int(stats.get("n_block_vectorized_drive_samples", 0)) + int(drive_mid_block.size)

    for k, (i0_raw, i1_raw) in enumerate(zip(update_idx[:-1], update_idx[1:])):
        i0 = int(i0_raw)
        i1 = int(i1_raw)
        dt = float(t_arr[i1] - t_arr[i0])
        if dt <= 0.0:
            return None
        z_mid = 0.5 * (float(z_arr[i0]) + float(z_arr[i1]))
        sign_i = float(np.sign(v_ref_arr[i0] if v_ref_arr[i0] != 0.0 else 1.0))
        dphi_dt_i = _local_dphi_dt(
            z_m=float(z_arr[i0]),
            gamma=float(gamma),
            mu_J_per_T=float(mu),
            parallel_sign=sign_i,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            include_gradB=bool(include_gradB),
            include_curvature_drift=bool(include_curvature_drift),
        )
        phi_mid = phi_gc + 0.5 * dphi_dt_i * dt
        B_mid = float(np.asarray(_B_along(field, axial_profile, r0_m, z_mid)).reshape(()))
        drive_mid = None if drive_mid_block is None else complex(drive_mid_block[k])

        gamma_start = float(gamma)
        mu_start = float(mu)
        phase_start = float(phase_rf)
        phi_start = float(phi_gc)
        amp_start = complex(cavity_amplitude)

        gamma, mu, cavity_amplitude, phase_rf, _work_power = _radiative_state_step(
            gamma=float(gamma),
            mu_J_per_T=float(mu),
            B_T=float(B_mid),
            dt_s=float(dt),
            z_m=float(z_mid),
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mode_map=mode_map,
            resonance=resonance,
            cavity_interaction=cavity_interaction,
            cavity_amplitude_sqrt_J=complex(cavity_amplitude),
            phase_rf_rad=float(phase_rf),
            t_s=float(t_arr[i0]),
            energy_loss_scale=float(energy_loss_scale),
            gamma_floor=float(gamma_floor),
            phi_gc_rad=float(phi_mid),
            drive_mid_sqrt_J_per_s=drive_mid,
            stats=stats,
        )
        phi_gc = phi_gc + dphi_dt_i * dt
        cavity_energy = float(abs(cavity_amplitude) ** 2)

        z_check = np.asarray(z_arr[i0 + 1 : i1 + 1], dtype=float)
        if z_check.size:
            B_check = np.asarray(_B_along(field, axial_profile, r0_m, z_check), dtype=float)
            beta_check = np.asarray(beta_parallel2_from_B_gamma_mu(B_check, float(gamma), float(mu)), dtype=float)
            beta_min = float(np.min(beta_check)) if beta_check.size else 0.0
            if beta_min < -float(mirror_tol):
                if stats is not None:
                    stats["n_template_blocks_rejected_moving_mirror"] = int(stats.get("n_template_blocks_rejected_moving_mirror", 0)) + 1
                    stats["last_template_mirror_beta_parallel2"] = float(beta_min)
                LOGGER.debug(
                    "Fast template radiative block crossed the moving mirror between nodes %d and %d; falling back.",
                    i0,
                    i1,
                )
                return None

        if i1 > i0:
            local_t = np.asarray(t_arr[i0 + 1 : i1 + 1], dtype=float)
            frac = (local_t - float(t_arr[i0])) / max(float(t_arr[i1] - t_arr[i0]), 1.0e-300)
            gamma_arr[i0 + 1 : i1 + 1] = gamma_start + frac * (float(gamma) - gamma_start)
            mu_arr[i0 + 1 : i1 + 1] = mu_start + frac * (float(mu) - mu_start)
            phase_arr[i0 + 1 : i1 + 1] = phase_start + frac * (float(phase_rf) - phase_start)
            phi_arr[i0 + 1 : i1 + 1] = phi_start + frac * (float(phi_gc) - phi_start)
            amp_end = complex(cavity_amplitude)
            amplitude_arr[i0 + 1 : i1 + 1] = amp_start + frac * (amp_end - amp_start)
            cavity_arr[i0 + 1 : i1 + 1] = np.abs(amplitude_arr[i0 + 1 : i1 + 1]) ** 2
            energy_arr[i0 + 1 : i1 + 1] = np.asarray(kinetic_energy_eV_from_gamma(gamma_arr[i0 + 1 : i1 + 1]), dtype=float)

    B_nodes = np.asarray(_B_along(field, axial_profile, r0_m, z_arr), dtype=float)
    beta_nodes = np.asarray(beta_parallel2_from_B_gamma_mu(B_nodes, gamma_arr, mu_arr), dtype=float)
    bz_factor_nodes = np.abs(np.asarray(_bz_over_B(axial_profile, z_arr), dtype=float))
    v_arr[:] = np.sign(np.where(v_ref_arr != 0.0, v_ref_arr, 1.0)) * const.C0 * np.sqrt(np.maximum(beta_nodes, 0.0)) * bz_factor_nodes

    if use_vectorized_drive:
        rel_E = abs(float(energy_arr[-1]) - float(energy_arr[0])) / max(abs(float(energy_arr[0])), 1.0e-30)
        rel_mu = abs(float(mu_arr[-1]) - float(mu_arr[0])) / max(abs(float(mu_arr[0])), 1.0e-30)
        if max(rel_E, rel_mu) > float(back_reaction_block_max_rel_state_change):
            if stats is not None:
                stats["n_template_blocks_rejected_state_change"] = int(stats.get("n_template_blocks_rejected_state_change", 0)) + 1
            return None

        if int(n_block) > 1:
            try:
                solver_end = AxialSolver(
                    field=field,
                    r0_m=float(r0_m),
                    E0_eV=float(energy_arr[-1]),
                    mu0_J_per_T=float(mu_arr[-1]),
                    dt_max_s=float(dt_max_s),
                    dt_min_s=float(dt_min_s),
                    safety=float(safety),
                    v_turn_threshold_c=float(v_turn_threshold_c),
                    axial_profile=axial_profile,
                )
                tpl_end = build_bounce_template(
                    solver_end,
                    z0_m=float(z0_m),
                    vpar_sign0=int(vpar_sign),
                    duration_hint_s=min(float(template_max_duration_s), 5.0e-5),
                    max_duration_s=float(template_max_duration_s),
                    return_z_tol_m=float(template_return_z_tol_m),
                    min_reflections=int(template_min_reflections),
                    method=str(template_build),
                    mirror_z0_tol_m=float(mirror_z0_tol_m),
                    mirror_symmetry_check=bool(mirror_symmetry_check),
                    mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
                    mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
                    mirror_quadrature_min_theta_nodes=int(mirror_quadrature_min_theta_nodes),
                    mirror_quadrature_max_theta_nodes=int(mirror_quadrature_max_theta_nodes),
                    mirror_template_max_period_rel_error=float(mirror_template_max_period_rel_error),
                )
                period_rel = abs(float(tpl_end.bounce_period_s) - float(period)) / max(abs(float(period)), 1.0e-300)
                B_mean = float(np.mean(B_nodes)) if B_nodes.size else float(_B_along(field, axial_profile, r0_m, z0_m))
                gamma_mean = float(np.mean(gamma_arr)) if gamma_arr.size else float(gamma)
                fc_mean = float(np.asarray(cyclotron_frequency_hz(B_mean, gamma_mean)).reshape(()))
                omega_ref = 2.0 * np.pi * fc_mean
                if cavity_interaction is not None:
                    omega_ref = abs(omega_ref - float(cavity_interaction.omega_lo_rad_per_s))
                phase_slip_gate = float(omega_ref) * abs(float(tpl_end.bounce_period_s) - float(period)) * float(n_block)
                if stats is not None:
                    stats["last_template_period_change_rel"] = float(period_rel)
                    stats["last_template_multibounce_phase_slip_rad"] = float(phase_slip_gate)
                if (
                    period_rel > float(mirror_template_max_period_rel_error)
                    or phase_slip_gate > float(mirror_template_max_phase_slip_rad)
                ):
                    if stats is not None:
                        stats["n_template_blocks_rejected_multibounce_gate"] = int(stats.get("n_template_blocks_rejected_multibounce_gate", 0)) + 1
                    return None
            except Exception:
                if stats is not None:
                    stats["n_template_blocks_rejected_multibounce_gate"] = int(stats.get("n_template_blocks_rejected_multibounce_gate", 0)) + 1
                return None
    if stats is not None:
        stats["n_template_blocks_succeeded"] = int(stats.get("n_template_blocks_succeeded", 0)) + 1
    return t_arr, z_arr, v_arr, energy_arr, mu_arr, cavity_arr, phase_arr, amplitude_arr, phi_arr


def _integrate_radiative_bounce_block(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile],
    gamma0: float,
    mu0_J_per_T: float,
    t0_s: float,
    z0_m: float,
    vpar_sign: int,
    energy_floor_eV: float,
    energy_loss_scale: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    bounce_period_guess_s: float,
    block_bounces: int,
    template_return_z_tol_m: float,
    template_min_reflections: int,
    template_max_duration_s: float,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    phase_rf0_rad: float = 0.0,
    phi_gc0_rad: float = 0.0,
    cavity_amplitude0_sqrt_J: complex | None = None,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    stats: Optional[dict[str, int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate several full bounces continuously before restarting the compact solve.

    This keeps the cached 1D radiative solver while reducing the accumulated phase bias
    from restarting after every individual bounce return.
    """
    n_block = max(1, int(block_bounces))
    guess = max(float(bounce_period_guess_s), 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))
    target_duration = n_block * guess
    energy0_eV = _energy_eV_from_gamma(float(gamma0))

    factors = (1.06, 1.12, 1.25, 1.40, 1.65, 2.00)
    for fac in factors:
        duration = min(float(template_max_duration_s), fac * target_duration)
        if duration <= 1.01 * guess:
            duration = min(float(template_max_duration_s), 1.01 * guess)

        seg = integrate_axial_track_energy_analytic(
            field=field,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            mu0_J_per_T=float(mu0_J_per_T),
            t0_s=float(t0_s),
            duration_s=float(duration),
            z0_m=float(z0_m),
            vpar_sign=int(vpar_sign),
            energy0_eV=float(energy0_eV),
            energy_floor_eV=float(energy_floor_eV),
            energy_loss_scale=float(energy_loss_scale),
            dt_max_s=float(dt_max_s),
            dt_min_s=float(dt_min_s),
            safety=float(safety),
            v_turn_threshold_c=float(v_turn_threshold_c),
            mode_map=mode_map,
            resonance=resonance,
            cavity_interaction=cavity_interaction,
            cavity_energy0_J=float(cavity_energy0_J),
            cavity_amplitude0_sqrt_J=cavity_amplitude0_sqrt_J,
            phase_rf0_rad=float(phase_rf0_rad),
            phi_gc0_rad=float(phi_gc0_rad),
            include_gradB=bool(include_gradB),
            include_curvature_drift=bool(include_curvature_drift),
            stats=stats,
            return_aux=True,
        )
        t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_aux, amp_aux, phi_aux = seg
        truncated = _truncate_track_at_bounce_return(
            t_s=t_seg,
            z_m=z_seg,
            vz_m_per_s=v_seg,
            energy_eV=E_seg,
            mu_J_per_T=mu_seg,
            cavity_energy_J=U_seg,
            z0_m=float(z0_m),
            vpar_sign0=int(vpar_sign),
            z_tol_m=float(template_return_z_tol_m),
            min_reflections=int(template_min_reflections),
            n_bounce_returns=n_block,
        )
        if truncated is not None:
            t_tr, z_tr, v_tr, E_tr, mu_tr, U_tr = truncated
            phase_tr = np.interp(t_tr, t_seg, phase_aux)
            amp_tr = np.interp(t_tr, t_seg, np.real(amp_aux)) + 1j * np.interp(t_tr, t_seg, np.imag(amp_aux))
            phi_tr = np.interp(t_tr, t_seg, phi_aux)
            return t_tr, z_tr, v_tr, E_tr, mu_tr, U_tr, phase_tr, amp_tr, phi_tr

    raise RuntimeError(
        f"Failed to bracket {n_block} radiative bounce return(s) within the configured maximum duration. "
        "Increase dynamics.template_max_duration_s, reduce dynamics.per_bounce_block_bounces, or inspect the mirror condition."
    )


def build_axial_track_energy_per_bounce(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile] = None,
    mu_J_per_T: float,
    t0_s: float,
    duration_s: float,
    z0_m: float,
    vpar_sign: int,
    energy0_eV: float,
    energy_floor_eV: float,
    energy_loss_scale: float,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    template_build: str,
    template_return_z_tol_m: float,
    template_max_duration_s: float,
    template_min_reflections: int,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    per_bounce_block_bounces: int = 1,
    phi_gc0_rad: float = 0.0,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    back_reaction_block_vectorized: bool = True,
    back_reaction_block_max_rel_state_change: float = 5.0e-3,
    back_reaction_max_updates_per_bounce: int = 96,
    back_reaction_predictor_corrector: bool = True,
    multi_bounce_auto_max_bounces: int = 8,
    mirror_quadrature_min_theta_nodes: int = 513,
    mirror_quadrature_max_theta_nodes: int = 8193,
    mirror_template_max_period_rel_error: float = 1.0e-3,
    mirror_template_max_phase_slip_rad: float = 5.0e-3,
    mirror_template_moving_mirror_tol: float = 1.0e-8,
    stats: Optional[dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast bouncewise radiative correction using a direct 1D cached-field-line solve per bounce.

    The previous implementation evolved (γ, μ) on a *fixed spatial bounce template* and only
    corrected the time-of-flight inside each segment. That improved axial timing, but the
    template geometry itself could still extend slightly beyond the moving mirror implied by the
    updated (γ, μ). The resulting over-extended z(t) produced the kind of slowly growing field /
    frequency discrepancy seen against the high-fidelity tracks.

    The new implementation keeps the computationally cheap cached 1D field-line model and the
    bouncewise structure, but integrates each block of one or more bounces directly with the same
    radiative ODE used by analytic mode. The symmetric mirror-template machinery is still exploited
    to obtain an initial bounce-period guess quickly; subsequent blocks reuse the last actual period,
    so the compact solver stays efficient without freezing the spatial geometry. Using multi-bounce
    blocks reduces the small phase-reset error that otherwise accumulates when restarting exactly at
    every bounce return.
    """
    t_end = float(t0_s) + float(duration_s)
    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), _gamma_floor_from_energy_floor(float(energy_floor_eV)))
    mu = max(float(mu_J_per_T), 0.0)
    cavity_energy = max(float(cavity_energy0_J), 0.0)
    cavity_amplitude = _cavity_amplitude_from_energy(cavity_interaction, cavity_energy)
    phase_rf = 0.0
    phi_gc = float(phi_gc0_rad)
    t_cursor = float(t0_s)
    bounce_period_guess_s: float | None = None
    max_block_bounces_next = max(1, int(per_bounce_block_bounces))

    t_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    v_out: list[np.ndarray] = []
    E_out: list[np.ndarray] = []
    mu_out: list[np.ndarray] = []
    cavity_energy_out: list[np.ndarray] = []

    while t_cursor < t_end - 1.0e-18:
        remaining = t_end - t_cursor
        E_cur = _energy_eV_from_gamma(gamma)
        mu_cur = float(mu)

        if bounce_period_guess_s is None:
            bounce_period_guess_s = _initial_bounce_period_guess(
                field=field,
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                z0_m=float(z0_m),
                vpar_sign=int(vpar_sign),
                energy_eV=float(E_cur),
                mu_J_per_T=float(mu),
                dt_max_s=float(dt_max_s),
                dt_min_s=float(dt_min_s),
                safety=float(safety),
                v_turn_threshold_c=float(v_turn_threshold_c),
                template_build=str(template_build),
                template_return_z_tol_m=float(template_return_z_tol_m),
                template_max_duration_s=float(template_max_duration_s),
                template_min_reflections=int(template_min_reflections),
                mirror_z0_tol_m=float(mirror_z0_tol_m),
                mirror_symmetry_check=bool(mirror_symmetry_check),
                mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
                mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
                mirror_quadrature_min_theta_nodes=int(mirror_quadrature_min_theta_nodes),
                mirror_quadrature_max_theta_nodes=int(mirror_quadrature_max_theta_nodes),
                mirror_template_max_period_rel_error=float(mirror_template_max_period_rel_error),
            )

        is_first = len(t_out) == 0

        if remaining <= 1.05 * max(float(bounce_period_guess_s), 8.0 * float(dt_max_s)):
            seg = integrate_axial_track_energy_analytic(
                field=field,
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                mu0_J_per_T=float(mu),
                t0_s=float(t_cursor),
                duration_s=float(remaining),
                z0_m=float(z0_m),
                vpar_sign=int(vpar_sign),
                energy0_eV=float(E_cur),
                energy_floor_eV=float(energy_floor_eV),
                energy_loss_scale=float(energy_loss_scale),
                dt_max_s=float(dt_max_s),
                dt_min_s=float(dt_min_s),
                safety=float(safety),
                v_turn_threshold_c=float(v_turn_threshold_c),
                mode_map=mode_map,
                resonance=resonance,
                cavity_interaction=cavity_interaction,
                cavity_energy0_J=float(cavity_energy),
                cavity_amplitude0_sqrt_J=complex(cavity_amplitude),
                phase_rf0_rad=float(phase_rf),
                phi_gc0_rad=float(phi_gc),
                include_gradB=bool(include_gradB),
                include_curvature_drift=bool(include_curvature_drift),
                stats=stats,
                return_aux=True,
            )
            t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_seg, amp_seg, phi_seg = seg
            phase_rf = float(phase_seg[-1]) if phase_seg.size else phase_rf
            cavity_amplitude = complex(amp_seg[-1]) if amp_seg.size else cavity_amplitude
            if not is_first:
                t_seg = t_seg[1:]
                z_seg = z_seg[1:]
                v_seg = v_seg[1:]
                E_seg = E_seg[1:]
                mu_seg = mu_seg[1:]
                U_seg = U_seg[1:]
            t_out.append(t_seg)
            z_out.append(z_seg)
            v_out.append(v_seg)
            E_out.append(E_seg)
            mu_out.append(mu_seg)
            cavity_energy_out.append(U_seg)
            break

        target_block_bounces = max(1, min(int(max_block_bounces_next), int(multi_bounce_auto_max_bounces)))
        if bounce_period_guess_s > 0.0:
            est_remaining = max(1, int(np.floor(remaining / bounce_period_guess_s)))
            target_block_bounces = min(target_block_bounces, est_remaining)

        last_exc: Exception | None = None
        use_fast_template_block = bool(
            back_reaction_block_vectorized
            and _complex_back_reaction_available(cavity_interaction, mode_map)
        )
        for block_bounces in range(target_block_bounces, 0, -1):
            fast_segment = None
            if use_fast_template_block:
                fast_segment = _integrate_template_periodic_radiative_block(
                    field=field,
                    r0_m=float(r0_m),
                    axial_profile=axial_profile,
                    gamma0=float(gamma),
                    mu0_J_per_T=float(mu),
                    t0_s=float(t_cursor),
                    z0_m=float(z0_m),
                    vpar_sign=int(vpar_sign),
                    energy_floor_eV=float(energy_floor_eV),
                    energy_loss_scale=float(energy_loss_scale),
                    dt_max_s=float(dt_max_s),
                    dt_min_s=float(dt_min_s),
                    safety=float(safety),
                    v_turn_threshold_c=float(v_turn_threshold_c),
                    block_bounces=int(block_bounces),
                    template_build=str(template_build),
                    template_return_z_tol_m=float(template_return_z_tol_m),
                    template_max_duration_s=float(template_max_duration_s),
                    template_min_reflections=int(template_min_reflections),
                    mirror_z0_tol_m=float(mirror_z0_tol_m),
                    mirror_symmetry_check=bool(mirror_symmetry_check),
                    mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
                    mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
                    mode_map=mode_map,
                    resonance=resonance,
                    cavity_interaction=cavity_interaction,
                    cavity_energy0_J=float(cavity_energy),
                    phase_rf0_rad=float(phase_rf),
                    phi_gc0_rad=float(phi_gc),
                    cavity_amplitude0_sqrt_J=complex(cavity_amplitude),
                    include_gradB=bool(include_gradB),
                    include_curvature_drift=bool(include_curvature_drift),
                    back_reaction_block_vectorized=bool(back_reaction_block_vectorized),
                    back_reaction_block_max_rel_state_change=float(back_reaction_block_max_rel_state_change),
                    back_reaction_max_updates_per_bounce=int(back_reaction_max_updates_per_bounce),
                    back_reaction_predictor_corrector=bool(back_reaction_predictor_corrector),
                    mirror_quadrature_min_theta_nodes=int(mirror_quadrature_min_theta_nodes),
                    mirror_quadrature_max_theta_nodes=int(mirror_quadrature_max_theta_nodes),
                    mirror_template_max_period_rel_error=float(mirror_template_max_period_rel_error),
                    mirror_template_max_phase_slip_rad=float(mirror_template_max_phase_slip_rad),
                    mirror_tol=float(mirror_template_moving_mirror_tol),
                    stats=stats,
                    )
            if fast_segment is not None:
                t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_seg, amp_seg, phi_seg = fast_segment
                phase_rf = float(phase_seg[-1]) if phase_seg.size else phase_rf
                phi_gc = float(phi_seg[-1]) if phi_seg.size else phi_gc
                cavity_amplitude = complex(amp_seg[-1]) if amp_seg.size else cavity_amplitude
                used_block_bounces = int(block_bounces)
                LOGGER.debug(
                    "Used fast mirror/template radiative block with %d bounce return(s).",
                    used_block_bounces,
                )
                break

            try:
                seg = _integrate_radiative_bounce_block(
                    field=field,
                    r0_m=float(r0_m),
                    axial_profile=axial_profile,
                    gamma0=float(gamma),
                    mu0_J_per_T=float(mu),
                    t0_s=float(t_cursor),
                    z0_m=float(z0_m),
                    vpar_sign=int(vpar_sign),
                    energy_floor_eV=float(energy_floor_eV),
                    energy_loss_scale=float(energy_loss_scale),
                    dt_max_s=float(dt_max_s),
                    dt_min_s=float(dt_min_s),
                    safety=float(safety),
                    v_turn_threshold_c=float(v_turn_threshold_c),
                    bounce_period_guess_s=float(bounce_period_guess_s),
                    block_bounces=int(block_bounces),
                    template_return_z_tol_m=float(template_return_z_tol_m),
                    template_min_reflections=int(template_min_reflections),
                    template_max_duration_s=float(template_max_duration_s),
                    mode_map=mode_map,
                    resonance=resonance,
                    cavity_interaction=cavity_interaction,
                    cavity_energy0_J=float(cavity_energy),
                    phase_rf0_rad=float(phase_rf),
                    phi_gc0_rad=float(phi_gc),
                    cavity_amplitude0_sqrt_J=complex(cavity_amplitude),
                    include_gradB=bool(include_gradB),
                    include_curvature_drift=bool(include_curvature_drift),
                    stats=stats,
                )
                t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_seg, amp_seg, phi_seg = seg
                phase_rf = float(phase_seg[-1]) if phase_seg.size else phase_rf
                phi_gc = float(phi_seg[-1]) if phi_seg.size else phi_gc
                cavity_amplitude = complex(amp_seg[-1]) if amp_seg.size else cavity_amplitude
                used_block_bounces = int(block_bounces)
                LOGGER.info(
                    "Fell back to continuous compact radiative block with %d bounce return(s).",
                    used_block_bounces,
                )
                break
            except RuntimeError as exc:
                last_exc = exc
        else:
            # When radiation changes the state too strongly for even a single full return to
            # be bracketed robustly, fall back to the continuous analytic integrator for the
            # remaining interval. This preserves physics and avoids wasting time repeatedly
            # trying to force a bouncewise decomposition where one is no longer appropriate.
            seg = integrate_axial_track_energy_analytic(
                field=field,
                r0_m=float(r0_m),
                axial_profile=axial_profile,
                mu0_J_per_T=float(mu),
                t0_s=float(t_cursor),
                duration_s=float(remaining),
                z0_m=float(z0_m),
                vpar_sign=int(vpar_sign),
                energy0_eV=float(E_cur),
                energy_floor_eV=float(energy_floor_eV),
                energy_loss_scale=float(energy_loss_scale),
                dt_max_s=float(dt_max_s),
                dt_min_s=float(dt_min_s),
                safety=float(safety),
                v_turn_threshold_c=float(v_turn_threshold_c),
                mode_map=mode_map,
                resonance=resonance,
                cavity_interaction=cavity_interaction,
                cavity_energy0_J=float(cavity_energy),
                cavity_amplitude0_sqrt_J=complex(cavity_amplitude),
                phase_rf0_rad=float(phase_rf),
                phi_gc0_rad=float(phi_gc),
                include_gradB=bool(include_gradB),
                include_curvature_drift=bool(include_curvature_drift),
                stats=stats,
                return_aux=True,
            )
            t_seg, z_seg, v_seg, E_seg, mu_seg, U_seg, phase_seg, amp_seg, phi_seg = seg
            phase_rf = float(phase_seg[-1]) if phase_seg.size else phase_rf
            phi_gc = float(phi_seg[-1]) if phi_seg.size else phi_gc
            cavity_amplitude = complex(amp_seg[-1]) if amp_seg.size else cavity_amplitude
            used_block_bounces = 1

        if not is_first:
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]
            E_seg = E_seg[1:]
            mu_seg = mu_seg[1:]
            U_seg = U_seg[1:]
            if 'phase_seg' in locals():
                phase_seg = phase_seg[1:]
            if 'amp_seg' in locals():
                amp_seg = amp_seg[1:]
            if 'phi_seg' in locals():
                phi_seg = phi_seg[1:]

        t_out.append(t_seg)
        z_out.append(z_seg)
        v_out.append(v_seg)
        E_out.append(E_seg)
        mu_out.append(mu_seg)
        cavity_energy_out.append(U_seg)

        if t_seg.size == 0:
            break

        if float(t_seg[-1]) >= t_end - 1.0e-18:
            break

        gamma = _gamma_from_energy_eV(float(E_seg[-1]))
        mu = float(mu_seg[-1])
        cavity_energy = float(U_seg[-1])
        # phase_rf and cavity_amplitude were advanced by the selected block helper.
        t_cursor = float(t_seg[-1])
        block_period = float(t_seg[-1] - t_seg[0]) / max(float(used_block_bounces), 1.0)
        bounce_period_guess_s = max(block_period, 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))

        E_start_block = max(float(E_cur), 1.0e-30)
        mu_start_block = max(float(mu_cur), 1.0e-30)
        rel_E_per_bounce = max(0.0, (float(E_cur) - float(E_seg[-1])) / E_start_block) / max(float(used_block_bounces), 1.0)
        rel_mu_per_bounce = max(0.0, (mu_start_block - float(mu_seg[-1])) / mu_start_block) / max(float(used_block_bounces), 1.0)
        rel_state_per_bounce = max(rel_E_per_bounce, rel_mu_per_bounce)
        if rel_state_per_bounce > 0.0:
            # Keep the per-block radiative state change modest. This prevents the fast
            # production path from wasting time trying to bracket large multi-bounce blocks
            # after the bounce period has changed substantially.
            target_rel_state_change = 0.15
            adaptive_cap = max(1, int(np.floor(target_rel_state_change / rel_state_per_bounce)))
            max_block_bounces_next = max(1, min(int(per_bounce_block_bounces), adaptive_cap))
        else:
            max_block_bounces_next = max(1, int(per_bounce_block_bounces))

    if not t_out:
        return (
            np.asarray([float(t0_s)], dtype=float),
            np.asarray([float(z0_m)], dtype=float),
            np.asarray([0.0], dtype=float),
            np.asarray([float(energy0_eV)], dtype=float),
            np.asarray([float(mu_J_per_T)], dtype=float),
            np.asarray([max(float(cavity_energy0_J), 0.0)], dtype=float),
        )

    return (
        np.concatenate(t_out),
        np.concatenate(z_out),
        np.concatenate(v_out),
        np.concatenate(E_out),
        np.concatenate(mu_out),
        np.concatenate(cavity_energy_out),
    )


# -----------------------------------------------------------------------------
# Linear-fit helper (kept for completeness / compatibility)
# -----------------------------------------------------------------------------


def estimate_linear_radiative_state_rates(
    *,
    field: FieldMap,
    r0_m: float,
    axial_profile: Optional[AxialFieldProfile] = None,
    mu0_J_per_T: float,
    energy0_eV: float,
    energy_floor_eV: float,
    energy_loss_scale: float,
    n_bounces: int,
    dt_max_s: float,
    dt_min_s: float,
    safety: float,
    v_turn_threshold_c: float,
    z0_m: float,
    vpar_sign: int,
    template_build: str,
    template_return_z_tol_m: float,
    template_max_duration_s: float,
    template_min_reflections: int,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
    mode_map: Optional[ModeMap] = None,
    resonance: Optional[ResonanceCurve] = None,
    cavity_interaction: Optional[CavityInteraction] = None,
    cavity_energy0_J: float = 0.0,
    phi_gc0_rad: float = 0.0,
    include_gradB: bool = True,
    include_curvature_drift: bool = False,
    stats: Optional[dict[str, int]] = None,
) -> tuple[float, float]:
    """Estimate linear dE/dt and dμ/dt from bouncewise direct evolving-(γ, μ) updates."""
    if int(n_bounces) < 2:
        raise ValueError("n_bounces must be >= 2 to estimate linear rates.")

    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), _gamma_floor_from_energy_floor(float(energy_floor_eV)))
    mu = max(float(mu0_J_per_T), 0.0)
    cavity_energy = max(float(cavity_energy0_J), 0.0)
    t_cursor = 0.0
    bounce_period_guess_s = _initial_bounce_period_guess(
        field=field,
        r0_m=float(r0_m),
        axial_profile=axial_profile,
        z0_m=float(z0_m),
        vpar_sign=int(vpar_sign),
        energy_eV=float(energy0_eV),
        mu_J_per_T=float(mu0_J_per_T),
        dt_max_s=float(dt_max_s),
        dt_min_s=float(dt_min_s),
        safety=float(safety),
        v_turn_threshold_c=float(v_turn_threshold_c),
        template_build=str(template_build),
        template_return_z_tol_m=float(template_return_z_tol_m),
        template_max_duration_s=float(template_max_duration_s),
        template_min_reflections=int(template_min_reflections),
        mirror_z0_tol_m=float(mirror_z0_tol_m),
        mirror_symmetry_check=bool(mirror_symmetry_check),
        mirror_symmetry_rel_tol=float(mirror_symmetry_rel_tol),
        mirror_symmetry_ncheck=int(mirror_symmetry_ncheck),
    )

    times = [0.0]
    energies = [_energy_eV_from_gamma(gamma)]
    mus = [mu]

    for _ in range(int(n_bounces)):
        t_seg, _, _, E_seg, mu_seg, U_seg, _phase_seg, _amp_seg, _phi_seg = _integrate_one_radiative_bounce(
            field=field,
            r0_m=float(r0_m),
            axial_profile=axial_profile,
            gamma0=float(gamma),
            mu0_J_per_T=float(mu),
            t0_s=float(t_cursor),
            z0_m=float(z0_m),
            vpar_sign=int(vpar_sign),
            energy_floor_eV=float(energy_floor_eV),
            energy_loss_scale=float(energy_loss_scale),
            dt_max_s=float(dt_max_s),
            dt_min_s=float(dt_min_s),
            safety=float(safety),
            v_turn_threshold_c=float(v_turn_threshold_c),
            bounce_period_guess_s=float(bounce_period_guess_s),
            template_return_z_tol_m=float(template_return_z_tol_m),
            template_min_reflections=int(template_min_reflections),
            template_max_duration_s=float(template_max_duration_s),
            mode_map=mode_map,
            resonance=resonance,
            cavity_interaction=cavity_interaction,
            cavity_energy0_J=float(cavity_energy),
            phi_gc0_rad=float(phi_gc0_rad),
            include_gradB=bool(include_gradB),
            include_curvature_drift=bool(include_curvature_drift),
            stats=stats,
        )
        gamma = _gamma_from_energy_eV(float(E_seg[-1]))
        mu = float(mu_seg[-1])
        cavity_energy = float(U_seg[-1])
        t_cursor = float(t_seg[-1])
        bounce_period_guess_s = max(float(t_seg[-1] - t_seg[0]), 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))
        times.append(t_cursor)
        energies.append(float(E_seg[-1]))
        mus.append(mu)

    t = np.asarray(times, dtype=float)
    if t.size < 2:
        return 0.0, 0.0

    A = np.vstack([np.ones_like(t), t]).T
    coeff_E, *_ = np.linalg.lstsq(A, np.asarray(energies, dtype=float), rcond=None)
    coeff_mu, *_ = np.linalg.lstsq(A, np.asarray(mus, dtype=float), rcond=None)
    return max(0.0, -float(coeff_E[1])), max(0.0, -float(coeff_mu[1]))
