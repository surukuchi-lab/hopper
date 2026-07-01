"""
Module: hopper.dynamics.track

Developer: ehtkarim
Date: April 29, 2026

Builds dynamic electron tracks, reconstructs observable samples, and prepares arrays for signal synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import logging

import numpy as np

from ..cavity.cavity import Cavity
from ..cavity.interaction import CavityInteraction
from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig
from .. import constants as const
from ..field.field_map import FieldMap
from ..utils.math import cumulative_trapezoid, resample_linear
from .axial_profile import AxialFieldProfile
from .axial_solver import AxialSolver
from .drifts import curvature_drift_vphi, gradB_drift_vphi, integrate_phi_from_vphi
from .kinematics import (
    cyclotron_frequency_hz,
    gamma_beta_v_from_kinetic,
    larmor_radius_m_array,
    mu_from_pitch,
    vpar_m_per_s_from_B,
)
from .radiation import (
    build_axial_track_energy_per_bounce,
    estimate_linear_radiative_state_rates,
    integrate_axial_track_energy_analytic,
)
from .template import (
    build_bounce_template,
    tile_bounce_template_constant_energy,
    tile_bounce_template_linear_energy,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class DynamicTrack:
    """
    Compact non-uniform dynamics track.

    The guiding-center state follows the trapping field line. The instantaneous orbit is
    reconstructed around that guiding center using a local perpendicular basis aligned with
    the guiding-center magnetic field, so the cyclotron orbit acquires the expected axial
    displacement when the field line is tilted away from z.  In true-orbit mode the completed
    orbit point is then used for field/coupling evaluation; it is not fed back into the
    gyro phase or Larmor-radius construction.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    x_gc: np.ndarray
    y_gc: np.ndarray
    z_gc: np.ndarray
    vx_gc: np.ndarray
    vy_gc: np.ndarray
    vz_gc: np.ndarray
    r_gc_m: np.ndarray
    phi_gc_rad: np.ndarray
    parallel_sign: np.ndarray
    b_cross_kappa_phi_per_m: np.ndarray

    f_c_hz: np.ndarray
    amp: np.ndarray
    phase_rf: np.ndarray
    B_T: np.ndarray
    energy_eV: np.ndarray
    mu_J_per_T: np.ndarray
    axial_profile: AxialFieldProfile | None = None
    cavity_energy_J: np.ndarray | None = None
    cavity_power_W: np.ndarray | None = None
    cavity_source_power_W: np.ndarray | None = None
    cavity_work_power_W: np.ndarray | None = None
    cavity_amplitude_sqrt_J: np.ndarray | None = None
    cavity_drive_sqrt_J_per_s: np.ndarray | None = None
    solver_info: dict[str, Any] | None = None


@dataclass(frozen=True)
class InitialOrbitState:
    r_gc0_m: float
    phi_gc0_rad: float
    z_gc0_m: float
    x_target_m: float
    y_target_m: float
    z_target_m: float
    B_ref_T: float
    mu0_J_per_T: float


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _infer_initial_xy(cfg: MainConfig) -> Tuple[float, float]:
    e = cfg.electron
    coords = str(getattr(e, "position_coordinates", "auto"))
    if coords == "cartesian":
        if e.x0_m is None or e.y0_m is None:
            raise ValueError("electron.position_coordinates='cartesian' requires electron.x0_m and electron.y0_m.")
        return float(e.x0_m), float(e.y0_m)

    if coords == "cylindrical":
        return float(e.r0_m) * np.cos(float(e.phi0_rad)), float(e.r0_m) * np.sin(float(e.phi0_rad))

    if e.x0_m is not None and e.y0_m is not None:
        return float(e.x0_m), float(e.y0_m)
    return float(e.r0_m) * np.cos(float(e.phi0_rad)), float(e.r0_m) * np.sin(float(e.phi0_rad))


def _cart_to_cyl(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(np.hypot(x, y)), float(np.arctan2(y, x)), float(z)


def _scalar_mu_from_pitch(E0_eV: float, pitch_angle_deg: float, B_ref_T: float) -> float:
    return float(np.asarray(mu_from_pitch(E0_eV, pitch_angle_deg, B_ref_T)).reshape(()))


def _scalar_larmor_radius(B_ref_T: float, E0_eV: float, mu0_J_per_T: float) -> float:
    return float(
        larmor_radius_m_array(
            float(B_ref_T),
            np.asarray([float(E0_eV)], dtype=float),
            np.asarray([float(mu0_J_per_T)], dtype=float),
            q_C=-const.E_CHARGE,
        )[0]
    )


def _local_perp_basis_from_field(
    phi_rad: np.ndarray | float,
    Br_T: np.ndarray | float,
    Bphi_T: np.ndarray | float,
    Bz_T: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed orthonormal basis (u1, u2, b).

    - b is the magnetic-field unit vector.
    - u1 lies in the meridional plane and is perpendicular to b.
    - u2 = b × u1, which reduces to e_phi when Bphi=0.

    For a purely axial field this gives u1=e_r and u2=e_phi, so the old horizontal-orbit
    construction is recovered as a special case.
    """
    phi = np.asarray(phi_rad, dtype=float)
    Br = np.asarray(Br_T, dtype=float)
    Bphi = np.asarray(Bphi_T, dtype=float)
    Bz = np.asarray(Bz_T, dtype=float)

    c = np.cos(phi)
    s = np.sin(phi)
    zeros = np.zeros_like(c)
    ones = np.ones_like(c)

    e_r = np.stack([c, s, zeros], axis=-1)
    e_phi = np.stack([-s, c, zeros], axis=-1)
    e_z = np.stack([zeros, zeros, ones], axis=-1)

    B_vec = Br[..., None] * e_r + Bphi[..., None] * e_phi + Bz[..., None] * e_z
    B_mag = np.linalg.norm(B_vec, axis=-1)
    b = B_vec / np.maximum(B_mag[..., None], 1.0e-300)

    # Natural meridional-plane perpendicular direction.
    u1 = np.cross(e_phi, b)
    u1_norm = np.linalg.norm(u1, axis=-1)

    # Fallbacks for pathological cases; these are not expected for the repo's production fields
    # but keep the basis construction numerically stable.
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
    return u1, u2, b


def _orbit_delta_and_velocity(
    *,
    rho_m: np.ndarray,
    omega_c_rad_per_s: np.ndarray,
    psi_rad: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    delta = rho_m[..., None] * (cos_psi[..., None] * u1 + sin_psi[..., None] * u2)
    v_cyc = (rho_m * omega_c_rad_per_s)[..., None] * (-sin_psi[..., None] * u1 + cos_psi[..., None] * u2)
    return delta, v_cyc



def _phase_from_frequency(
    t_s: np.ndarray,
    f_hz: np.ndarray,
    *,
    phase_start_rad: float = 0.0,
) -> np.ndarray:
    return cumulative_trapezoid(2.0 * np.pi * np.asarray(f_hz, dtype=float), np.asarray(t_s, dtype=float), initial=float(phase_start_rad))


def _select_frequency_reference(cfg: MainConfig) -> str:
    return str(getattr(cfg.dynamics, "cyclotron_frequency_reference", "guiding_center"))


def _true_orbit_phase_iterations(cfg: MainConfig) -> int:
    return max(0, int(getattr(cfg.dynamics, "true_orbit_phase_iterations", 1)))



def _target_compact_output_dt_s(cfg: MainConfig, t_s: np.ndarray) -> float | None:
    explicit = getattr(cfg.dynamics, "compact_output_dt_s", None)
    if explicit is not None:
        return float(explicit)
    # For coherent baseband readout with IF-sampled output, the dynamics record only
    # needs to resolve the readout/source grid.  The mirror-template itself is still
    # built on the fine solver grid; this step only compresses the exported compact
    # state used by later interpolation and vector-mode coupling.
    readout_model = str(getattr(cfg.readout, "model", "none"))
    if _coherent_cavity_response_enabled(cfg) and readout_model in {"locust_like_baseband", "locust_exact_baseband"}:
        if str(getattr(cfg.output, "track_sampling", "if_sampled")).lower() == "if_sampled":
            fs = float(getattr(cfg.signal, "fs_if_hz", 0.0)) * max(int(getattr(cfg.readout, "fast_decimation_factor", 1)), 1)
            if fs > 0.0:
                return 1.0 / fs
    return None


def _compress_axial_state_for_output(
    cfg: MainConfig,
    t_s: np.ndarray,
    z_m: np.ndarray,
    vz_m_per_s: np.ndarray,
    energy_eV: np.ndarray,
    mu_J_per_T: np.ndarray,
    cavity_energy_J: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t = np.asarray(t_s, dtype=float)
    if t.size < 3:
        return t_s, z_m, vz_m_per_s, energy_eV, mu_J_per_T, cavity_energy_J, {"compact_compression_applied": False}
    target_dt = _target_compact_output_dt_s(cfg, t)
    if target_dt is None or target_dt <= 0.0:
        return t_s, z_m, vz_m_per_s, energy_eV, mu_J_per_T, cavity_energy_J, {"compact_compression_applied": False}
    median_dt = float(np.median(np.diff(t)))
    if target_dt <= 1.25 * median_dt:
        return t_s, z_m, vz_m_per_s, energy_eV, mu_J_per_T, cavity_energy_J, {
            "compact_compression_applied": False,
            "compact_compression_reason": "target_dt_not_coarser_than_solver_dt",
            "compact_target_dt_s": float(target_dt),
        }

    t0 = float(t[0])
    t1 = float(t[-1])
    n = max(2, int(np.floor((t1 - t0) / float(target_dt))) + 1)
    uniform = t0 + np.arange(n, dtype=float) * float(target_dt)
    if uniform[-1] < t1:
        uniform = np.concatenate([uniform, np.asarray([t1], dtype=float)])
    else:
        uniform[-1] = t1

    if bool(getattr(cfg.dynamics, "compact_output_include_turning_points", True)):
        v = np.asarray(vz_m_per_s, dtype=float)
        sign = np.sign(v)
        crossing = np.where((sign[:-1] * sign[1:]) < 0.0)[0]
        turn_times = []
        for idx in crossing:
            t_a, t_b = float(t[idx]), float(t[idx + 1])
            v_a, v_b = float(v[idx]), float(v[idx + 1])
            if v_b != v_a:
                frac = -v_a / (v_b - v_a)
                if 0.0 <= frac <= 1.0:
                    turn_times.append(t_a + frac * (t_b - t_a))
        if turn_times:
            uniform = np.concatenate([uniform, np.asarray(turn_times, dtype=float)])

    t_new = np.unique(np.clip(uniform, t0, t1))
    return (
        t_new,
        np.interp(t_new, t, np.asarray(z_m, dtype=float)),
        np.interp(t_new, t, np.asarray(vz_m_per_s, dtype=float)),
        np.interp(t_new, t, np.asarray(energy_eV, dtype=float)),
        np.interp(t_new, t, np.asarray(mu_J_per_T, dtype=float)),
        np.interp(t_new, t, np.asarray(cavity_energy_J, dtype=float)),
        {
            "compact_compression_applied": True,
            "compact_points_before": int(t.size),
            "compact_points_after": int(t_new.size),
            "compact_target_dt_s": float(target_dt),
            "compact_solver_median_dt_s": median_dt,
        },
    )

def _parallel_sign_from_z_velocity(
    vz_gc_m_per_s: np.ndarray,
    bz_over_B: np.ndarray,
    *,
    initial_sign: int,
) -> np.ndarray:
    vz = np.asarray(vz_gc_m_per_s, dtype=float)
    bz = np.asarray(bz_over_B, dtype=float)
    if vz.ndim != 1 or bz.ndim != 1 or vz.size != bz.size:
        raise ValueError("vz_gc_m_per_s and bz_over_B must be 1D arrays with equal length")

    out = np.zeros_like(vz, dtype=float)
    valid = (np.abs(vz) > 1.0e-14) & (np.abs(bz) > 1.0e-14)
    out[valid] = np.sign(vz[valid]) * np.sign(bz[valid])

    cur = 1.0 if int(initial_sign) >= 0 else -1.0
    for i in range(out.size):
        if out[i] == 0.0:
            out[i] = cur
        else:
            cur = out[i]
    return out


def _step_resample(t_old: np.ndarray, y_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    t_old = np.asarray(t_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    idx = np.searchsorted(t_old, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(y_old) - 1)
    return np.asarray(y_old[idx], dtype=float)


def _scalar_resample(t_old: np.ndarray, y_old: np.ndarray, t_new0: float) -> float:
    return float(np.asarray(resample_linear(t_old, y_old, np.asarray([float(t_new0)], dtype=float))).reshape(()))


def _apply_instantaneous_anchor(cfg: MainConfig, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    if str(getattr(cfg.electron, "position_reference", "guiding_center")) != "instantaneous":
        return
    mask = np.isclose(np.asarray(t, dtype=float), float(cfg.simulation.starting_time_s), rtol=0.0, atol=1.0e-18)
    if not np.any(mask):
        return
    x0, y0 = _infer_initial_xy(cfg)
    z0 = float(cfg.electron.z0_m)
    x[mask] = float(x0)
    y[mask] = float(y0)
    z[mask] = float(z0)


# -----------------------------------------------------------------------------
# Initial-condition helpers
# -----------------------------------------------------------------------------


def _solve_gc_from_instantaneous_reduced_model(
    *,
    field: FieldMap,
    x_target_m: float,
    y_target_m: float,
    z_target_m: float,
    energy_eV: float,
    pitch_angle_deg: float,
    cyclotron_phase0_rad: float,
) -> tuple[float, float, float, float, float]:
    """
    Infer the initial guiding-center point when the user specifies the instantaneous
    electron position and the reduced model evaluates the local field at the guiding center.
    """
    G = np.asarray([float(x_target_m), float(y_target_m), float(z_target_m)], dtype=float)
    psi0 = float(cyclotron_phase0_rad)

    for _ in range(64):
        r_gc, phi_gc, z_gc = _cart_to_cyl(float(G[0]), float(G[1]), float(G[2]))
        B_ref = float(np.asarray(field.B(r_gc, z_gc)).reshape(()))
        Br_ref, Bphi_ref, Bz_ref = field.components(r_gc, z_gc)
        mu0 = _scalar_mu_from_pitch(float(energy_eV), float(pitch_angle_deg), B_ref)
        rho0 = _scalar_larmor_radius(B_ref, float(energy_eV), mu0)
        u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_ref, Bphi_ref, Bz_ref)
        delta = rho0 * (np.cos(psi0) * np.asarray(u1, dtype=float) + np.sin(psi0) * np.asarray(u2, dtype=float))
        G_new = np.asarray([x_target_m, y_target_m, z_target_m], dtype=float) - delta
        if np.linalg.norm(G_new - G) <= 1.0e-15 + 1.0e-12 * max(1.0, np.linalg.norm(G_new)):
            G = G_new
            break
        G = G_new

    r_gc, phi_gc, z_gc = _cart_to_cyl(float(G[0]), float(G[1]), float(G[2]))
    B_ref = float(np.asarray(field.B(r_gc, z_gc)).reshape(()))
    mu0 = _scalar_mu_from_pitch(float(energy_eV), float(pitch_angle_deg), B_ref)
    return r_gc, phi_gc, z_gc, B_ref, mu0




def _solve_gc_from_instantaneous_true_orbit_model(
    *,
    field: FieldMap,
    x_target_m: float,
    y_target_m: float,
    z_target_m: float,
    energy_eV: float,
    pitch_angle_deg: float,
    cyclotron_phase0_rad: float,
) -> tuple[float, float, float, float, float]:
    """
    Infer the initial guiding-center point for an instantaneous-position start in
    true-orbit mode.

    The user-provided pitch angle is interpreted at the actual electron position,
    because that is the specified physical particle state. The gyro-center orbit
    geometry is still a guiding-center quantity: the Larmor radius and local
    perpendicular basis are evaluated at the guiding center. This solves

        P0 = G0 + rho(B(G0), mu(P0)) [cos(psi) u1(G0) + sin(psi) u2(G0)].

    Using the instantaneous-point field for the orbit geometry makes the initial
    orbit slightly too stiff in a field-line-following trap and produces a small
    phase/position bias that accumulates over long tracks.
    """
    P = np.asarray([float(x_target_m), float(y_target_m), float(z_target_m)], dtype=float)
    psi0 = float(cyclotron_phase0_rad)

    r_p, _, z_p = _cart_to_cyl(float(P[0]), float(P[1]), float(P[2]))
    B_particle = float(np.asarray(field.B(r_p, z_p)).reshape(()))
    mu0 = _scalar_mu_from_pitch(float(energy_eV), float(pitch_angle_deg), B_particle)

    G = P.copy()
    for _ in range(80):
        r_gc, phi_gc, z_gc = _cart_to_cyl(float(G[0]), float(G[1]), float(G[2]))
        B_gc = float(np.asarray(field.B(r_gc, z_gc)).reshape(()))
        Br_gc, Bphi_gc, Bz_gc = field.components(r_gc, z_gc)
        rho0 = _scalar_larmor_radius(B_gc, float(energy_eV), mu0)
        u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_gc, Bphi_gc, Bz_gc)
        delta = rho0 * (np.cos(psi0) * np.asarray(u1, dtype=float) + np.sin(psi0) * np.asarray(u2, dtype=float))
        G_new = P - delta
        if np.linalg.norm(G_new - G) <= 1.0e-15 + 1.0e-12 * max(1.0, np.linalg.norm(G_new)):
            G = G_new
            break
        G = G_new

    r_gc, phi_gc, z_gc = _cart_to_cyl(float(G[0]), float(G[1]), float(G[2]))
    return r_gc, phi_gc, z_gc, B_particle, mu0


def _resolve_initial_orbit_state(cfg: MainConfig, field: FieldMap) -> InitialOrbitState:
    e = cfg.electron
    feat = cfg.features

    x0_in, y0_in = _infer_initial_xy(cfg)
    z0_in = float(e.z0_m)
    position_reference = str(getattr(e, "position_reference", "guiding_center"))
    E0 = float(e.energy_eV)
    theta = float(e.pitch_angle_deg)
    psi0 = float(e.cyclotron_phase0_rad)

    if position_reference == "guiding_center":
        r_gc0, phi_gc0, z_gc0 = _cart_to_cyl(float(x0_in), float(y0_in), float(z0_in))
        B_ref = float(np.asarray(field.B(r_gc0, z_gc0)).reshape(()))
        mu0 = _scalar_mu_from_pitch(E0, theta, B_ref)
        return InitialOrbitState(
            r_gc0_m=r_gc0,
            phi_gc0_rad=phi_gc0,
            z_gc0_m=z_gc0,
            x_target_m=float(x0_in),
            y_target_m=float(y0_in),
            z_target_m=float(z0_in),
            B_ref_T=B_ref,
            mu0_J_per_T=mu0,
        )

    if position_reference != "instantaneous":
        raise ValueError(
            f"Unknown electron.position_reference={position_reference!r}; expected 'guiding_center' or 'instantaneous'."
        )

    if bool(feat.include_true_orbit):
        r_gc0, phi_gc0, z_gc0, B_ref, mu0 = _solve_gc_from_instantaneous_true_orbit_model(
            field=field,
            x_target_m=float(x0_in),
            y_target_m=float(y0_in),
            z_target_m=float(z0_in),
            energy_eV=E0,
            pitch_angle_deg=theta,
            cyclotron_phase0_rad=psi0,
        )
    else:
        r_gc0, phi_gc0, z_gc0, B_ref, mu0 = _solve_gc_from_instantaneous_reduced_model(
            field=field,
            x_target_m=float(x0_in),
            y_target_m=float(y0_in),
            z_target_m=float(z0_in),
            energy_eV=E0,
            pitch_angle_deg=theta,
            cyclotron_phase0_rad=psi0,
        )
        return InitialOrbitState(
            r_gc0_m=r_gc0,
            phi_gc0_rad=phi_gc0,
            z_gc0_m=z_gc0,
            x_target_m=float(x0_in),
            y_target_m=float(y0_in),
            z_target_m=float(z0_in),
            B_ref_T=B_ref,
            mu0_J_per_T=mu0,
        )

    return InitialOrbitState(
        r_gc0_m=r_gc0,
        phi_gc0_rad=phi_gc0,
        z_gc0_m=z_gc0,
        x_target_m=float(x0_in),
        y_target_m=float(y0_in),
        z_target_m=float(z0_in),
        B_ref_T=float(B_ref),
        mu0_J_per_T=float(mu0),
    )


def _extract_axial_solution_arrays(sol: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def pick(*names: str) -> np.ndarray:
        for n in names:
            if n in sol:
                return np.asarray(sol[n], dtype=float)
        raise KeyError(f"AxialSolver.integrate() missing expected keys; tried {names}.")

    t_s = pick("t_s", "t")
    z_m = pick("z_m", "z")
    vpar = pick("vpar_m_per_s", "vpar")
    return t_s, z_m, vpar




def _make_cavity_interaction(cfg: MainConfig) -> CavityInteraction:
    cav_cfg = cfg.cavity
    cavity = Cavity(
        radius_m=float(cav_cfg.radius_m),
        length_m=float(cav_cfg.length_m),
        f0_hz=float(cav_cfg.f0_hz),
        Q=float(cav_cfg.Q),
    )
    coherent_response = bool(getattr(cav_cfg, "excitation_enabled", True)) and str(getattr(cav_cfg, "response_model", "time_evolution")) in {"time_evolution", "baseband_envelope"}
    # For coherent cavity response, compact dynamics uses the complex mode amplitude
    # only when back-reaction is requested.  One-way signal generation skips trajectory
    # feedback entirely and the final IQ path computes the cavity response once.
    coherent_back_reaction = coherent_response and bool(getattr(cav_cfg, "back_reaction_enabled", False))
    dynamics_cavity_enabled = bool(getattr(cav_cfg, "excitation_enabled", True)) and (coherent_back_reaction or not coherent_response)
    return CavityInteraction(
        cavity=cavity,
        excitation_enabled=dynamics_cavity_enabled,
        ringup_enabled=bool(getattr(cav_cfg, "ringup_enabled", True)) and dynamics_cavity_enabled,
        back_reaction_enabled=bool(getattr(cav_cfg, "back_reaction_enabled", True)) and dynamics_cavity_enabled,
        stimulated_back_reaction=bool(getattr(cav_cfg, "stimulated_back_reaction", True)) and dynamics_cavity_enabled,
        mode_volume_m3=getattr(cav_cfg, "mode_volume_m3", None),
        source_power_scale=float(getattr(cav_cfg, "source_power_scale", 1.0)),
        back_reaction_scale=float(getattr(cav_cfg, "back_reaction_scale", 1.0)),
        initial_energy_J=float(getattr(cav_cfg, "initial_stored_energy_J", 0.0)),
        response_model=str(getattr(cav_cfg, "response_model", "time_evolution")),
        lo_hz=(None if getattr(cfg.signal, "lo_hz", None) is None else float(cfg.signal.lo_hz)),
        initial_phase_rad=float(getattr(cav_cfg, "initial_cavity_phase_rad", 0.0)),
        cyclotron_phase0_rad=float(getattr(cfg.electron, "cyclotron_phase0_rad", 0.0)),
    )


def _coherent_cavity_response_enabled(cfg: MainConfig) -> bool:
    return bool(getattr(cfg.cavity, "excitation_enabled", False)) and str(getattr(cfg.cavity, "response_model", "time_evolution")) in {"time_evolution", "baseband_envelope"}


def _effective_energy_loss_model(cfg: MainConfig) -> tuple[str, str | None]:
    requested = str(getattr(cfg.dynamics, "energy_loss_model", "none"))
    if _coherent_cavity_response_enabled(cfg) and bool(getattr(cfg.mode_map, "vector_e_field_map", None)):
        if bool(getattr(cfg.cavity, "back_reaction_enabled", False)):
            return requested, (
                "coherent complex cavity back-reaction is enabled; the radiative state is "
                "updated from signed 2 Re(a* d) cavity work on the compact field-line grid"
            )
        if requested != "none":
            return "none", (
                "coherent cavity response is enabled but back-reaction is off; "
                "free-space/scalar energy-loss feedback is skipped so the trajectory is one-way"
            )
    return requested, None


def _cavity_power_arrays(
    *,
    cfg: MainConfig,
    interaction: CavityInteraction,
    B_T: np.ndarray,
    gamma: np.ndarray,
    mu_J_per_T: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    f_c_hz: np.ndarray,
    stored_energy_J: np.ndarray | None,
    mode_map: ModeMap,
    resonance: ResonanceCurve | None,
    phi_rad: np.ndarray | None = None,
    u1: np.ndarray | None = None,
    u2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Coherent cavity IQ is generated by the time-evolution/baseband response in
    # hopper.signal.synth from the complex vector-map drive.  Do not compute a
    # legacy scalar source-power diagnostic here: it is both physically incomplete
    # for stimulated work and was the dominant repeated vector-map cost in recent
    # profiles.  The signal path replaces these diagnostics with the coherent
    # cavity amplitude/drive on the requested readout grid.
    coherent_response = bool(getattr(cfg.cavity, "excitation_enabled", False)) and str(getattr(cfg.cavity, "response_model", "time_evolution")) in {"time_evolution", "baseband_envelope"}
    if coherent_response:
        base = np.asarray(f_c_hz, dtype=float)
        source_power = np.zeros_like(base, dtype=float)
        if stored_energy_J is None:
            stored_energy = np.zeros_like(base, dtype=float)
        else:
            stored_energy = np.maximum(np.asarray(stored_energy_J, dtype=float), 0.0)
        output_power = np.zeros_like(base, dtype=float)
        return source_power, stored_energy, output_power

    use_resonance = bool(cfg.features.include_resonance) or bool(interaction.excitation_enabled)
    if use_resonance and resonance is not None:
        response = np.asarray(resonance(f_c_hz), dtype=float)
    else:
        response = np.ones_like(np.asarray(f_c_hz, dtype=float), dtype=float)

    if bool(getattr(mode_map, "is_vector_e_field", False)):
        if phi_rad is None:
            phi_rad = np.zeros_like(np.asarray(r_m, dtype=float), dtype=float)
        if u1 is None or u2 is None:
            # Fallback to a horizontal local orbit basis.  Production calls pass the
            # guiding-center perpendicular basis, so this branch is for tests only.
            zeros = np.zeros_like(np.asarray(r_m, dtype=float), dtype=float)
            ones = np.ones_like(zeros)
            u1 = np.stack([np.cos(phi_rad), np.sin(phi_rad), zeros], axis=-1)
            u2 = np.stack([-np.sin(phi_rad), np.cos(phi_rad), zeros], axis=-1)
        drive = mode_map.gyro_drive_coupling_W_per_sqrt_J(  # type: ignore[attr-defined]
            r_gc_m=np.asarray(r_m, dtype=float),
            phi_gc_rad=np.asarray(phi_rad, dtype=float),
            z_gc_m=np.asarray(z_m, dtype=float),
            B_T=np.asarray(B_T, dtype=float),
            gamma=np.asarray(gamma, dtype=float),
            mu_J_per_T=np.asarray(mu_J_per_T, dtype=float),
            u1=np.asarray(u1, dtype=float),
            u2=np.asarray(u2, dtype=float),
        )
        source_power = np.asarray(interaction.source_power_from_drive_W(drive, response), dtype=float)
    else:
        # Coherent analytic TE011 fallback is handled in signal.synth.  Do not use
        # a scalar Larmor/Purcell power proxy for cavity diagnostics or back-reaction.
        source_power = np.zeros_like(np.asarray(f_c_hz, dtype=float), dtype=float)

    if stored_energy_J is None:
        # Compact tracks carry the true ring-up history; if no history is available
        # use the instantaneous steady-state output as a safe fallback.
        stored_energy = source_power * interaction.tau_energy_s
    else:
        stored_energy = np.maximum(np.asarray(stored_energy_J, dtype=float), 0.0)
    output_power = np.asarray(
        interaction.output_power_W(stored_energy, output_coupling_fraction=float(getattr(cfg.cavity, "output_coupling_fraction", 1.0))),
        dtype=float,
    )
    if not interaction.ringup_enabled:
        output_power = source_power
    return source_power, stored_energy, output_power

# -----------------------------------------------------------------------------
# Core track construction
# -----------------------------------------------------------------------------


def build_dynamic_track(
    cfg: MainConfig,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve,
) -> DynamicTrack:
    sim = cfg.simulation
    feat = cfg.features
    dyn = cfg.dynamics
    elec = cfg.electron

    const.configure_constants(cfg.physics.constants_preset)

    t0 = float(sim.starting_time_s)
    Tdur = float(sim.track_length_s)

    init = _resolve_initial_orbit_state(cfg, field)
    r0_m = float(init.r_gc0_m)
    phi0 = float(init.phi_gc0_rad)
    z0_gc_m = float(init.z_gc0_m)
    E0 = float(elec.energy_eV)
    mu0 = float(init.mu0_J_per_T)
    cavity_interaction = _make_cavity_interaction(cfg)
    cavity_energy0 = max(float(cavity_interaction.initial_energy_J), 0.0)

    axial_profile = AxialFieldProfile.from_field(field, r0_m=r0_m, z0_m=z0_gc_m)
    if hasattr(mode_map, "reset_counters"):
        mode_map.reset_counters()
    radiation_stats: dict[str, int] = {}

    solver = AxialSolver(
        field=field,
        r0_m=r0_m,
        E0_eV=E0,
        mu0_J_per_T=mu0,
        dt_max_s=float(dyn.dt_max_s),
        dt_min_s=float(dyn.dt_min_s),
        safety=float(dyn.safety),
        v_turn_threshold_c=float(dyn.v_turn_threshold_c),
        axial_profile=axial_profile,
    )

    axial_strategy = str(getattr(dyn, "axial_strategy", "direct"))
    energy_loss_model_requested = str(getattr(dyn, "energy_loss_model", "none"))
    energy_loss_model, energy_loss_override_reason = _effective_energy_loss_model(cfg)
    template_build = str(getattr(dyn, "template_build", "auto"))
    solver_info: dict[str, Any] = {
        "constants_preset": const.active_constants_name(),
        "axial_strategy": axial_strategy,
        "energy_loss_model_requested": energy_loss_model_requested,
        "energy_loss_model_effective": energy_loss_model,
        "energy_loss_override_reason": energy_loss_override_reason,
        "template_build": template_build if axial_strategy == "template_tiling" else None,
        "cyclotron_frequency_reference": _select_frequency_reference(cfg),
        "include_true_orbit": bool(feat.include_true_orbit),
        "include_gradB": bool(feat.include_gradB),
        "include_curvature_drift": bool(getattr(feat, "include_curvature_drift", False)),
        "cavity_back_reaction_enabled": bool(getattr(cfg.cavity, "back_reaction_enabled", False)),
        "cavity_back_reaction_model": (
            "complex_time_evolution_signed_work"
            if bool(getattr(cfg.cavity, "back_reaction_enabled", False))
            else "off"
        ),
        "cavity_back_reaction_lo_hz": (
            float(cfg.signal.lo_hz) if cfg.signal.lo_hz is not None else float(cfg.cavity.f0_hz)
        ),
    }

    LOGGER.info(
        "dynamics solver selected: axial_strategy=%s energy_loss_model=%s template_build=%s frequency_reference=%s constants=%s",
        axial_strategy,
        energy_loss_model,
        template_build if axial_strategy == "template_tiling" else "n/a",
        solver_info["cyclotron_frequency_reference"],
        solver_info["constants_preset"],
    )

    if axial_strategy == "direct":
        if energy_loss_model == "none":
            sol = solver.integrate(
                t0_s=t0,
                duration_s=Tdur,
                z0_m=z0_gc_m,
                vpar_sign=int(elec.vpar_sign),
                stop_at_turning=False,
            )
            t_s, z_gc_m, vz_gc_m_per_s = _extract_axial_solution_arrays(sol)
            E_eV = np.full_like(t_s, E0, dtype=float)
            mu_t = np.full_like(t_s, mu0, dtype=float)
            cavity_energy_t = np.full_like(t_s, cavity_energy0, dtype=float)
            solver_info["radiative_solver"] = "none_direct"
        elif energy_loss_model == "analytic":
            t_s, z_gc_m, vz_gc_m_per_s, E_eV, mu_t, cavity_energy_t = integrate_axial_track_energy_analytic(
                field=field,
                r0_m=r0_m,
                axial_profile=axial_profile,
                mu0_J_per_T=mu0,
                t0_s=t0,
                duration_s=Tdur,
                z0_m=z0_gc_m,
                vpar_sign=int(elec.vpar_sign),
                energy0_eV=E0,
                energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                energy_loss_scale=float(getattr(dyn, "energy_loss_scale", 1.0)),
                dt_max_s=float(dyn.dt_max_s),
                dt_min_s=float(dyn.dt_min_s),
                safety=float(dyn.safety),
                v_turn_threshold_c=float(dyn.v_turn_threshold_c),
                mode_map=mode_map,
                resonance=resonance,
                cavity_interaction=cavity_interaction,
                cavity_energy0_J=cavity_energy0,
                phi_gc0_rad=float(phi0),
                include_gradB=bool(feat.include_gradB),
                include_curvature_drift=bool(getattr(feat, "include_curvature_drift", False)),
                stats=radiation_stats,
            )
            solver_info["radiative_solver"] = "analytic_direct"
        else:
            raise ValueError(
                f"dynamics.energy_loss_model={energy_loss_model!r} is incompatible with "
                "dynamics.axial_strategy='direct'. Use 'none' or 'analytic'."
            )

    elif axial_strategy == "template_tiling":
        if energy_loss_model == "per_bounce":
            t_s, z_gc_m, vz_gc_m_per_s, E_eV, mu_t, cavity_energy_t = build_axial_track_energy_per_bounce(
                field=field,
                r0_m=r0_m,
                axial_profile=axial_profile,
                mu_J_per_T=mu0,
                t0_s=t0,
                duration_s=Tdur,
                z0_m=z0_gc_m,
                vpar_sign=int(elec.vpar_sign),
                energy0_eV=E0,
                energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                energy_loss_scale=float(getattr(dyn, "energy_loss_scale", 1.0)),
                dt_max_s=float(dyn.dt_max_s),
                dt_min_s=float(dyn.dt_min_s),
                safety=float(dyn.safety),
                v_turn_threshold_c=float(dyn.v_turn_threshold_c),
                template_build=template_build,
                template_return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
                template_max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
                template_min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
                mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
                mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
                mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
                mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
                mode_map=mode_map,
                resonance=resonance,
                cavity_interaction=cavity_interaction,
                cavity_energy0_J=cavity_energy0,
                per_bounce_block_bounces=int(getattr(dyn, "per_bounce_block_bounces", 1)),
                phi_gc0_rad=float(phi0),
                include_gradB=bool(feat.include_gradB),
                include_curvature_drift=bool(getattr(feat, "include_curvature_drift", False)),
                back_reaction_block_vectorized=bool(getattr(dyn, "back_reaction_block_vectorized", True)),
                back_reaction_block_max_rel_state_change=float(getattr(dyn, "back_reaction_block_max_rel_state_change", 5.0e-3)),
                back_reaction_max_updates_per_bounce=int(getattr(dyn, "back_reaction_max_updates_per_bounce", 96)),
                back_reaction_predictor_corrector=bool(getattr(dyn, "back_reaction_predictor_corrector", True)),
                multi_bounce_auto_max_bounces=int(getattr(dyn, "multi_bounce_auto_max_bounces", 8)),
                mirror_quadrature_min_theta_nodes=int(getattr(dyn, "mirror_quadrature_min_theta_nodes", 513)),
                mirror_quadrature_max_theta_nodes=int(getattr(dyn, "mirror_quadrature_max_theta_nodes", 8193)),
                mirror_template_max_period_rel_error=float(getattr(dyn, "mirror_template_max_period_rel_error", 1.0e-3)),
                mirror_template_max_phase_slip_rad=float(getattr(dyn, "mirror_template_max_phase_slip_rad", 5.0e-3)),
                mirror_template_moving_mirror_tol=float(getattr(dyn, "mirror_template_moving_mirror_tol", 1.0e-8)),
                stats=radiation_stats,
            )
            solver_info["radiative_solver"] = "per_bounce_template_tiling"
            solver_info["per_bounce_block_bounces"] = int(getattr(dyn, "per_bounce_block_bounces", 1))
            solver_info["template_build"] = str(template_build)
            solver_info["back_reaction_predictor_corrector"] = bool(getattr(dyn, "back_reaction_predictor_corrector", True))
            solver_info["back_reaction_max_updates_per_bounce"] = int(getattr(dyn, "back_reaction_max_updates_per_bounce", 0))
        else:
            tpl = build_bounce_template(
                solver,
                z0_m=float(z0_gc_m),
                vpar_sign0=int(elec.vpar_sign),
                duration_hint_s=min(Tdur, float(getattr(dyn, "template_duration_hint_s", 5e-5))),
                max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
                return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
                min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
                method=template_build,
                mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
                mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
                mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
                mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
                mirror_quadrature_min_theta_nodes=int(getattr(dyn, "mirror_quadrature_min_theta_nodes", 513)),
                mirror_quadrature_max_theta_nodes=int(getattr(dyn, "mirror_quadrature_max_theta_nodes", 8193)),
                mirror_template_max_period_rel_error=float(getattr(dyn, "mirror_template_max_period_rel_error", 1.0e-3)),
            )
            solver_info["bounce_period_s"] = float(tpl.bounce_period_s)
            solver_info["template_method"] = str(getattr(tpl, "method", template_build))
            if getattr(tpl, "z_turn_positive_m", None) is not None:
                solver_info["template_z_turn_positive_m"] = float(tpl.z_turn_positive_m)
            if getattr(tpl, "theta_node_count", None) is not None:
                solver_info["template_theta_node_count"] = int(tpl.theta_node_count)
            if getattr(tpl, "period_rel_error_estimate", None) is not None:
                solver_info["template_period_rel_error_estimate"] = float(tpl.period_rel_error_estimate)
            if energy_loss_model == "none":
                t_s, z_gc_m, vz_gc_m_per_s = tile_bounce_template_constant_energy(tpl, t0_s=t0, duration_s=Tdur)
                E_eV = np.full_like(t_s, E0, dtype=float)
                mu_t = np.full_like(t_s, mu0, dtype=float)
                cavity_energy_t = np.full_like(t_s, cavity_energy0, dtype=float)
                solver_info["radiative_solver"] = "none_template_tiling"
            elif energy_loss_model == "linear_fit":
                loss_rate_eV_per_s, mu_loss_rate = estimate_linear_radiative_state_rates(
                    field=field,
                    r0_m=r0_m,
                    axial_profile=axial_profile,
                    mu0_J_per_T=mu0,
                    energy0_eV=E0,
                    energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                    energy_loss_scale=float(getattr(dyn, "energy_loss_scale", 1.0)),
                    n_bounces=int(getattr(dyn, "energy_loss_fit_bounces", 10)),
                    dt_max_s=float(dyn.dt_max_s),
                    dt_min_s=float(dyn.dt_min_s),
                    safety=float(dyn.safety),
                    v_turn_threshold_c=float(dyn.v_turn_threshold_c),
                    z0_m=z0_gc_m,
                    vpar_sign=int(elec.vpar_sign),
                    template_build=template_build,
                    template_return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
                    template_max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
                    template_min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
                    mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
                    mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
                    mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
                    mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
                    mode_map=mode_map,
                    resonance=resonance,
                    cavity_interaction=cavity_interaction,
                    cavity_energy0_J=cavity_energy0,
                    phi_gc0_rad=float(phi0),
                    include_gradB=bool(feat.include_gradB),
                    include_curvature_drift=bool(getattr(feat, "include_curvature_drift", False)),
                    stats=radiation_stats,
                )
                t_s, z_gc_m, vz_gc_m_per_s, E_eV = tile_bounce_template_linear_energy(
                    tpl,
                    t0_s=t0,
                    duration_s=Tdur,
                    energy0_eV=E0,
                    loss_rate_eV_per_s=float(loss_rate_eV_per_s),
                    energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                )
                mu_t = np.maximum(mu0 - float(mu_loss_rate) * np.maximum(np.asarray(t_s, dtype=float) - t0, 0.0), 0.0)
                cavity_energy_t = np.full_like(t_s, cavity_energy0, dtype=float)
                solver_info["radiative_solver"] = "linear_fit_template_tiling"
                solver_info["linear_fit_loss_rate_eV_per_s"] = float(loss_rate_eV_per_s)
                solver_info["linear_fit_mu_loss_rate_J_per_T_s"] = float(mu_loss_rate)
            else:
                raise ValueError(
                    f"dynamics.energy_loss_model={energy_loss_model!r} is incompatible with "
                    "dynamics.axial_strategy='template_tiling'. Use 'none', 'per_bounce', or 'linear_fit'."
                )
    else:
        raise ValueError(f"Unknown dynamics.axial_strategy={axial_strategy!r}")

    (
        t_s,
        z_gc_m,
        vz_gc_m_per_s,
        E_eV,
        mu_t,
        cavity_energy_t,
        compact_compression_info,
    ) = _compress_axial_state_for_output(
        cfg,
        t_s,
        z_gc_m,
        vz_gc_m_per_s,
        E_eV,
        mu_t,
        cavity_energy_t,
    )
    solver_info.update(compact_compression_info)
    solver_info["radiation_counters"] = dict(radiation_stats)
    if hasattr(mode_map, "counter_snapshot"):
        solver_info["mode_map_counters_after_dynamics"] = mode_map.counter_snapshot()

    t = np.asarray(t_s, dtype=float)
    z_gc = np.asarray(z_gc_m, dtype=float)
    vz_gc = np.asarray(vz_gc_m_per_s, dtype=float)
    E_eV = np.asarray(E_eV, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    cavity_energy_t = np.asarray(cavity_energy_t, dtype=float)

    r_gc = np.asarray(axial_profile.r_at_z(z_gc), dtype=float)
    B_gc = np.asarray(axial_profile.B(z_gc), dtype=float)
    Br_gc, Bphi_gc, Bz_gc = axial_profile.components(z_gc)
    dBdr_gc, dBdz_gc = axial_profile.gradB(z_gc)
    bz_over_B_gc = np.asarray(axial_profile.bz_over_B(z_gc), dtype=float)
    br_over_B_gc = np.asarray(axial_profile.br_over_B(z_gc), dtype=float)
    bphi_over_B_gc = np.asarray(axial_profile.bphi_over_B(z_gc), dtype=float)

    gamma = np.asarray(gamma_beta_v_from_kinetic(E_eV)[0], dtype=float)
    vpar_abs = np.asarray(vpar_m_per_s_from_B(B_gc, E_eV, mu_t), dtype=float)
    parallel_sign = _parallel_sign_from_z_velocity(vz_gc, bz_over_B_gc, initial_sign=int(elec.vpar_sign))

    vr_gc = parallel_sign * vpar_abs * br_over_B_gc
    vphi_parallel = parallel_sign * vpar_abs * bphi_over_B_gc
    if feat.include_gradB:
        vphi_gradB = gradB_drift_vphi(
            mu_t,
            q_C=-const.E_CHARGE,
            Bmag_T=B_gc,
            Br_T=Br_gc,
            Bz_T=Bz_gc,
            dBdr_T_per_m=dBdr_gc,
            dBdz_T_per_m=dBdz_gc,
            gamma=gamma,
        )
    else:
        vphi_gradB = np.zeros_like(t)
    b_cross_kappa_phi = np.asarray(axial_profile.b_cross_kappa_phi(z_gc), dtype=float)
    if bool(getattr(feat, "include_curvature_drift", False)):
        vphi_curv = curvature_drift_vphi(
            gamma,
            vpar_abs,
            q_C=-const.E_CHARGE,
            Bmag_T=B_gc,
            b_cross_kappa_phi_per_m=b_cross_kappa_phi,
        )
    else:
        vphi_curv = np.zeros_like(t)
    vphi_gc_total = vphi_parallel + vphi_gradB + vphi_curv

    phi_gc = integrate_phi_from_vphi(t, vphi_gc_total, r_gc, float(phi0))
    x_gc = r_gc * np.cos(phi_gc)
    y_gc = r_gc * np.sin(phi_gc)
    vx_gc = vr_gc * np.cos(phi_gc) - np.sin(phi_gc) * vphi_gc_total
    vy_gc = vr_gc * np.sin(phi_gc) + np.cos(phi_gc) * vphi_gc_total

    psi0 = float(elec.cyclotron_phase0_rad)

    # The gyro-orbit geometry is a guiding-center construction.  Even when
    # include_true_orbit=True, the instantaneous orbit is built from the local
    # field at the guiding center; true-orbit mode only changes where B, coupling,
    # and delivered power are sampled after the position is reconstructed.  Feeding
    # the instantaneous orbit-point B back into the gyro phase creates a spurious
    # self-modulation and makes x/y oscillate too fast compared with a proper
    # guiding-center expansion.
    fc_gc = cyclotron_frequency_hz(B_gc, gamma, q_C=-const.E_CHARGE)
    phase_reference = _select_frequency_reference(cfg)
    phase_rate_hz = np.asarray(fc_gc, dtype=float)
    phase_rf = _phase_from_frequency(t, phase_rate_hz, phase_start_rad=0.0)

    u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_gc, Bphi_gc, Bz_gc)
    rho = larmor_radius_m_array(B_gc, E_eV, mu_t, q_C=-const.E_CHARGE)

    B_T = np.asarray(B_gc, dtype=float)
    r_for_coupling = r_gc
    z_for_coupling = z_gc
    z_true = z_gc.copy()
    v_cyc = np.zeros((t.size, 3), dtype=float)
    x = x_gc.copy()
    y = y_gc.copy()
    z = z_gc.copy()

    n_phase_iter = _true_orbit_phase_iterations(cfg) if phase_reference == "instantaneous" else 0
    for phase_iter in range(max(1, n_phase_iter + 1)):
        psi = phase_rf + psi0
        omega_c = 2.0 * np.pi * phase_rate_hz
        delta, v_cyc = _orbit_delta_and_velocity(
            rho_m=rho,
            omega_c_rad_per_s=omega_c,
            psi_rad=psi,
            u1=u1,
            u2=u2,
        )
        x = x_gc + delta[:, 0]
        y = y_gc + delta[:, 1]
        z = z_gc + delta[:, 2]
        _apply_instantaneous_anchor(cfg, t, x, y, z)

        if feat.include_true_orbit and phase_reference == "instantaneous":
            r_true = np.hypot(x, y)
            z_true = z
            B_T = np.asarray(field.B(r_true, z_true), dtype=float)
            phase_rate_hz = np.asarray(cyclotron_frequency_hz(B_T, gamma, q_C=-const.E_CHARGE), dtype=float)
            if phase_iter < n_phase_iter:
                phase_rf = _phase_from_frequency(t, phase_rate_hz, phase_start_rad=0.0)
            r_for_coupling = r_true
            z_for_coupling = z_true
        elif feat.include_true_orbit and phase_reference == "guiding_center":
            # Keep phase, exported B/fc, and signal frequency in the guiding-center frame.
            z_true = z
            B_T = np.asarray(B_gc, dtype=float)
            r_for_coupling = r_gc
            z_for_coupling = z_gc
        else:
            z_true = z
            B_T = np.asarray(B_gc, dtype=float)
            r_for_coupling = r_gc
            z_for_coupling = z_gc

    f_c = np.asarray(phase_rate_hz, dtype=float)
    if bool(getattr(mode_map, "is_vector_e_field", False)):
        # Vector cavity coupling is a gyro-averaged source around the guiding center.
        # It should use the guiding-center orbit geometry even when true-orbit fields
        # are exported for diagnostics.
        B_for_cavity = np.asarray(B_gc, dtype=float)
        r_for_cavity = np.asarray(r_gc, dtype=float)
        z_for_cavity = np.asarray(z_gc, dtype=float)
        f_for_cavity = np.asarray(fc_gc, dtype=float)
    else:
        B_for_cavity = np.asarray(B_T, dtype=float)
        r_for_cavity = np.asarray(r_for_coupling, dtype=float)
        z_for_cavity = np.asarray(z_for_coupling, dtype=float)
        f_for_cavity = np.asarray(f_c, dtype=float)
    source_power, cavity_energy_t, cavity_output_power = _cavity_power_arrays(
        cfg=cfg,
        interaction=cavity_interaction,
        B_T=B_for_cavity,
        gamma=gamma,
        mu_J_per_T=mu_t,
        r_m=r_for_cavity,
        z_m=z_for_cavity,
        f_c_hz=f_for_cavity,
        stored_energy_J=cavity_energy_t,
        mode_map=mode_map,
        resonance=resonance,
        phi_rad=phi_gc,
        u1=u1,
        u2=u2,
    )

    if feat.amplitude_mode == "sqrtP":
        amp = np.sqrt(np.maximum(cavity_output_power, 0.0))
    else:
        amp = np.maximum(cavity_output_power, 0.0)

    if feat.normalize_amp_at_t0 and amp.size > 0:
        amp = amp / max(float(amp[0]), 1e-30)

    vx = vx_gc + v_cyc[:, 0]
    vy = vy_gc + v_cyc[:, 1]
    vz = vz_gc + v_cyc[:, 2]

    return DynamicTrack(
        t=t,
        x=x,
        y=y,
        z=z_true,
        vx=vx,
        vy=vy,
        vz=vz,
        x_gc=x_gc,
        y_gc=y_gc,
        z_gc=z_gc,
        vx_gc=vx_gc,
        vy_gc=vy_gc,
        vz_gc=vz_gc,
        r_gc_m=r_gc,
        phi_gc_rad=phi_gc,
        parallel_sign=parallel_sign,
        b_cross_kappa_phi_per_m=b_cross_kappa_phi,
        f_c_hz=f_c,
        amp=amp,
        phase_rf=phase_rf,
        B_T=np.asarray(B_T, dtype=float),
        energy_eV=E_eV,
        mu_J_per_T=mu_t,
        axial_profile=axial_profile,
        cavity_energy_J=cavity_energy_t,
        cavity_power_W=cavity_output_power,
        cavity_source_power_W=source_power,
        cavity_work_power_W=None,
        solver_info=solver_info,
    )


# -----------------------------------------------------------------------------
# Output-grid sampling helpers
# -----------------------------------------------------------------------------


def sample_dynamic_track(
    cfg: MainConfig,
    track: DynamicTrack,
    *,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve | None,
    t_new: np.ndarray,
    phi_gc_start_rad: float | None = None,
    phase_rf_start_rad: float | None = None,
) -> DynamicTrack:
    """
    Sample a compact DynamicTrack on an arbitrary output grid.

    The RF-sampled reconstruction preserves the field-line guiding-center motion and rebuilds
    the cyclotron orbit in the local plane perpendicular to B. True-orbit mode then iterates
    the local field magnitude *and* orientation at the actual orbit point.
    """
    t_new = np.asarray(t_new, float)
    if t_new.ndim != 1:
        raise ValueError("t_new must be 1D")
    if t_new.size == 0:
        empty = np.asarray([], dtype=float)
        return DynamicTrack(
            t=empty,
            x=empty,
            y=empty,
            z=empty,
            vx=empty,
            vy=empty,
            vz=empty,
            x_gc=empty,
            y_gc=empty,
            z_gc=empty,
            vx_gc=empty,
            vy_gc=empty,
            vz_gc=empty,
            r_gc_m=empty,
            phi_gc_rad=empty,
            parallel_sign=empty,
            b_cross_kappa_phi_per_m=empty,
            f_c_hz=empty,
            amp=empty,
            phase_rf=empty,
            B_T=empty,
            energy_eV=empty,
            mu_J_per_T=empty,
            axial_profile=track.axial_profile,
        )

    feat = cfg.features
    cavity_interaction = _make_cavity_interaction(cfg)
    if t_new.size >= 2 and not np.all(np.diff(t_new) >= 0.0):
        raise ValueError("t_new must be monotonically increasing")

    def rs(y: np.ndarray) -> np.ndarray:
        return resample_linear(track.t, y, t_new)

    z_gc = rs(track.z_gc)
    parallel_sign = _step_resample(track.t, track.parallel_sign, t_new)
    energy_eV = np.maximum(rs(track.energy_eV), 0.0)
    mu_t = np.maximum(rs(track.mu_J_per_T), 0.0)
    if track.cavity_energy_J is not None:
        cavity_energy = np.maximum(rs(track.cavity_energy_J), 0.0)
    else:
        cavity_energy = None
    gamma = np.asarray(gamma_beta_v_from_kinetic(energy_eV)[0], dtype=float)

    axial_profile = track.axial_profile
    if axial_profile is not None:
        r_gc = np.asarray(axial_profile.r_at_z(z_gc), dtype=float)
        B_gc = np.asarray(axial_profile.B(z_gc), dtype=float)
        Br_gc, Bphi_gc, Bz_gc = axial_profile.components(z_gc)
        dBdr_gc, dBdz_gc = axial_profile.gradB(z_gc)
        b_cross_kappa_phi = np.asarray(axial_profile.b_cross_kappa_phi(z_gc), dtype=float)
    else:
        r_gc = rs(track.r_gc_m)
        B_gc = np.asarray(field.B(r_gc, z_gc), dtype=float)
        Br_gc, Bphi_gc, Bz_gc = field.components(r_gc, z_gc)
        dBdr_gc, dBdz_gc = field.gradB(r_gc, z_gc)
        b_cross_kappa_phi = rs(track.b_cross_kappa_phi_per_m)

    br_over_B_gc = np.asarray(Br_gc, dtype=float) / np.maximum(B_gc, 1.0e-300)
    bphi_over_B_gc = np.asarray(Bphi_gc, dtype=float) / np.maximum(B_gc, 1.0e-300)

    vpar_abs = np.asarray(vpar_m_per_s_from_B(B_gc, energy_eV, mu_t), dtype=float)
    vr_gc = parallel_sign * vpar_abs * br_over_B_gc
    vz_gc = parallel_sign * vpar_abs * (np.asarray(Bz_gc, dtype=float) / np.maximum(B_gc, 1.0e-300))
    vphi_parallel = parallel_sign * vpar_abs * bphi_over_B_gc
    if feat.include_gradB:
        vphi_gradB = gradB_drift_vphi(
            mu_t,
            q_C=-const.E_CHARGE,
            Bmag_T=B_gc,
            Br_T=Br_gc,
            Bz_T=Bz_gc,
            dBdr_T_per_m=dBdr_gc,
            dBdz_T_per_m=dBdz_gc,
            gamma=gamma,
        )
    else:
        vphi_gradB = np.zeros_like(t_new)
    if bool(getattr(feat, "include_curvature_drift", False)):
        vphi_curv = curvature_drift_vphi(
            gamma,
            vpar_abs,
            q_C=-const.E_CHARGE,
            Bmag_T=B_gc,
            b_cross_kappa_phi_per_m=b_cross_kappa_phi,
        )
    else:
        vphi_curv = np.zeros_like(t_new)
    vphi_gc_total = vphi_parallel + vphi_gradB + vphi_curv

    dphi_dt = np.zeros_like(t_new, dtype=float)
    np.divide(vphi_gc_total, r_gc, out=dphi_dt, where=np.abs(r_gc) >= 1.0e-14)
    phi_start = float(phi_gc_start_rad) if phi_gc_start_rad is not None else _scalar_resample(track.t, track.phi_gc_rad, float(t_new[0]))
    phi_gc = cumulative_trapezoid(dphi_dt, t_new, initial=phi_start)

    x_gc = r_gc * np.cos(phi_gc)
    y_gc = r_gc * np.sin(phi_gc)

    vx_gc = vr_gc * np.cos(phi_gc) - np.sin(phi_gc) * vphi_gc_total
    vy_gc = vr_gc * np.sin(phi_gc) + np.cos(phi_gc) * vphi_gc_total

    fc_gc = cyclotron_frequency_hz(B_gc, gamma, q_C=-const.E_CHARGE)
    phase_reference = _select_frequency_reference(cfg)
    phase_rate_hz = np.asarray(fc_gc, dtype=float)
    phase_start = float(phase_rf_start_rad) if phase_rf_start_rad is not None else _scalar_resample(track.t, track.phase_rf, float(t_new[0]))
    phase_rf = _phase_from_frequency(t_new, phase_rate_hz, phase_start_rad=phase_start)

    u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_gc, Bphi_gc, Bz_gc)
    rho = larmor_radius_m_array(B_gc, energy_eV, mu_t, q_C=-const.E_CHARGE)

    B_T = np.asarray(B_gc, dtype=float)
    r_for_coupling = r_gc
    z_for_coupling = z_gc
    z_true = z_gc.copy()
    v_cyc = np.zeros((t_new.size, 3), dtype=float)
    x = x_gc.copy()
    y = y_gc.copy()
    z = z_gc.copy()

    n_phase_iter = _true_orbit_phase_iterations(cfg) if phase_reference == "instantaneous" else 0
    for phase_iter in range(max(1, n_phase_iter + 1)):
        psi = phase_rf + float(cfg.electron.cyclotron_phase0_rad)
        omega_c = 2.0 * np.pi * phase_rate_hz
        delta, v_cyc = _orbit_delta_and_velocity(
            rho_m=rho,
            omega_c_rad_per_s=omega_c,
            psi_rad=psi,
            u1=u1,
            u2=u2,
        )
        x = x_gc + delta[:, 0]
        y = y_gc + delta[:, 1]
        z = z_gc + delta[:, 2]
        _apply_instantaneous_anchor(cfg, t_new, x, y, z)

        if feat.include_true_orbit and phase_reference == "instantaneous":
            r_true = np.hypot(x, y)
            z_true = z
            B_T = np.asarray(field.B(r_true, z_true), dtype=float)
            phase_rate_hz = np.asarray(cyclotron_frequency_hz(B_T, gamma, q_C=-const.E_CHARGE), dtype=float)
            if phase_iter < n_phase_iter:
                phase_rf = _phase_from_frequency(t_new, phase_rate_hz, phase_start_rad=phase_start)
            r_for_coupling = r_true
            z_for_coupling = z_true
        elif feat.include_true_orbit and phase_reference == "guiding_center":
            z_true = z
            B_T = np.asarray(B_gc, dtype=float)
            r_for_coupling = r_gc
            z_for_coupling = z_gc
        else:
            z_true = z
            B_T = np.asarray(B_gc, dtype=float)
            r_for_coupling = r_gc
            z_for_coupling = z_gc

    f_c = np.asarray(phase_rate_hz, dtype=float)
    if bool(getattr(mode_map, "is_vector_e_field", False)):
        B_for_cavity = np.asarray(B_gc, dtype=float)
        r_for_cavity = np.asarray(r_gc, dtype=float)
        z_for_cavity = np.asarray(z_gc, dtype=float)
        f_for_cavity = np.asarray(fc_gc, dtype=float)
    else:
        B_for_cavity = np.asarray(B_T, dtype=float)
        r_for_cavity = np.asarray(r_for_coupling, dtype=float)
        z_for_cavity = np.asarray(z_for_coupling, dtype=float)
        f_for_cavity = np.asarray(f_c, dtype=float)
    source_power, cavity_energy_sampled, cavity_output_power = _cavity_power_arrays(
        cfg=cfg,
        interaction=cavity_interaction,
        B_T=B_for_cavity,
        gamma=gamma,
        mu_J_per_T=mu_t,
        r_m=r_for_cavity,
        z_m=z_for_cavity,
        f_c_hz=f_for_cavity,
        stored_energy_J=cavity_energy,
        mode_map=mode_map,
        resonance=resonance,
        phi_rad=phi_gc,
        u1=u1,
        u2=u2,
    )

    if feat.amplitude_mode == "sqrtP":
        amp = np.sqrt(np.maximum(cavity_output_power, 0.0))
    else:
        amp = np.maximum(cavity_output_power, 0.0)

    if feat.normalize_amp_at_t0 and amp.size > 0:
        amp = amp / max(float(amp[0]), 1e-30)

    return DynamicTrack(
        t=t_new,
        x=x,
        y=y,
        z=z_true,
        vx=vx_gc + v_cyc[:, 0],
        vy=vy_gc + v_cyc[:, 1],
        vz=vz_gc + v_cyc[:, 2],
        x_gc=x_gc,
        y_gc=y_gc,
        z_gc=z_gc,
        vx_gc=vx_gc,
        vy_gc=vy_gc,
        vz_gc=vz_gc,
        r_gc_m=r_gc,
        phi_gc_rad=phi_gc,
        parallel_sign=parallel_sign,
        b_cross_kappa_phi_per_m=b_cross_kappa_phi,
        f_c_hz=f_c,
        amp=amp,
        phase_rf=phase_rf,
        B_T=np.asarray(B_T, dtype=float),
        energy_eV=energy_eV,
        mu_J_per_T=mu_t,
        axial_profile=axial_profile,
        cavity_energy_J=cavity_energy_sampled,
        cavity_power_W=cavity_output_power,
        cavity_source_power_W=source_power,
        cavity_work_power_W=None,
    )
