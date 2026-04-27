from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig
from ..constants import E_CHARGE
from ..field.field_map import FieldMap
from ..utils.math import cumulative_trapezoid, resample_linear
from .axial_profile import AxialFieldProfile
from .axial_solver import AxialSolver
from .drifts import curvature_drift_vphi, gradB_drift_vphi, integrate_phi_from_vphi
from .kinematics import (
    cyclotron_frequency_hz,
    gamma_beta_v_from_kinetic,
    larmor_power_W_array,
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
            q_C=-E_CHARGE,
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

    # Fallbacks for pathological cases; these are not expected for the repo's production fields but keep the basis construction numerically stable.
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

    t0 = float(sim.starting_time_s)
    Tdur = float(sim.track_length_s)

    init = _resolve_initial_orbit_state(cfg, field)
    r0_m = float(init.r_gc0_m)
    phi0 = float(init.phi_gc0_rad)
    z0_gc_m = float(init.z_gc0_m)
    E0 = float(elec.energy_eV)
    mu0 = float(init.mu0_J_per_T)

    axial_profile = AxialFieldProfile.from_field(field, r0_m=r0_m, z0_m=z0_gc_m)

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
    energy_loss_model = str(getattr(dyn, "energy_loss_model", "none"))

    if energy_loss_model == "analytic":
        t_s, z_gc_m, vz_gc_m_per_s, E_eV, mu_t = integrate_axial_track_energy_analytic(
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
        )
    elif energy_loss_model == "per_bounce":
        t_s, z_gc_m, vz_gc_m_per_s, E_eV, mu_t = build_axial_track_energy_per_bounce(
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
            template_build=str(getattr(dyn, "template_build", "auto")),
            template_return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
            template_max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
            template_min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
            mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
            mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
            mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
            mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
        )
    elif axial_strategy == "direct":
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
    elif axial_strategy == "template_tiling":
        tpl = build_bounce_template(
            solver,
            z0_m=float(z0_gc_m),
            vpar_sign0=int(elec.vpar_sign),
            duration_hint_s=min(Tdur, float(getattr(dyn, "template_duration_hint_s", 5e-5))),
            max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
            return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
            min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
            method=str(getattr(dyn, "template_build", "auto")),
            mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
            mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
            mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
            mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
        )
        if energy_loss_model == "none":
            t_s, z_gc_m, vz_gc_m_per_s = tile_bounce_template_constant_energy(tpl, t0_s=t0, duration_s=Tdur)
            E_eV = np.full_like(t_s, E0, dtype=float)
            mu_t = np.full_like(t_s, mu0, dtype=float)
        else:
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
                template_build=str(getattr(dyn, "template_build", "auto")),
                template_return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
                template_max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
                template_min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
                mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
                mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
                mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
                mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
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
    else:
        raise ValueError(f"Unknown dynamics.energy_loss_model={energy_loss_model!r}")

    t = np.asarray(t_s, dtype=float)
    z_gc = np.asarray(z_gc_m, dtype=float)
    vz_gc = np.asarray(vz_gc_m_per_s, dtype=float)
    E_eV = np.asarray(E_eV, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)

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
            q_C=-E_CHARGE,
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
            q_C=-E_CHARGE,
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
    fc_gc = cyclotron_frequency_hz(B_gc, gamma, q_C=-E_CHARGE)
    phase_rf = cumulative_trapezoid(2.0 * np.pi * fc_gc, t, initial=0.0)
    psi = phase_rf + psi0

    u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_gc, Bphi_gc, Bz_gc)
    rho = larmor_radius_m_array(B_gc, E_eV, mu_t, q_C=-E_CHARGE)
    omega_c = 2.0 * np.pi * fc_gc
    delta, v_cyc = _orbit_delta_and_velocity(rho_m=rho, omega_c_rad_per_s=omega_c, psi_rad=psi, u1=u1, u2=u2)
    x = x_gc + delta[:, 0]
    y = y_gc + delta[:, 1]
    z = z_gc + delta[:, 2]
    _apply_instantaneous_anchor(cfg, t, x, y, z)

    if feat.include_true_orbit:
        r_true = np.hypot(x, y)
        z_true = z
        B_T = np.asarray(field.B(r_true, z_true), dtype=float)
        r_for_coupling = r_true
        z_for_coupling = z_true
    else:
        z_true = z
        B_T = np.asarray(B_gc, dtype=float)
        r_for_coupling = r_gc
        z_for_coupling = z_gc

    f_c = cyclotron_frequency_hz(B_T, gamma, q_C=-E_CHARGE)
    P_e = np.asarray(larmor_power_W_array(B_T, E_eV, mu_t, q_C=-E_CHARGE), dtype=float)
    s_spatial = mode_map(r_for_coupling, z_for_coupling)
    Pin = P_e * (s_spatial ** 2)

    if feat.include_resonance:
        resp = resonance(f_c)
        Pin = Pin * (resp ** 2)

    if feat.amplitude_mode == "sqrtP":
        amp = np.sqrt(np.maximum(Pin, 0.0))
    else:
        amp = np.maximum(Pin, 0.0)

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
        )

    feat = cfg.features
    def rs(y: np.ndarray) -> np.ndarray:
        return resample_linear(track.t, y, t_new)

    r_gc = rs(track.r_gc_m)
    phi_gc = rs(track.phi_gc_rad)
    z_gc = rs(track.z_gc)
    parallel_sign = _step_resample(track.t, track.parallel_sign, t_new)
    phase_rf = rs(track.phase_rf)
    energy_eV = np.maximum(rs(track.energy_eV), 0.0)
    mu_t = np.maximum(rs(track.mu_J_per_T), 0.0)
    gamma = np.asarray(gamma_beta_v_from_kinetic(energy_eV)[0], dtype=float)

    x_gc = r_gc * np.cos(phi_gc)
    y_gc = r_gc * np.sin(phi_gc)

    B_gc = np.asarray(field.B(r_gc, z_gc), dtype=float)
    Br_gc, Bphi_gc, Bz_gc = field.components(r_gc, z_gc)
    dBdr_gc, dBdz_gc = field.gradB(r_gc, z_gc)
    br_over_B_gc = np.asarray(Br_gc, dtype=float) / np.maximum(B_gc, 1.0e-300)
    bphi_over_B_gc = np.asarray(Bphi_gc, dtype=float) / np.maximum(B_gc, 1.0e-300)

    vpar_abs = np.asarray(vpar_m_per_s_from_B(B_gc, energy_eV, mu_t), dtype=float)
    vr_gc = parallel_sign * vpar_abs * br_over_B_gc
    vz_gc = parallel_sign * vpar_abs * (np.asarray(Bz_gc, dtype=float) / np.maximum(B_gc, 1.0e-300))
    vphi_parallel = parallel_sign * vpar_abs * bphi_over_B_gc
    if feat.include_gradB:
        vphi_gradB = gradB_drift_vphi(
            mu_t,
            q_C=-E_CHARGE,
            Bmag_T=B_gc,
            Br_T=Br_gc,
            Bz_T=Bz_gc,
            dBdr_T_per_m=dBdr_gc,
            dBdz_T_per_m=dBdz_gc,
            gamma=gamma,
        )
    else:
        vphi_gradB = np.zeros_like(t_new)
    b_cross_kappa_phi = rs(track.b_cross_kappa_phi_per_m)
    if bool(getattr(feat, "include_curvature_drift", False)):
        vphi_curv = curvature_drift_vphi(
            gamma,
            vpar_abs,
            q_C=-E_CHARGE,
            Bmag_T=B_gc,
            b_cross_kappa_phi_per_m=b_cross_kappa_phi,
        )
    else:
        vphi_curv = np.zeros_like(t_new)
    vphi_gc_total = vphi_parallel + vphi_gradB + vphi_curv

    vx_gc = vr_gc * np.cos(phi_gc) - np.sin(phi_gc) * vphi_gc_total
    vy_gc = vr_gc * np.sin(phi_gc) + np.cos(phi_gc) * vphi_gc_total

    psi = phase_rf + float(cfg.electron.cyclotron_phase0_rad)

    # Preserve the same guiding-center gyro-orbit convention used for the compact track.
    # The RF/ROOT reconstruction samples B at the true orbit point only
    # after the orbit position has been built from the guiding-center field.
    fc_gc = cyclotron_frequency_hz(B_gc, gamma, q_C=-E_CHARGE)
    u1, u2, _ = _local_perp_basis_from_field(phi_gc, Br_gc, Bphi_gc, Bz_gc)
    rho = larmor_radius_m_array(B_gc, energy_eV, mu_t, q_C=-E_CHARGE)
    omega_c = 2.0 * np.pi * fc_gc
    delta, v_cyc = _orbit_delta_and_velocity(rho_m=rho, omega_c_rad_per_s=omega_c, psi_rad=psi, u1=u1, u2=u2)
    x = x_gc + delta[:, 0]
    y = y_gc + delta[:, 1]
    z = z_gc + delta[:, 2]
    _apply_instantaneous_anchor(cfg, t_new, x, y, z)

    if feat.include_true_orbit:
        r_true = np.hypot(x, y)
        z_true = z
        B_T = np.asarray(field.B(r_true, z_true), dtype=float)
        r_for_coupling = r_true
        z_for_coupling = z_true
    else:
        z_true = z
        B_T = B_gc
        r_for_coupling = r_gc
        z_for_coupling = z_gc

    f_c = cyclotron_frequency_hz(B_T, gamma, q_C=-E_CHARGE)
    P_e = np.asarray(larmor_power_W_array(B_T, energy_eV, mu_t, q_C=-E_CHARGE), dtype=float)
    s_spatial = mode_map(r_for_coupling, z_for_coupling)
    Pin = P_e * (s_spatial ** 2)

    if feat.include_resonance and resonance is not None:
        resp = resonance(f_c)
        Pin = Pin * (resp ** 2)

    if feat.amplitude_mode == "sqrtP":
        amp = np.sqrt(np.maximum(Pin, 0.0))
    else:
        amp = np.maximum(Pin, 0.0)

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
    )
