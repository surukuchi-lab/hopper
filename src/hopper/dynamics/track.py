from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig
from ..constants import E_CHARGE, M_E
from ..field.field_map import FieldMap
from ..utils.math import cumulative_trapezoid, resample_linear, unwrap_angle
from .axial_solver import AxialSolver
from .drifts import gradB_drift_vphi, integrate_phi_from_vphi
from .kinematics import (
    cyclotron_frequency_hz,
    gamma_beta_v_from_kinetic,
    larmor_power_W,
    larmor_power_W_array,
    larmor_radius_m,
    larmor_radius_m_array,
    mu_from_pitch,
)
from .template import (
    build_bounce_template,
    tile_bounce_template_constant_energy,
    build_tiled_axial_track_energy_per_bounce,
    estimate_linear_energy_loss_rate,
)


@dataclass
class DynamicTrack:
    """
    Non-uniform-time track arrays produced by the dynamics integrator.
    These arrays are typically resampled onto a uniform sampling grid by the signal node.
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

    f_c_hz: np.ndarray
    amp: np.ndarray
    phase_rf: np.ndarray  # RF phase integral (radians)
    B_T: np.ndarray  # instantaneous |B| experienced by the electron


def _infer_r_phi_from_xy(cfg: MainConfig) -> Tuple[float, float]:
    e = cfg.electron
    if e.x0_m is None or e.y0_m is None:
        return float(e.r0_m), float(e.phi0_rad)
    x = float(e.x0_m)
    y = float(e.y0_m)
    r = float(np.hypot(x, y))
    phi = float(np.arctan2(y, x))
    return r, phi


def _extract_axial_solution_arrays(sol: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Support both legacy keys ('t','z','vpar') and newer keys ('t_s','z_m','vpar_m_per_s').
    """
    def pick(*names: str) -> np.ndarray:
        for n in names:
            if n in sol:
                return np.asarray(sol[n], dtype=float)
        raise KeyError(f"AxialSolver.integrate() missing expected keys; tried {names}.")

    t_s = pick("t_s", "t")
    z_m = pick("z_m", "z")
    vpar = pick("vpar_m_per_s", "vpar")
    return t_s, z_m, vpar


def build_dynamic_track(
    cfg: MainConfig,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve,
) -> DynamicTrack:
    """
    Build a non-uniform dynamics track using:
      - adiabatic axial motion in B(r0,z) (direct integration OR template tiling)
      - cyclotron motion around guiding center (x,y)
      - optional grad-B azimuthal drift of the guiding center
      - amplitude model from radiated power proxy and mode-map coupling
      - optional cavity resonance amplitude weighting
    """
    sim = cfg.simulation
    feat = cfg.features
    dyn = cfg.dynamics
    elec = cfg.electron

    t0 = float(sim.starting_time_s)
    Tdur = float(sim.track_length_s)

    r0_m, phi0 = _infer_r_phi_from_xy(cfg)
    z0_m = float(elec.z0_m)

    # Initial conditions (also used for drift + radius/power proxies)
    B0 = float(field.B(float(r0_m), float(z0_m)))
    E0 = float(elec.energy_eV)
    theta = float(elec.pitch_angle_deg)
    mu0 = mu_from_pitch(E0, theta, B0)

    # ------------------------------------------------------------
    # Axial track generation: direct integration OR template tiling
    # ------------------------------------------------------------
    # Instantiate AxialSolver with new signature if available; fall back to legacy.
    try:
        solver = AxialSolver(
            field=field,
            r0_m=float(r0_m),
            energy_eV=float(elec.energy_eV),
            pitch_angle_deg=float(elec.pitch_angle_deg),
            dt_max_s=float(dyn.dt_max_s),
            dt_min_s=float(dyn.dt_min_s),
            safety=float(dyn.safety),
            v_turn_threshold_c=float(dyn.v_turn_threshold_c),
        )
    except TypeError:
        solver = AxialSolver(
            field=field,
            r0_m=float(r0_m),
            E0_eV=float(elec.energy_eV),
            mu0_J_per_T=float(mu0),
            dt_max_s=float(dyn.dt_max_s),
            dt_min_s=float(dyn.dt_min_s),
            safety=float(dyn.safety),
            v_turn_threshold_c=float(dyn.v_turn_threshold_c),
        )

    axial_strategy = str(getattr(dyn, "axial_strategy", "direct_integration"))
    energy_loss_model = str(getattr(dyn, "energy_loss_model", "none"))

    # Always build a bounce template if we need energy-loss-per-bounce or tiling.
    need_template = (axial_strategy == "template_tiling") or (energy_loss_model != "none")

    if not need_template:
        sol = solver.integrate(
            t0_s=t0,
            duration_s=Tdur,
            z0_m=float(z0_m),
            vpar_sign=int(elec.vpar_sign),
        )
        t_s, z_m, vpar_m_per_s = _extract_axial_solution_arrays(sol)
        E_eV = np.full_like(t_s, float(elec.energy_eV), dtype=float)
    else:
        tpl = build_bounce_template(
            solver,
            z0_m=float(z0_m),
            vpar_sign0=int(elec.vpar_sign),
            duration_hint_s=min(Tdur, 5e-5),
            max_duration_s=float(getattr(dyn, "template_max_duration_s", Tdur)),
            return_z_tol_m=float(getattr(dyn, "template_return_z_tol_m", 1e-6)),
            min_reflections=int(getattr(dyn, "template_min_reflections", 2)),
            method=str(getattr(dyn, "template_build", "auto")),
            mirror_require_z0_near_zero=bool(getattr(dyn, "mirror_require_z0_near_zero", False)),
            mirror_z0_tol_m=float(getattr(dyn, "mirror_z0_tol_m", 0.0)),
            mirror_symmetry_check=bool(getattr(dyn, "mirror_symmetry_check", True)),
            mirror_symmetry_rel_tol=float(getattr(dyn, "mirror_symmetry_rel_tol", 1e-3)),
            mirror_symmetry_ncheck=int(getattr(dyn, "mirror_symmetry_ncheck", 5)),
        )

        if energy_loss_model == "none":
            t_s, z_m, vpar_m_per_s = tile_bounce_template_constant_energy(
                tpl,
                t0_s=t0,
                duration_s=Tdur,
            )
            E_eV = np.full_like(t_s, float(elec.energy_eV), dtype=float)

        elif energy_loss_model == "per_bounce":
            t_s, z_m, vpar_m_per_s, E_eV = build_tiled_axial_track_energy_per_bounce(
                tpl,
                field=field,
                r0_m=float(r0_m),
                z0_m=float(z0_m),
                pitch_angle_deg=float(elec.pitch_angle_deg),
                t0_s=t0,
                duration_s=Tdur,
                energy0_eV=float(elec.energy_eV),
                energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                energy_loss_scale=float(getattr(dyn, "energy_loss_scale", 1.0)),
            )

        elif energy_loss_model == "linear_fit":
            loss_rate = estimate_linear_energy_loss_rate(
                tpl,
                field=field,
                r0_m=float(r0_m),
                z0_m=float(z0_m),
                pitch_angle_deg=float(elec.pitch_angle_deg),
                energy0_eV=float(elec.energy_eV),
                energy_floor_eV=float(getattr(dyn, "energy_floor_eV", 0.0)),
                energy_loss_scale=float(getattr(dyn, "energy_loss_scale", 1.0)),
                n_bounces=int(getattr(dyn, "energy_loss_fit_bounces", 10)),
            )

            t_s, z_m, vpar_m_per_s = tile_bounce_template_constant_energy(
                tpl,
                t0_s=t0,
                duration_s=Tdur,
            )
            E_eV = float(elec.energy_eV) - float(loss_rate) * (t_s - t0)
            E_eV = np.maximum(E_eV, float(getattr(dyn, "energy_floor_eV", 0.0)))

        else:
            raise ValueError(f"Unknown dynamics.energy_loss_model={energy_loss_model!r}")

    # ------------------------------------------------------------
    # Optional: cyclotron-phase-based internal stepping (expensive)
    # ------------------------------------------------------------
    time_step_strategy = str(getattr(dyn, "time_step_strategy", "axial_adaptive"))
    samples_per_turn = getattr(dyn, "samples_per_cyclotron_turn", None)

    if (samples_per_turn is not None) and (time_step_strategy != "axial_adaptive"):
        n_per = int(samples_per_turn)
        if n_per <= 1:
            raise ValueError("samples_per_cyclotron_turn must be >= 2.")

        # Use guiding-center B(r0, z) to define phase stepping.
        B_gc_tmp = field.B(float(r0_m), np.asarray(z_m, dtype=float))
        gamma_tmp = np.asarray(gamma_beta_v_from_kinetic(E_eV)[0], dtype=float)
        fc_tmp = (E_CHARGE * B_gc_tmp) / (2.0 * np.pi * gamma_tmp * M_E)

        # Cyclotron phase (monotone)
        phi = np.zeros_like(t_s, dtype=float)
        if len(t_s) >= 2:
            dt = (t_s[1:] - t_s[:-1])
            phi[1:] = np.cumsum(2.0 * np.pi * 0.5 * (fc_tmp[1:] + fc_tmp[:-1]) * dt)

        dphi_max = (2.0 * np.pi) / float(n_per)

        if time_step_strategy == "phase_uniform":
            phi_end = float(phi[-1]) if len(phi) else 0.0
            n_new = int(np.floor(phi_end / dphi_max)) + 1
            phi_u = dphi_max * np.arange(n_new, dtype=float)

            t_new = np.interp(phi_u, phi, t_s)
            z_new = np.interp(t_new, t_s, z_m)
            v_new = np.interp(t_new, t_s, vpar_m_per_s)

            idx = np.searchsorted(t_s, t_new, side="right") - 1
            idx = np.clip(idx, 0, len(E_eV) - 1)
            E_new = E_eV[idx]

        elif time_step_strategy == "phase_bounded":
            t_list = [float(t_s[0])]
            z_list = [float(z_m[0])]
            v_list = [float(vpar_m_per_s[0])]
            E_list = [float(E_eV[0])]

            for i in range(len(t_s) - 1):
                dphi = float(phi[i + 1] - phi[i])
                k = int(np.ceil(dphi / dphi_max))
                k = max(1, k)

                ti0, ti1 = float(t_s[i]), float(t_s[i + 1])
                for j in range(1, k + 1):
                    frac = j / k
                    tj = ti0 + (ti1 - ti0) * frac
                    t_list.append(tj)
                    z_list.append(float(z_m[i] + (z_m[i + 1] - z_m[i]) * frac))
                    v_list.append(float(vpar_m_per_s[i] + (vpar_m_per_s[i + 1] - vpar_m_per_s[i]) * frac))
                    E_list.append(float(E_eV[i]))  # piecewise-constant by "previous"

            t_new = np.asarray(t_list, dtype=float)
            z_new = np.asarray(z_list, dtype=float)
            v_new = np.asarray(v_list, dtype=float)
            E_new = np.asarray(E_list, dtype=float)

        else:
            raise ValueError(f"Unknown dynamics.time_step_strategy={time_step_strategy!r}")

        max_internal = int(getattr(dyn, "max_internal_points", 2_000_000))
        if len(t_new) > max_internal:
            raise RuntimeError(
                f"Internal time grid exploded to {len(t_new)} points. "
                "This is expected if you bound true cyclotron phase at GHz rates over long tracks. "
                "Increase dynamics.max_internal_points only if you really intend this."
            )

        t_s, z_m, vpar_m_per_s, E_eV = t_new, z_new, v_new, E_new

    # Adopt names used by the rest of this function
    t = np.asarray(t_s, dtype=float)
    z = np.asarray(z_m, dtype=float)
    vpar = np.asarray(vpar_m_per_s, dtype=float)
    E_eV = np.asarray(E_eV, dtype=float)

    # Guiding-center field along the axial track
    B_gc = field.B(float(r0_m), z)
    dBdr_gc, dBdz_gc = field.gradB(float(r0_m), z)
    Br_gc, _, Bz_gc = field.components(float(r0_m), z)

    # Guiding-center azimuthal drift
    if feat.include_gradB:
        vphi_gc = gradB_drift_vphi(
            mu0,
            q_C=-E_CHARGE,
            Bmag_T=B_gc,
            Br_T=Br_gc,
            Bz_T=Bz_gc,
            dBdr_T_per_m=dBdr_gc,
            dBdz_T_per_m=dBdz_gc,
        )
        phi_gc = integrate_phi_from_vphi(t, vphi_gc, float(r0_m), float(phi0))
    else:
        vphi_gc = np.zeros_like(t)
        phi_gc = np.full_like(t, float(phi0))

    x_gc = float(r0_m) * np.cos(phi_gc)
    y_gc = float(r0_m) * np.sin(phi_gc)
    z_gc = z.copy()

    vx_gc = -np.sin(phi_gc) * vphi_gc
    vy_gc = np.cos(phi_gc) * vphi_gc
    vz_gc = vpar.copy()

    # Energy-dependent gamma
    gamma = np.asarray(gamma_beta_v_from_kinetic(E_eV)[0], dtype=float)

    # Cyclotron phase for the *orbit geometry* (use GC field for consistency & to avoid implicit coupling)
    f_c_gc = cyclotron_frequency_hz(B_gc, gamma)
    psi0 = float(elec.cyclotron_phase0_rad)
    psi = cumulative_trapezoid(2.0 * np.pi * f_c_gc, t, initial=psi0)
    psi = unwrap_angle(psi)

    # Larmor radius
    try:
        rho = np.asarray(larmor_radius_m_array(mu0, gamma, B_gc, q_C=-E_CHARGE), dtype=float)
    except TypeError:
        B_arr = np.asarray(B_gc, dtype=float)
        g_arr = np.asarray(gamma, dtype=float)
        if g_arr.ndim == 0:
            g_arr = np.full_like(B_arr, float(g_arr), dtype=float)
        rho = np.array(
            [larmor_radius_m(mu0, float(gi), float(Bi), q_C=-E_CHARGE) for gi, Bi in zip(g_arr, B_arr)],
            dtype=float,
        )

    # True orbit around guiding center
    x = x_gc + rho * np.cos(psi)
    y = y_gc + rho * np.sin(psi)
    z_true = z.copy()

    # ------------------------------------------------------------
    # Instantaneous |B| experienced by the electron (stored as B_T)
    # ------------------------------------------------------------
    if feat.include_true_orbit:
        r_true = np.hypot(x, y)
        B_T = field.B(r_true, z_true)
        r_for_coupling = r_true
    else:
        B_T = B_gc
        r_for_coupling = np.full_like(t, float(r0_m), dtype=float)

    # Use B_T for RF phase + envelope power proxy (consistent with "instantaneous |B|")
    f_c = cyclotron_frequency_hz(B_T, gamma)

    try:
        P_e = np.asarray(larmor_power_W_array(B_T, E_eV, mu0), dtype=float)
    except TypeError:
        B_arr = np.asarray(B_T, dtype=float)
        E_arr = np.asarray(E_eV, dtype=float)
        if E_arr.ndim == 0:
            E_arr = np.full_like(B_arr, float(E_arr), dtype=float)
        P_e = np.array([larmor_power_W(float(Bi), float(Ei), mu0) for Bi, Ei in zip(B_arr, E_arr)], dtype=float)

    s_spatial = mode_map(r_for_coupling, z_true)
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

    phase_rf = cumulative_trapezoid(2.0 * np.pi * f_c, t, initial=0.0)
    phase_rf = unwrap_angle(phase_rf)

    # Velocities (GC drift + cyclotron motion)
    omega_c = 2.0 * np.pi * f_c_gc
    vx_cyc = -rho * np.sin(psi) * omega_c
    vy_cyc = rho * np.cos(psi) * omega_c

    vx = vx_gc + vx_cyc
    vy = vy_gc + vy_cyc
    vz = vz_gc.copy()

    return DynamicTrack(
        t=t,
        x=x, y=y, z=z_true,
        vx=vx, vy=vy, vz=vz,
        x_gc=x_gc, y_gc=y_gc, z_gc=z_gc,
        vx_gc=vx_gc, vy_gc=vy_gc, vz_gc=vz_gc,
        f_c_hz=f_c,
        amp=amp,
        phase_rf=phase_rf,
        B_T=np.asarray(B_T, dtype=float),
    )


def resample_dynamic_track(track: DynamicTrack, t_new: np.ndarray) -> DynamicTrack:
    """
    Resample a DynamicTrack onto a new (uniform) time grid.

    Note: phase_rf is unwrapped and linearly interpolated.
    """
    t_new = np.asarray(t_new, float)

    def rs(y: np.ndarray) -> np.ndarray:
        return resample_linear(track.t, y, t_new)

    phase = unwrap_angle(track.phase_rf)
    phase_rs = resample_linear(track.t, phase, t_new)

    B_T_rs = resample_linear(track.t, track.B_T, t_new)

    return DynamicTrack(
        t=t_new,
        x=rs(track.x), y=rs(track.y), z=rs(track.z),
        vx=rs(track.vx), vy=rs(track.vy), vz=rs(track.vz),
        x_gc=rs(track.x_gc), y_gc=rs(track.y_gc), z_gc=rs(track.z_gc),
        vx_gc=rs(track.vx_gc), vy_gc=rs(track.vy_gc), vz_gc=rs(track.vz_gc),
        f_c_hz=rs(track.f_c_hz),
        amp=rs(track.amp),
        phase_rf=phase_rs,
        B_T=B_T_rs,
    )