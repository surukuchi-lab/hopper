from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..constants import C0, MEC2_EV
from ..field.field_map import FieldMap
from .axial_profile import AxialFieldProfile
from .axial_solver import AxialSolver
from .kinematics import (
    beta_parallel2_from_B_gamma_mu,
    gamma_mu_after_radiation_step_fixed_upar,
    kinetic_energy_eV_from_gamma,
)
from .template import build_bounce_template


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _gamma_from_energy_eV(energy_eV: float) -> float:
    return float(1.0 + float(energy_eV) / MEC2_EV)


def _energy_eV_from_gamma(gamma: float) -> float:
    return float(np.asarray(kinetic_energy_eV_from_gamma(float(gamma))).reshape(()))


def _vpar_abs_from_B_gamma_mu(B_T: float, gamma: float, mu_J_per_T: float) -> float:
    beta_par2 = float(np.asarray(beta_parallel2_from_B_gamma_mu(float(B_T), float(gamma), float(mu_J_per_T))).reshape(()))
    return C0 * float(np.sqrt(max(beta_par2, 0.0)))


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
    z0_m: float,
    vpar_sign0: int,
    z_tol_m: float,
    min_reflections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Truncate a compact radiative track at the first full-bounce return to z0.

    A full bounce is the first return to z0 with the original z-velocity sign after at least ``min_reflections`` genuine mirror reflections.
    The endpoint is linearly interpolated on the z(t) crossing, which avoids the template-free per-bounce path accumulating a small phase bias from simply snapping to the nearest adaptive node.
    """
    t = np.asarray(t_s, dtype=float)
    z = np.asarray(z_m, dtype=float)
    v = np.asarray(vz_m_per_s, dtype=float)
    E = np.asarray(energy_eV, dtype=float)
    mu = np.asarray(mu_J_per_T, dtype=float)
    if not (t.ndim == z.ndim == v.ndim == E.ndim == mu.ndim == 1):
        raise ValueError("Compact radiative track arrays must be 1D.")
    if not (t.size == z.size == v.size == E.size == mu.size):
        raise ValueError("Compact radiative track arrays must have equal length.")
    if t.size < 3:
        return None

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
    for i in range(start_idx, t.size - 1):
        left = float(dz0[i])
        right = float(dz0[i + 1])

        if abs(left) <= tol and s[i] == s0:
            end = i + 1
            return t[:end], z[:end], v[:end], E[:end], mu[:end]

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

        t_cross = float(t[i] + frac * (t[i + 1] - t[i]))
        z_cross = float(z[i] + frac * (z[i + 1] - z[i]))
        E_cross = float(E[i] + frac * (E[i + 1] - E[i]))
        mu_cross = float(mu[i] + frac * (mu[i + 1] - mu[i]))

        t_out = np.concatenate([t[: i + 1], np.asarray([t_cross], dtype=float)])
        z_out = np.concatenate([z[: i + 1], np.asarray([z_cross], dtype=float)])
        v_out = np.concatenate([v[: i + 1], np.asarray([v_cross], dtype=float)])
        E_out = np.concatenate([E[: i + 1], np.asarray([E_cross], dtype=float)])
        mu_out = np.concatenate([mu[: i + 1], np.asarray([mu_cross], dtype=float)])
        keep = np.concatenate([[True], np.diff(t_out) > 0.0])
        return t_out[keep], z_out[keep], v_out[keep], E_out[keep], mu_out[keep]

    if abs(float(dz0[-1])) <= tol and s[-1] == s0:
        return t, z, v, E, mu
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
    max_steps: int = 5_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct adaptive axial integration with continuous radiative evolution of γ and μ.

    Turning points are now located by bisection in *time* for the coupled trial state.
    The accepted (t, z, γ, μ) state at the mirror is therefore self-consistent: the time advance, radiative loss, and mirror location all correspond to the same shortened step.
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
        )

    gamma_floor = _gamma_floor_from_energy_floor(float(energy_floor_eV))
    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), gamma_floor)
    mu = max(float(mu0_J_per_T), 0.0)
    sign = 1.0 if int(vpar_sign) >= 0 else -1.0
    z = float(z0_m)
    t = t0

    z_min = float(field.z[0])
    z_max = float(field.z[-1])
    c_turn = max(float(v_turn_threshold_c), 0.0) * C0
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

    for _ in range(int(max_steps)):
        if t >= t_end - 1.0e-24:
            break

        B_here = float(_B_along(field, axial_profile, float(r0_m), z))
        beta_par2_here = float(np.asarray(beta_parallel2_from_B_gamma_mu(B_here, gamma, mu)).reshape(()))
        if beta_par2_here < -1.0e-12:
            raise RuntimeError("Radiative axial integrator drifted outside the mirror condition.")

        vpar_abs = C0 * float(np.sqrt(max(beta_par2_here, 0.0)))
        if vpar_abs < c_turn and c_turn > 0.0:
            dt = float(safety) * float(dt_max_s) * (vpar_abs / (c_turn + 1.0e-300))
            dt = max(float(dt_min_s), dt)
        else:
            dt = float(dt_max_s)
        dt = min(max(float(dt_min_s), dt), t_end - t)

        z_speed_factor_here = float(np.asarray(_bz_over_B(axial_profile, z)).reshape(()))

        if vpar_abs <= 0.0 or beta_par2_here <= turn_tol:
            # Move an infinitesimal distance inward along the reflected branch so the next accepted step starts from a strictly allowed state.
            nudge = max(1.0e-12, 1.0e-9 * max(abs(z), 1.0))
            z_dir = np.sign(sign * z_speed_factor_here) if abs(z_speed_factor_here) > 0.0 else sign
            z_candidate = float(np.clip(z + z_dir * nudge, z_min, z_max))
            beta_candidate = float(
                np.asarray(beta_parallel2_from_B_gamma_mu(float(_B_along(field, axial_profile, float(r0_m), z_candidate)), gamma, mu)).reshape(())
            )
            if beta_candidate > 0.0:
                z = z_candidate
                continue

        def trial_state(dt_trial: float) -> tuple[float, float, float, float]:
            z_trial = z + sign * vpar_abs * z_speed_factor_here * float(dt_trial)
            z_mid = 0.5 * (z + z_trial)
            B_mid = float(_B_along(field, axial_profile, float(r0_m), z_mid))

            if gamma <= gamma_floor * (1.0 + 1.0e-14) or float(energy_loss_scale) == 0.0:
                gamma_trial = gamma
                mu_trial = mu
            else:
                gamma_trial, mu_trial = gamma_mu_after_radiation_step_fixed_upar(
                    gamma,
                    mu,
                    B_mid,
                    float(dt_trial),
                    energy_loss_scale=float(energy_loss_scale),
                    gamma_floor=float(gamma_floor),
                )
                gamma_trial = float(np.asarray(gamma_trial).reshape(()))
                mu_trial = float(np.asarray(mu_trial).reshape(()))

            B_trial = float(_B_along(field, axial_profile, float(r0_m), z_trial))
            beta_par2_trial = float(np.asarray(beta_parallel2_from_B_gamma_mu(B_trial, gamma_trial, mu_trial)).reshape(()))
            return z_trial, gamma_trial, mu_trial, beta_par2_trial

        z_trial, gamma_trial, mu_trial, beta_par2_trial = trial_state(dt)

        if beta_par2_trial >= -turn_tol:
            turning_step = beta_par2_trial <= turn_tol
            t_new = t + dt
            z_new = float(z_trial)
            gamma_new = float(gamma_trial)
            mu_new = float(mu_trial)
        else:
            # The mirror lies inside the proposed interval. Locate the turning time with a consistent time/gamma/mu bisection instead of advancing radiation for the full dt and then projecting only z.
            dt_lo = 0.0
            z_lo = z
            gamma_lo = gamma
            mu_lo = mu
            for _ in range(80):
                dt_mid = 0.5 * (dt_lo + dt)
                z_mid, gamma_mid, mu_mid, beta_mid = trial_state(dt_mid)
                if beta_mid >= 0.0:
                    dt_lo = dt_mid
                    z_lo = float(z_mid)
                    gamma_lo = float(gamma_mid)
                    mu_lo = float(mu_mid)
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

        t_list.append(t)
        z_list.append(z)
        v_list.append(v_new)
        E_list.append(_energy_eV_from_gamma(gamma))
        mu_list.append(mu)

    if t_list[-1] < t_end - 1.0e-18:
        raise RuntimeError(
            "Radiative axial integrator did not reach the requested end time. "
            "Increase max_steps or inspect the mirror condition / field-map bounds."
        )

    return (
        np.asarray(t_list, dtype=float),
        np.asarray(z_list, dtype=float),
        np.asarray(v_list, dtype=float),
        np.asarray(E_list, dtype=float),
        np.asarray(mu_list, dtype=float),
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    guess = max(float(bounce_period_guess_s), 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))
    factors = (1.10, 1.35, 1.70, 2.20)
    energy0_eV = _energy_eV_from_gamma(float(gamma0))

    for fac in factors:
        duration = min(float(template_max_duration_s), fac * guess)
        t_seg, z_seg, v_seg, E_seg, mu_seg = integrate_axial_track_energy_analytic(
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
        )
        truncated = _truncate_track_at_bounce_return(
            t_s=t_seg,
            z_m=z_seg,
            vz_m_per_s=v_seg,
            energy_eV=E_seg,
            mu_J_per_T=mu_seg,
            z0_m=float(z0_m),
            vpar_sign0=int(vpar_sign),
            z_tol_m=float(template_return_z_tol_m),
            min_reflections=int(template_min_reflections),
        )
        if truncated is not None:
            return truncated

    raise RuntimeError(
        "Failed to bracket a full radiative bounce return within the configured maximum duration. "
        "Increase dynamics.template_max_duration_s or inspect the mirror condition."
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast bouncewise radiative correction using a direct 1D cached-field-line solve per bounce.

    The previous implementation evolved (γ, μ) on a *fixed spatial bounce template* and only
    corrected the time-of-flight inside each segment. That improved axial timing, but the
    template geometry itself could still extend slightly beyond the moving mirror implied by the
    updated (γ, μ). The resulting over-extended z(t) produced the kind of slowly growing field /
    frequency discrepancy seen against the high-fidelity tracks.

    The new implementation keeps the computationally cheap cached 1D field-line model and the
    bouncewise structure, but integrates each bounce directly with the same radiative ODE used by
    analytic mode.
    """
    t_end = float(t0_s) + float(duration_s)
    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), _gamma_floor_from_energy_floor(float(energy_floor_eV)))
    mu = max(float(mu_J_per_T), 0.0)
    t_cursor = float(t0_s)
    bounce_period_guess_s: float | None = None

    t_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    v_out: list[np.ndarray] = []
    E_out: list[np.ndarray] = []
    mu_out: list[np.ndarray] = []

    while t_cursor < t_end - 1.0e-18:
        remaining = t_end - t_cursor
        E_cur = _energy_eV_from_gamma(gamma)

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
            )

        is_first = len(t_out) == 0

        if remaining <= 1.05 * max(float(bounce_period_guess_s), 8.0 * float(dt_max_s)):
            t_seg, z_seg, v_seg, E_seg, mu_seg = integrate_axial_track_energy_analytic(
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
            )
            if not is_first:
                t_seg = t_seg[1:]
                z_seg = z_seg[1:]
                v_seg = v_seg[1:]
                E_seg = E_seg[1:]
                mu_seg = mu_seg[1:]
            t_out.append(t_seg)
            z_out.append(z_seg)
            v_out.append(v_seg)
            E_out.append(E_seg)
            mu_out.append(mu_seg)
            break

        t_seg, z_seg, v_seg, E_seg, mu_seg = _integrate_one_radiative_bounce(
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
        )

        if not is_first:
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]
            E_seg = E_seg[1:]
            mu_seg = mu_seg[1:]

        t_out.append(t_seg)
        z_out.append(z_seg)
        v_out.append(v_seg)
        E_out.append(E_seg)
        mu_out.append(mu_seg)

        gamma = _gamma_from_energy_eV(float(E_seg[-1]))
        mu = float(mu_seg[-1])
        t_cursor = float(t_seg[-1])
        bounce_period_guess_s = max(float(t_seg[-1] - t_seg[0]), 8.0 * float(dt_max_s), 10.0 * float(dt_min_s))

    if not t_out:
        return (
            np.asarray([float(t0_s)], dtype=float),
            np.asarray([float(z0_m)], dtype=float),
            np.asarray([0.0], dtype=float),
            np.asarray([float(energy0_eV)], dtype=float),
            np.asarray([float(mu_J_per_T)], dtype=float),
        )

    return (
        np.concatenate(t_out),
        np.concatenate(z_out),
        np.concatenate(v_out),
        np.concatenate(E_out),
        np.concatenate(mu_out),
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
) -> tuple[float, float]:
    """Estimate linear dE/dt and dμ/dt from bouncewise direct evolving-(γ, μ) updates."""
    if int(n_bounces) < 2:
        raise ValueError("n_bounces must be >= 2 to estimate linear rates.")

    gamma = max(_gamma_from_energy_eV(float(energy0_eV)), _gamma_floor_from_energy_floor(float(energy_floor_eV)))
    mu = max(float(mu0_J_per_T), 0.0)
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
        t_seg, _, _, E_seg, mu_seg = _integrate_one_radiative_bounce(
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
        )
        gamma = _gamma_from_energy_eV(float(E_seg[-1]))
        mu = float(mu_seg[-1])
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
