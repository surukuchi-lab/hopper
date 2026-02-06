from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from ..constants import E_CHARGE
from ..field.field_map import FieldMap
from .axial_solver import AxialSolver
from .kinematics import gamma_beta_v_from_kinetic, larmor_power_W_array, mu_from_pitch


@dataclass(frozen=True)
class BounceTemplate:
    """
    A single full-bounce axial template (z(t), v_par(t)) built at a reference energy.

    The template starts at t=0, z=z0 with the specified initial sign of v_par, and ends
    at the first return to z0 moving in the same direction after >= template_min_reflections
    reflections. This makes it tileable.
    """
    t_rel_s: np.ndarray
    z_m: np.ndarray
    vpar_ref_m_per_s: np.ndarray

    energy_ref_eV: float
    v_ref_m_per_s: float

    z0_m: float
    vpar_sign0: int

    @property
    def bounce_period_s(self) -> float:
        return float(self.t_rel_s[-1] - self.t_rel_s[0])


def _get_solver_energy_eV(solver: AxialSolver) -> float:
    """
    Compatibility helper: supports both new-style solver.energy_eV and legacy solver.E0_eV.
    """
    if hasattr(solver, "energy_eV"):
        return float(getattr(solver, "energy_eV"))
    if hasattr(solver, "E0_eV"):
        return float(getattr(solver, "E0_eV"))
    raise AttributeError("AxialSolver must provide either .energy_eV or .E0_eV")


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


def _sign_no_zeros(x: np.ndarray) -> np.ndarray:
    s = np.sign(x).astype(int)
    # Fill zeros with previous non-zero sign (forward fill), then backward fill for any leading zeros.
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    # If still all zeros, default to +1
    if np.all(s == 0):
        s[:] = 1
    return s


def _reflection_indices(vpar: np.ndarray) -> np.ndarray:
    s = _sign_no_zeros(vpar)
    flips = np.flatnonzero(s[1:] != s[:-1]) + 1
    return flips


def _find_bounce_end_index(
    t_s: np.ndarray,
    z_m: np.ndarray,
    vpar: np.ndarray,
    z0_m: float,
    vpar_sign0: int,
    z_tol_m: float,
    min_reflections: int,
) -> Optional[int]:
    flips = _reflection_indices(vpar)
    if len(flips) < min_reflections:
        return None

    s = _sign_no_zeros(vpar)
    s0 = 1 if vpar_sign0 >= 0 else -1

    start_search = flips[min_reflections - 1]
    dz = z_m - float(z0_m)

    candidates = np.flatnonzero((np.abs(dz) <= z_tol_m) & (s == s0))
    candidates = candidates[candidates >= start_search]
    if len(candidates) == 0:
        return None
    return int(candidates[0])


def _check_symmetry_about_zero(
    field: FieldMap,
    r0_m: float,
    z_samples_m: np.ndarray,
    rel_tol: float,
) -> bool:
    z = np.asarray(z_samples_m, dtype=float)
    Bp = field.B(r0_m, z)
    Bm = field.B(r0_m, -z)
    denom = np.maximum(np.abs(Bp), 1e-30)
    rel = np.max(np.abs(Bp - Bm) / denom)
    return bool(rel <= rel_tol)


def build_bounce_template(
    solver: AxialSolver,
    *,
    z0_m: float,
    vpar_sign0: int,
    duration_hint_s: float,
    max_duration_s: float,
    return_z_tol_m: float,
    min_reflections: int,
    method: str,  # accept "integrate", "mirror", or "auto"
    mirror_require_z0_near_zero: bool,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
) -> BounceTemplate:
    """
    Build one full bounce template.

    method:
      - "integrate": integrate until a full return to z0 is detected (robust)
      - "mirror": integrate to first turning point, then mirror about z=0 (fast, requires symmetry)
      - "auto": use mirror if checks pass, otherwise fall back to integrate
    """
    Eref = _get_solver_energy_eV(solver)
    _, _, vref = gamma_beta_v_from_kinetic(Eref)

    m = str(method).strip().lower()
    if m == "auto":
        can_mirror = True
        if mirror_require_z0_near_zero and abs(z0_m) > mirror_z0_tol_m:
            can_mirror = False
        if can_mirror and mirror_symmetry_check:
            ztest = np.linspace(0.0, 0.5 * abs(z0_m) + 0.01, int(mirror_symmetry_ncheck))
            if np.allclose(ztest, 0.0):
                ztest = np.linspace(0.0, 0.01, int(mirror_symmetry_ncheck))
            can_mirror = _check_symmetry_about_zero(solver.field, solver.r0_m, ztest, float(mirror_symmetry_rel_tol))
        m = "mirror" if can_mirror else "integrate"

    if m not in ("integrate", "mirror"):
        raise ValueError(f"template_build method must be 'integrate', 'mirror', or 'auto'; got {method!r}")

    if m == "mirror":
        if mirror_require_z0_near_zero and abs(z0_m) > mirror_z0_tol_m:
            raise ValueError(
                f"mirror template build requires |z0| <= {mirror_z0_tol_m} m, got z0={z0_m} m."
            )
        if mirror_symmetry_check:
            ztest = np.linspace(0.0, 0.5 * abs(z0_m) + 0.01, int(mirror_symmetry_ncheck))
            if np.allclose(ztest, 0.0):
                ztest = np.linspace(0.0, 0.01, int(mirror_symmetry_ncheck))
            if not _check_symmetry_about_zero(solver.field, solver.r0_m, ztest, float(mirror_symmetry_rel_tol)):
                raise ValueError(
                    "mirror template build requested, but field symmetry check about z=0 failed. "
                    "Use template_build='integrate' or template_build='auto'."
                )

        # Integrate until we have at least one reflection (first turning point)
        T = max(float(duration_hint_s), 10.0 * float(solver.dt_min_s))
        while T <= float(max_duration_s):
            sol = solver.integrate(t0_s=0.0, duration_s=T, z0_m=z0_m, vpar_sign=vpar_sign0)
            t, z, vpar = _extract_axial_solution_arrays(sol)

            flips = _reflection_indices(vpar)
            if len(flips) >= 1:
                idx_turn = int(flips[0])

                t_q = np.asarray(t[: idx_turn + 1], dtype=float)
                z_q = np.asarray(z[: idx_turn + 1], dtype=float)
                v_q = np.asarray(vpar[: idx_turn + 1], dtype=float)

                # Ensure the quarter segment starts at t=0
                t_q = t_q - float(t_q[0])
                Tq = float(t_q[-1])
                if Tq <= 0.0:
                    raise RuntimeError("Mirror template build found non-positive quarter duration.")

                # Construct a full bounce:
                # A: 0 -> +turn
                # B: +turn -> 0 (reverse A)
                # C: 0 -> -turn (negate z of A)
                # D: -turn -> 0 (reverse C)
                tA, zA, vA = t_q, z_q, v_q

                rev = t_q[::-1][1:]                 # times excluding the turning point, descending to 0
                inc = (Tq - rev)                    # ascending from dt..Tq

                tB = Tq + inc
                zB = z_q[::-1][1:]
                vB = -v_q[::-1][1:]

                tC = tB[-1] + t_q[1:]
                zC = -z_q[1:]
                vC = -v_q[1:]

                tD = tC[-1] + inc
                zD = (-z_q)[::-1][1:]
                vD = v_q[::-1][1:]

                t_full = np.concatenate([tA, tB, tC, tD], axis=0)
                z_full = np.concatenate([zA, zB, zC, zD], axis=0)
                v_full = np.concatenate([vA, vB, vC, vD], axis=0)

                # Safety: enforce monotonic time
                if np.any(np.diff(t_full) <= 0):
                    raise RuntimeError("Mirror template construction produced non-monotonic time axis.")

                return BounceTemplate(
                    t_rel_s=t_full,
                    z_m=z_full,
                    vpar_ref_m_per_s=v_full,
                    energy_ref_eV=float(Eref),
                    v_ref_m_per_s=float(vref),
                    z0_m=float(z0_m),
                    vpar_sign0=int(1 if vpar_sign0 >= 0 else -1),
                )

            T *= 2.0

        raise RuntimeError(
            f"Unable to build mirror bounce template (no turning point found) up to max_duration_s={max_duration_s}."
        )

    # ------------------------
    # integrate method (robust)
    # ------------------------
    T = max(float(duration_hint_s), 10.0 * float(solver.dt_min_s))
    while T <= float(max_duration_s):
        sol = solver.integrate(t0_s=0.0, duration_s=T, z0_m=z0_m, vpar_sign=vpar_sign0)
        t, z, vpar = _extract_axial_solution_arrays(sol)

        idx_end = _find_bounce_end_index(
            t, z, vpar,
            z0_m=float(z0_m),
            vpar_sign0=int(vpar_sign0),
            z_tol_m=float(return_z_tol_m),
            min_reflections=int(min_reflections),
        )
        if idx_end is not None and idx_end >= 2:
            t_tpl = t[: idx_end + 1] - t[0]
            z_tpl = z[: idx_end + 1]
            v_tpl = vpar[: idx_end + 1]
            return BounceTemplate(
                t_rel_s=np.asarray(t_tpl, dtype=float),
                z_m=np.asarray(z_tpl, dtype=float),
                vpar_ref_m_per_s=np.asarray(v_tpl, dtype=float),
                energy_ref_eV=float(Eref),
                v_ref_m_per_s=float(vref),
                z0_m=float(z0_m),
                vpar_sign0=int(1 if vpar_sign0 >= 0 else -1),
            )

        T *= 2.0

    raise RuntimeError(
        f"Unable to detect a full bounce return up to max_duration_s={max_duration_s}. "
        "Try increasing template_max_duration_s or loosening template_return_z_tol_m."
    )


def tile_bounce_template_constant_energy(
    tpl: BounceTemplate,
    *,
    t0_s: float,
    duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Repeat the bounce template in time to cover [t0, t0+duration].
    """
    T = tpl.bounce_period_s
    if T <= 0:
        raise ValueError("Bounce template has non-positive period.")

    n = int(np.ceil(duration_s / T)) + 1
    t_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []

    for i in range(n):
        t_seg = t0_s + tpl.t_rel_s + i * T
        z_seg = tpl.z_m
        v_seg = tpl.vpar_ref_m_per_s
        if i > 0:
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]
        t_list.append(t_seg)
        z_list.append(z_seg)
        v_list.append(v_seg)

    t = np.concatenate(t_list)
    z = np.concatenate(z_list)
    v = np.concatenate(v_list)

    t_end = t0_s + duration_s
    keep = t <= (t_end + 1e-15)
    return t[keep], z[keep], v[keep]


def build_tiled_axial_track_energy_per_bounce(
    tpl: BounceTemplate,
    *,
    field: FieldMap,
    r0_m: float,
    z0_m: float,
    pitch_angle_deg: float,
    t0_s: float,
    duration_s: float,
    energy0_eV: float,
    energy_floor_eV: float,
    energy_loss_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a long axial track by repeating a single-bounce template, but:
      - treat energy as constant within each bounce
      - update energy once per bounce using integrated radiated power
      - update the next bounce duration using the new speed (period scaling)
    """
    B_tpl = field.B(float(r0_m), np.asarray(tpl.z_m, dtype=float))
    B0 = float(field.B(float(r0_m), float(z0_m)))

    t_end = t0_s + duration_s

    t_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    v_out: list[np.ndarray] = []
    E_out: list[np.ndarray] = []

    E = float(energy0_eV)
    t_cursor = float(t0_s)

    t_rel = np.asarray(tpl.t_rel_s, dtype=float)
    vpar_ref = np.asarray(tpl.vpar_ref_m_per_s, dtype=float)
    T_ref = tpl.bounce_period_s

    while t_cursor < t_end - 1e-18:
        _, _, v_cur = gamma_beta_v_from_kinetic(E)
        scale_t = float(tpl.v_ref_m_per_s / v_cur)

        t_seg = t_cursor + scale_t * t_rel
        z_seg = tpl.z_m
        v_seg = vpar_ref * (v_cur / tpl.v_ref_m_per_s)

        if len(t_out) > 0:
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]

        if t_seg[-1] > t_end:
            keep = t_seg <= (t_end + 1e-15)
            t_seg = t_seg[keep]
            z_seg = z_seg[keep]
            v_seg = v_seg[keep]
            t_out.append(t_seg)
            z_out.append(z_seg)
            v_out.append(v_seg)
            E_out.append(np.full_like(t_seg, E, dtype=float))
            break

        t_out.append(t_seg)
        z_out.append(z_seg)
        v_out.append(v_seg)
        E_out.append(np.full_like(t_seg, E, dtype=float))

        mu = float(mu_from_pitch(E, float(pitch_angle_deg), B0))
        P = larmor_power_W_array(B_tpl, E, mu)
        E_loss_J = float(scale_t * np.trapz(P, t_rel))
        dE_eV = float(energy_loss_scale * (E_loss_J / E_CHARGE))

        E = max(float(energy_floor_eV), E - dE_eV)
        t_cursor = t_cursor + scale_t * T_ref

    t = np.concatenate(t_out) if len(t_out) else np.array([], dtype=float)
    z = np.concatenate(z_out) if len(z_out) else np.array([], dtype=float)
    v = np.concatenate(v_out) if len(v_out) else np.array([], dtype=float)
    E_arr = np.concatenate(E_out) if len(E_out) else np.array([], dtype=float)

    return t, z, v, E_arr


def estimate_linear_energy_loss_rate(
    tpl: BounceTemplate,
    *,
    field: FieldMap,
    r0_m: float,
    z0_m: float,
    pitch_angle_deg: float,
    energy0_eV: float,
    energy_floor_eV: float,
    energy_loss_scale: float,
    n_bounces: int,
) -> float:
    """
    Estimate a linear energy loss rate (eV/s) by simulating n_bounces with per-bounce updates
    and fitting a straight line E(t) = E0 - rate*t.
    """
    if n_bounces < 2:
        raise ValueError("n_bounces must be >= 2 to estimate a linear rate.")

    B_tpl = field.B(float(r0_m), np.asarray(tpl.z_m, dtype=float))
    B0 = float(field.B(float(r0_m), float(z0_m)))

    t_rel = np.asarray(tpl.t_rel_s, dtype=float)
    T_ref = tpl.bounce_period_s

    times = [0.0]
    energies = [float(energy0_eV)]

    E = float(energy0_eV)
    t_cursor = 0.0

    for _ in range(int(n_bounces)):
        _, _, v_cur = gamma_beta_v_from_kinetic(E)
        scale_t = float(tpl.v_ref_m_per_s / v_cur)

        mu = float(mu_from_pitch(E, float(pitch_angle_deg), B0))
        P = larmor_power_W_array(B_tpl, E, mu)
        E_loss_J = float(scale_t * np.trapz(P, t_rel))
        dE_eV = float(energy_loss_scale * (E_loss_J / E_CHARGE))

        E = max(float(energy_floor_eV), E - dE_eV)
        t_cursor = t_cursor + scale_t * T_ref

        times.append(t_cursor)
        energies.append(E)

    t = np.asarray(times, dtype=float)
    Evec = np.asarray(energies, dtype=float)

    A = np.vstack([np.ones_like(t), t]).T
    coeff, *_ = np.linalg.lstsq(A, Evec, rcond=None)
    b = float(coeff[1])
    loss_rate = max(0.0, -b)
    return loss_rate
