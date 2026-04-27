from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .axial_solver import AxialSolver
from .kinematics import gamma_beta_v_from_kinetic

TemplateBuildMethod = Literal["auto", "integrate", "mirror"]


@dataclass(frozen=True)
class BounceTemplate:
    """
    A single full-bounce axial template (z(t), v_par(t)) built at a reference energy.

    For tiling, the template is constructed to start at (t=0, z=z0) with the specified initial
    v_par sign, and to end at the first return to z0 moving in the same direction.

    In "mirror" mode, we integrate only a quarter bounce (center -> turning point) and assemble a full bounce via symmetry.
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


# -----------------------------------------------------------------------------
# Template construction helpers
# -----------------------------------------------------------------------------

def _sign_no_zeros(x: np.ndarray) -> np.ndarray:
    s = np.sign(x).astype(int)
    # Fill zeros with previous non-zero sign (forward fill), then backward fill for leading zeros.
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    if np.all(s == 0):
        s[:] = 1
    return s


def _reflection_indices(vpar: np.ndarray, *, min_dz_m: float, z_m: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Indices where v_par sign flips. Optionally reject pathological "chatter" flips by requiring
    a minimum axial excursion (|Δz| > min_dz_m) between accepted flips.
    """
    s = _sign_no_zeros(vpar)
    flips = np.flatnonzero(s[1:] != s[:-1]) + 1
    if z_m is None or min_dz_m <= 0.0 or len(flips) <= 1:
        return flips

    z = np.asarray(z_m, dtype=float)
    keep = [int(flips[0])]
    last = int(flips[0])
    for f in flips[1:]:
        if abs(float(z[int(f)]) - float(z[last])) >= float(min_dz_m):
            keep.append(int(f))
            last = int(f)
    return np.asarray(keep, dtype=int)


def _find_bounce_end_index(
    t_s: np.ndarray,
    z_m: np.ndarray,
    vpar: np.ndarray,
    *,
    z0_m: float,
    vpar_sign0: int,
    z_tol_m: float,
    min_reflections: int,
) -> Optional[int]:
    """
    Return the first index that qualifies as a "full bounce return":
      - |z - z0| <= z_tol_m
      - v_par sign matches initial sign
      - after at least min_reflections reflections
      - AND the trajectory has actually left z0 by a nontrivial amount (to suppress chatter).
    """
    z = np.asarray(z_m, dtype=float)
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(vpar, dtype=float)
    if len(t) < 3:
        return None

    # Require that at least one "real" excursion occurred.
    excursion_needed = max(10.0 * float(z_tol_m), 1e-9)

    flips = _reflection_indices(v, min_dz_m=excursion_needed, z_m=z)
    if len(flips) < int(min_reflections):
        return None

    s = _sign_no_zeros(v)
    s0 = 1 if int(vpar_sign0) >= 0 else -1

    start_search = int(flips[int(min_reflections) - 1])
    dz = z - float(z0_m)

    candidates = np.flatnonzero((np.abs(dz) <= float(z_tol_m)) & (s == s0))
    candidates = candidates[candidates >= start_search]
    if len(candidates) == 0:
        return None

    # Ensure the candidate corresponds to a true excursion away from z0.
    for idx in candidates:
        if np.max(np.abs(dz[: idx + 1])) >= excursion_needed:
            return int(idx)
    return None


def _check_symmetry_about_zero(
    solver: AxialSolver,
    z_samples_m: np.ndarray,
    rel_tol: float,
) -> bool:
    z = np.asarray(z_samples_m, dtype=float)
    Bp = solver.B_at_z(z)
    Bm = solver.B_at_z(-z)
    denom = np.maximum(np.abs(Bp), 1e-30)
    rel = np.max(np.abs(Bp - Bm) / denom)
    return bool(rel <= float(rel_tol))


def _build_full_bounce_from_center_to_positive_turn(
    t_q: np.ndarray,
    z_q: np.ndarray,
    v_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a center-referenced full bounce from a center -> +turn integration.

    For a midplane-symmetric trap, the magnetic-field-line profile satisfies B(z)=B(-z)
    along a given field line, so a single center-to-turn leg determines the full axial
    period.  The returned template starts at z=0 moving toward +z and ends one full
    period later at z=0 with the same sign.
    """
    t_q = np.asarray(t_q, dtype=float) - float(np.asarray(t_q, dtype=float)[0])
    z_q = np.asarray(z_q, dtype=float)
    v_q = np.asarray(v_q, dtype=float)
    if t_q.size < 3:
        raise ValueError("Center-to-turn segment is too short to mirror.")
    Tq = float(t_q[-1])
    if Tq <= 0.0:
        raise ValueError("Center-to-turn segment has non-positive duration.")

    # A: 0 -> +turn
    tA, zA, vA = t_q, z_q, v_q

    # B: +turn -> 0.  Exclude the turning point to avoid duplicate time.
    idxB = np.arange(t_q.size - 2, -1, -1)
    tB = Tq + (Tq - t_q[idxB])
    zB = z_q[idxB]
    vB = -v_q[idxB]

    # C: 0 -> -turn.  Exclude the center point to avoid duplicate time.
    idxC = np.arange(1, t_q.size, 1)
    tC = tB[-1] + t_q[idxC]
    zC = -z_q[idxC]
    vC = -v_q[idxC]

    # D: -turn -> 0.  Exclude the negative turning point.
    idxD = np.arange(t_q.size - 2, -1, -1)
    tD = tC[-1] + (Tq - t_q[idxD])
    zD = (-z_q)[idxD]
    vD = v_q[idxD]

    t_full = np.concatenate([tA, tB, tC, tD])
    z_full = np.concatenate([zA, zB, zC, zD])
    v_full = np.concatenate([vA, vB, vC, vD])

    keep = np.concatenate([[True], np.diff(t_full) > 0.0])
    t_full = t_full[keep]
    z_full = z_full[keep]
    v_full = v_full[keep]

    if not np.all(np.diff(t_full) > 0.0):
        raise RuntimeError("Internal error: mirrored template time axis is not strictly increasing.")
    return t_full, z_full, v_full


def _periodic_interp(t_full: np.ndarray, y_full: np.ndarray, phase_s: np.ndarray | float) -> np.ndarray:
    """Linear interpolation on a periodic template whose endpoint duplicates the start."""
    t = np.asarray(t_full, dtype=float)
    y = np.asarray(y_full, dtype=float)
    phase = np.asarray(phase_s, dtype=float)
    T = float(t[-1])
    if T <= 0.0:
        raise ValueError("Cannot interpolate a non-positive-period template.")
    ph = np.mod(phase, T)
    ph = np.where(np.isclose(ph, 0.0, rtol=0.0, atol=1.0e-18) & (phase > 0.0), T, ph)
    return np.asarray(np.interp(ph, t, y), dtype=float)


def _phase_for_z_and_sign(
    t_full: np.ndarray,
    z_full: np.ndarray,
    v_full: np.ndarray,
    *,
    z0_m: float,
    desired_zdot_sign: int,
    z_tol_m: float,
) -> float:
    """Find the bounce phase matching an arbitrary starting z and dz/dt sign."""
    t = np.asarray(t_full, dtype=float)
    z = np.asarray(z_full, dtype=float)
    v = np.asarray(v_full, dtype=float)
    T = float(t[-1])
    z0 = float(z0_m)
    s_des = 1 if int(desired_zdot_sign) >= 0 else -1

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    tol = max(float(z_tol_m), 1.0e-12, 1.0e-10 * max(1.0, abs(z0), abs(z_min), abs(z_max)))
    if z0 < z_min - tol or z0 > z_max + tol:
        raise ValueError(
            f"Requested mirror-template start z0={z0:.12g} m lies outside the mirrored turning interval "
            f"[{z_min:.12g}, {z_max:.12g}] m for the current (E, μ)."
        )
    z0 = float(np.clip(z0, z_min, z_max))

    candidates: list[tuple[float, float, float]] = []
    for i in range(t.size - 1):
        z1 = float(z[i])
        z2 = float(z[i + 1])
        dz = z2 - z1
        if abs(dz) <= tol:
            if abs(z1 - z0) <= tol:
                candidates.append((float(t[i]), float(v[i]), abs(z1 - z0)))
            continue
        if z0 < min(z1, z2) - tol or z0 > max(z1, z2) + tol:
            continue
        frac = float(np.clip((z0 - z1) / dz, 0.0, 1.0))
        tau = float(t[i] + frac * (t[i + 1] - t[i]))
        vv = float(v[i] + frac * (v[i + 1] - v[i]))
        candidates.append((tau, vv, abs((z1 + frac * dz) - z0)))

    if not candidates:
        idx = int(np.argmin(np.abs(z - z0)))
        if abs(float(z[idx]) - z0) <= 10.0 * tol:
            candidates.append((float(t[idx]), float(v[idx]), abs(float(z[idx]) - z0)))

    if not candidates:
        raise RuntimeError(f"Unable to locate z0={z0:.12g} m on the mirrored bounce template.")

    sign_matched: list[tuple[float, float, float]] = []
    for tau, vv, err in candidates:
        if abs(vv) <= 1.0e-12 or np.sign(vv) == s_des:
            sign_matched.append((tau, vv, err))
    pool = sign_matched if sign_matched else candidates
    pool.sort(key=lambda item: (item[2], 0 if (abs(item[1]) <= 1.0e-12 or np.sign(item[1]) == s_des) else 1))
    return float(np.mod(pool[0][0], T))


def _rotate_full_bounce_template(
    t_full: np.ndarray,
    z_full: np.ndarray,
    v_full: np.ndarray,
    *,
    z0_m: float,
    desired_zdot_sign: int,
    z_tol_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a center-referenced full bounce so it starts at arbitrary (z0, sign).

    This is the generalized mirror mode.  It preserves the fast one-quarter ODE solve but
    supports off-midplane guiding-center starts and instantaneous starts whose initial
    cyclotron phase shifts the inferred guiding-center z.
    """
    t = np.asarray(t_full, dtype=float)
    z = np.asarray(z_full, dtype=float)
    v = np.asarray(v_full, dtype=float)
    T = float(t[-1])
    if T <= 0.0:
        raise ValueError("Cannot rotate a non-positive-period template.")

    tau0 = _phase_for_z_and_sign(
        t,
        z,
        v,
        z0_m=float(z0_m),
        desired_zdot_sign=int(desired_zdot_sign),
        z_tol_m=float(z_tol_m),
    )

    base = t[:-1]
    after = base[base > tau0 + 1.0e-18]
    before = base[base < tau0 - 1.0e-18] + T
    t_abs = np.concatenate([np.asarray([tau0]), after, before, np.asarray([tau0 + T])])
    t_abs.sort()
    keep = np.concatenate([[True], np.diff(t_abs) > 1.0e-18])
    t_abs = t_abs[keep]
    if t_abs[-1] < tau0 + T - 1.0e-18:
        t_abs = np.append(t_abs, tau0 + T)

    t_rel = t_abs - tau0
    z_rot = _periodic_interp(t, z, t_abs)
    v_rot = _periodic_interp(t, v, t_abs)
    z_rot[0] = float(z0_m)
    z_rot[-1] = float(z0_m)

    if not np.all(np.diff(t_rel) > 0.0):
        raise RuntimeError("Internal error: rotated template time axis is not strictly increasing.")
    return t_rel, z_rot, v_rot


def build_bounce_template(
    solver: AxialSolver,
    *,
    z0_m: float,
    vpar_sign0: int,
    duration_hint_s: float,
    max_duration_s: float,
    return_z_tol_m: float,
    min_reflections: int,
    method: TemplateBuildMethod,
    # Mirror-only checks:
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
) -> BounceTemplate:
    """
    Build one full-bounce template.

    - method="mirror" uses midplane symmetry: integrate one center -> +turn leg, mirror it
      into a full periodic bounce, then phase-rotate that periodic template to the requested
      arbitrary starting z and direction.
    - method="integrate" directly integrates until a detected full return to z0.
    - method="auto" uses mirror when the cached field-line profile is symmetric enough.
    """
    Eref = float(solver.energy_eV)
    _, _, vref = gamma_beta_v_from_kinetic(Eref)

    m = str(method).lower()
    if m not in {"auto", "integrate", "mirror"}:
        raise ValueError(f"Unknown template build method {method!r}. Expected 'auto', 'integrate', or 'mirror'.")

    if m == "auto":
        ok = True
        if mirror_symmetry_check:
            z_extent = min(float(np.max(np.abs(solver.field.z))), max(0.01, abs(float(z0_m)) + 0.01))
            ztest = np.linspace(0.0, z_extent, int(mirror_symmetry_ncheck))
            ok = _check_symmetry_about_zero(solver, ztest, float(mirror_symmetry_rel_tol))
        m = "mirror" if ok else "integrate"

    if m == "mirror":
        if not (float(solver.field.z[0]) <= 0.0 <= float(solver.field.z[-1])):
            raise ValueError("mirror template build requires the field-map z range to contain the symmetry plane z=0.")
        if mirror_symmetry_check:
            z_extent = min(float(np.max(np.abs(solver.field.z))), max(0.01, abs(float(z0_m)) + 0.01))
            ztest = np.linspace(0.0, z_extent, int(mirror_symmetry_ncheck))
            if not _check_symmetry_about_zero(solver, ztest, float(mirror_symmetry_rel_tol)):
                raise ValueError(
                    "mirror template build requested, but field-line symmetry check about z=0 failed. "
                    "Use template_build='integrate' for non-symmetric traps."
                )

        bz_over_B_center = float(np.asarray(solver.bz_over_B_at_z(0.0)).reshape(()))
        center_sign = 1 if bz_over_B_center >= 0.0 else -1
        bz_over_B_start = float(np.asarray(solver.bz_over_B_at_z(float(z0_m))).reshape(()))
        desired_zdot_sign = 1 if (float(vpar_sign0) * bz_over_B_start) >= 0.0 else -1

        T = max(float(duration_hint_s), 10.0 * float(solver.dt_min_s))
        while T <= float(max_duration_s):
            sol = solver.integrate(
                t0_s=0.0,
                duration_s=T,
                z0_m=0.0,
                vpar_sign=int(center_sign),
                stop_at_turning=True,
            )
            t = np.asarray(sol["t_s"], dtype=float)
            z = np.asarray(sol["z_m"], dtype=float)
            vpar = np.asarray(sol["vpar_m_per_s"], dtype=float)
            hit = bool(np.asarray(sol.get("hit_turning", [False]))[0])

            if hit and len(t) >= 3:
                t_full, z_full, v_full = _build_full_bounce_from_center_to_positive_turn(t, z, vpar)
                if mirror_symmetry_check:
                    z_extent_tpl = max(abs(float(np.min(z_full))), abs(float(np.max(z_full))))
                    zcheck = np.linspace(0.0, z_extent_tpl, int(mirror_symmetry_ncheck))
                    if not _check_symmetry_about_zero(solver, zcheck, float(mirror_symmetry_rel_tol)):
                        raise ValueError(
                            "mirror template build requested, but field-line symmetry check failed over the bounce extent."
                        )
                t_tpl, z_tpl, v_tpl = _rotate_full_bounce_template(
                    t_full,
                    z_full,
                    v_full,
                    z0_m=float(z0_m),
                    desired_zdot_sign=int(desired_zdot_sign),
                    z_tol_m=max(float(return_z_tol_m), float(mirror_z0_tol_m)),
                )
                return BounceTemplate(
                    t_rel_s=t_tpl,
                    z_m=z_tpl,
                    vpar_ref_m_per_s=v_tpl,
                    energy_ref_eV=Eref,
                    v_ref_m_per_s=float(vref),
                    z0_m=float(z0_m),
                    vpar_sign0=int(1 if int(vpar_sign0) >= 0 else -1),
                )

            T *= 2.0

        raise RuntimeError(
            f"Unable to detect a positive turning point from the trap center up to max_duration_s={max_duration_s}. "
            "Try increasing dynamics.template_max_duration_s or switching template_build='integrate'."
        )

    # ------------------------
    # method == "integrate"
    # ------------------------
    T = max(float(duration_hint_s), 10.0 * float(solver.dt_min_s))
    best_dz = np.inf
    best_reflections = 0

    while T <= float(max_duration_s):
        sol = solver.integrate(
            t0_s=0.0,
            duration_s=T,
            z0_m=float(z0_m),
            vpar_sign=int(vpar_sign0),
            stop_at_turning=False,
        )
        t = np.asarray(sol["t_s"], dtype=float)
        z = np.asarray(sol["z_m"], dtype=float)
        vpar = np.asarray(sol["vpar_m_per_s"], dtype=float)

        dz = np.abs(z - float(z0_m))
        best_dz = min(best_dz, float(np.min(dz)))
        best_reflections = max(best_reflections, int(len(_reflection_indices(vpar, min_dz_m=0.0))))

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
                t_rel_s=t_tpl,
                z_m=z_tpl,
                vpar_ref_m_per_s=v_tpl,
                energy_ref_eV=Eref,
                v_ref_m_per_s=float(vref),
                z0_m=float(z0_m),
                vpar_sign0=int(1 if int(vpar_sign0) >= 0 else -1),
            )

        T *= 2.0

    raise RuntimeError(
        f"Unable to detect a full bounce return up to max_duration_s={max_duration_s}. "
        f"Closest approach to z0 was |z-z0|={best_dz:.3e} m with {best_reflections} reflection(s). "
        "Try increasing dynamics.template_max_duration_s or loosening dynamics.template_return_z_tol_m."
    )


# -----------------------------------------------------------------------------
# Tiling helpers
# -----------------------------------------------------------------------------

def tile_bounce_template_constant_energy(
    tpl: BounceTemplate,
    *,
    t0_s: float,
    duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Repeat the bounce template in time to cover [t0, t0+duration].
    """
    T = float(tpl.bounce_period_s)
    if T <= 0.0:
        raise ValueError("Bounce template has non-positive period.")

    n = int(np.ceil(float(duration_s) / T)) + 1
    t_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []

    for i in range(n):
        t_seg = float(t0_s) + tpl.t_rel_s + i * T
        z_seg = tpl.z_m
        v_seg = tpl.vpar_ref_m_per_s
        if i > 0:
            # Avoid duplicate boundary point
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]
        t_list.append(t_seg)
        z_list.append(z_seg)
        v_list.append(v_seg)

    t = np.concatenate(t_list)
    z = np.concatenate(z_list)
    v = np.concatenate(v_list)

    t_end = float(t0_s) + float(duration_s)
    keep = t <= (t_end + 1e-15)
    t = t[keep]
    z = z[keep]
    v = v[keep]

    if t.size == 0 or t[-1] < t_end - 1.0e-15:
        phase = np.mod(t_end - float(t0_s), T)
        z_end = float(_periodic_interp(tpl.t_rel_s, tpl.z_m, phase))
        v_end = float(_periodic_interp(tpl.t_rel_s, tpl.vpar_ref_m_per_s, phase))
        t = np.concatenate([t, np.asarray([t_end], dtype=float)])
        z = np.concatenate([z, np.asarray([z_end], dtype=float)])
        v = np.concatenate([v, np.asarray([v_end], dtype=float)])

    return t, z, v


def tile_bounce_template_linear_energy(
    tpl: BounceTemplate,
    *,
    t0_s: float,
    duration_s: float,
    energy0_eV: float,
    loss_rate_eV_per_s: float,
    energy_floor_eV: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Repeat the bounce template in time to cover [t0, t0+duration], but scale each bounce
    duration according to the instantaneous speed inferred from a linear-in-time energy model:

        E(t) = E0 - loss_rate * (t - t0)   (clipped to energy_floor_eV)

    The bounce *shape* z(t_rel) is kept fixed (template assumption); only the time axis
    (and v_par) are scaled per bounce using v_ref / v(E_start_of_bounce).

    Returns: (t_s, z_m, vpar_m_per_s, E_eV)
    """
    t0 = float(t0_s)
    t_end = t0 + float(duration_s)

    T_ref = float(tpl.bounce_period_s)
    if T_ref <= 0.0:
        raise ValueError("Bounce template has non-positive period.")

    v_ref = float(tpl.v_ref_m_per_s)
    if v_ref <= 0.0:
        raise ValueError("Bounce template has non-positive v_ref_m_per_s.")

    t_rel = np.asarray(tpl.t_rel_s, dtype=float)
    z_ref = np.asarray(tpl.z_m, dtype=float)
    vpar_ref = np.asarray(tpl.vpar_ref_m_per_s, dtype=float)

    E0 = float(energy0_eV)
    rate = max(0.0, float(loss_rate_eV_per_s))
    E_floor = float(energy_floor_eV)

    def energy_at_time(tt: np.ndarray | float) -> np.ndarray:
        tt_arr = np.asarray(tt, dtype=float)
        E = E0 - rate * (tt_arr - t0)
        if np.isfinite(E_floor):
            E = np.maximum(E, E_floor)
        return E

    t_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    v_out: list[np.ndarray] = []
    E_out: list[np.ndarray] = []

    t_cursor = t0
    first = True

    while t_cursor < t_end - 1e-18:
        # Evaluate energy at the start of this bounce to set the time-stretch.
        E_start = float(np.asarray(energy_at_time(t_cursor)).reshape(()))
        _, _, v_start = gamma_beta_v_from_kinetic(E_start)
        v_start = float(np.asarray(v_start).reshape(()))
        scale_t0 = float(v_ref / max(v_start, 1e-30))

        # Midpoint refinement
        t_mid = float(t_cursor + 0.5 * scale_t0 * T_ref)
        E_mid = float(np.asarray(energy_at_time(t_mid)).reshape(()))
        _, _, v_mid = gamma_beta_v_from_kinetic(E_mid)
        v_mid = float(np.asarray(v_mid).reshape(()))

        scale_t = float(v_ref / max(v_mid, 1e-30))

        # Build this bounce with the refined scaling
        t_seg = t_cursor + scale_t * t_rel
        z_seg = z_ref
        v_seg = vpar_ref * (v_mid / max(v_ref, 1e-30))
        E_seg = energy_at_time(t_seg)

        if not first:
            # Avoid duplicate boundary sample
            t_seg = t_seg[1:]
            z_seg = z_seg[1:]
            v_seg = v_seg[1:]
            E_seg = E_seg[1:]
        first = False

        if len(t_seg) == 0:
            break

        if t_seg[-1] > t_end:
            keep = t_seg <= (t_end + 1e-15)
            t_seg = t_seg[keep]
            z_seg = z_seg[keep]
            v_seg = v_seg[keep]
            E_seg = E_seg[keep]

            t_out.append(t_seg)
            z_out.append(z_seg)
            v_out.append(v_seg)
            E_out.append(E_seg)
            break

        t_out.append(t_seg)
        z_out.append(z_seg)
        v_out.append(v_seg)
        E_out.append(E_seg)

        # Advance by one (scaled) bounce period
        t_cursor = t_cursor + scale_t * T_ref

    t = np.concatenate(t_out) if len(t_out) else np.array([], dtype=float)
    z = np.concatenate(z_out) if len(z_out) else np.array([], dtype=float)
    v = np.concatenate(v_out) if len(v_out) else np.array([], dtype=float)
    E = np.concatenate(E_out) if len(E_out) else np.array([], dtype=float)

    return t, z, v, E
