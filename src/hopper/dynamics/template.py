from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from ..constants import E_CHARGE
from ..field.field_map import FieldMap
from .axial_solver import AxialSolver
from .kinematics import gamma_beta_v_from_kinetic, larmor_power_W_array, mu_from_pitch

TemplateBuildMethod = Literal["auto", "integrate", "mirror"]


@dataclass(frozen=True)
class BounceTemplate:
    """
    A single full-bounce axial template (z(t), v_par(t)) built at a reference energy.

    For tiling, the template is constructed to start at (t=0, z=z0) with the specified initial
    v_par sign, and to end at the first return to z0 moving in the same direction.

    In "mirror" mode , integrate only a quarter bounce
    (center -> turning point) and assemble a full bounce via symmetry.
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
    Indices where v_par sign flips. Optionally reject anomalous flips by requiring
    a minimum axial condition (|Δz| > min_dz_m) between accepted flips.
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

    # Require that at least one "real" axial excursion occurred.
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
    field: FieldMap,
    r0_m: float,
    z_samples_m: np.ndarray,
    rel_tol: float,
) -> bool:
    z = np.asarray(z_samples_m, dtype=float)
    Bp = field.B(float(r0_m), z)
    Bm = field.B(float(r0_m), -z)
    denom = np.maximum(np.abs(Bp), 1e-30)
    rel = np.max(np.abs(Bp - Bm) / denom)
    return bool(rel <= float(rel_tol))


def _build_full_bounce_from_quarter(t_q: np.ndarray, z_q: np.ndarray, v_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Notebook-style quarter -> full bounce:

      A: 0 -> +turn
      B: +turn -> 0 (reverse of A)
      C: 0 -> -turn (z -> -z)
      D: -turn -> 0 (reverse of C)

    IMPORTANT: t_full must be strictly increasing for downstream interpolation/resampling.
    """
    t_q = np.asarray(t_q, dtype=float)
    z_q = np.asarray(z_q, dtype=float)
    v_q = np.asarray(v_q, dtype=float)

    if len(t_q) < 3:
        raise ValueError("Quarter-bounce segment is too short to mirror.")

    # Ensure starts at t=0
    t_q = t_q - t_q[0]
    Tq = float(t_q[-1])

    # Segment A (include both endpoints)
    tA = t_q
    zA = z_q
    vA = v_q

    # Segment B: exclude the turning point (last sample), reverse to center.
    idxB = np.arange(len(t_q) - 2, -1, -1)  # len-2 ... 0
    tB = Tq + (Tq - t_q[idxB])
    zB = z_q[idxB]
    vB = -v_q[idxB]

    # Segment C: exclude the center point (first sample), go to negative turning.
    idxC = np.arange(1, len(t_q), 1)  # 1 ... end
    tC = tB[-1] + t_q[idxC]
    zC = -z_q[idxC]
    vC = -v_q[idxC]

    # Segment D: exclude the negative turning point (last of C), reverse to center.
    idxD = np.arange(len(t_q) - 2, -1, -1)  # len-2 ... 0
    tD = tC[-1] + (Tq - t_q[idxD])
    zD = (-z_q)[idxD]
    vD = v_q[idxD]

    t_full = np.concatenate([tA, tB, tC, tD], axis=0)
    z_full = np.concatenate([zA, zB, zC, zD], axis=0)
    v_full = np.concatenate([vA, vB, vC, vD], axis=0)

    # Final sanity check: monotone time
    if not np.all(np.diff(t_full) > 0):
        raise RuntimeError("Internal error: mirrored template time axis is not strictly increasing.")

    return t_full, z_full, v_full


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
    mirror_require_z0_near_zero: bool,
    mirror_z0_tol_m: float,
    mirror_symmetry_check: bool,
    mirror_symmetry_rel_tol: float,
    mirror_symmetry_ncheck: int,
) -> BounceTemplate:
    """
    Build one full bounce template.

    - method="mirror" reproduces the notebook's "quarter-bounce + mirroring" strategy.
    - method="integrate" integrates until a detected full return to z0 with the initial v_par sign.
    - method="auto" prefers "mirror" when z0≈0 and the field is symmetric about z=0.
    """
    Eref = float(solver.energy_eV)
    _, _, vref = gamma_beta_v_from_kinetic(Eref)

    m = str(method).lower()
    if m not in {"auto", "integrate", "mirror"}:
        raise ValueError(f"Unknown template build method {method!r}. Expected 'auto', 'integrate', or 'mirror'.")

    if m == "auto":
        # Default to "mirror" if we start at the trap center.
        if (abs(float(z0_m)) <= float(mirror_z0_tol_m)) or (not mirror_require_z0_near_zero):
            ok = True
            if mirror_symmetry_check:
                ztest = np.linspace(0.0, 0.01, int(mirror_symmetry_ncheck))
                ok = _check_symmetry_about_zero(solver.field, solver.r0_m, ztest, float(mirror_symmetry_rel_tol))
            m = "mirror" if ok else "integrate"
        else:
            m = "integrate"

    if m == "mirror":
        if mirror_require_z0_near_zero and abs(float(z0_m)) > float(mirror_z0_tol_m):
            raise ValueError(
                f"mirror template build requires |z0| <= {mirror_z0_tol_m} m, got z0={z0_m} m."
            )
        if mirror_symmetry_check:
            ztest = np.linspace(0.0, 0.01, int(mirror_symmetry_ncheck))
            if not _check_symmetry_about_zero(solver.field, solver.r0_m, ztest, float(mirror_symmetry_rel_tol)):
                raise ValueError(
                    "mirror template build requested, but field symmetry check about z=0 failed. "
                    "Use template_build='integrate' or disable mirror_symmetry_check."
                )

        T = max(float(duration_hint_s), 10.0 * float(solver.dt_min_s))
        while T <= float(max_duration_s):
            sol = solver.integrate(
                t0_s=0.0,
                duration_s=T,
                z0_m=float(z0_m),
                vpar_sign=int(vpar_sign0),
                stop_at_turning=True,
            )
            t = np.asarray(sol["t_s"], dtype=float)
            z = np.asarray(sol["z_m"], dtype=float)
            vpar = np.asarray(sol["vpar_m_per_s"], dtype=float)
            hit = bool(np.asarray(sol.get("hit_turning", [False]))[0])

            if hit and len(t) >= 3:
                # Quarter segment: start -> turning (already starts at 0)
                t_full, z_full, v_full = _build_full_bounce_from_quarter(t, z, vpar)
                return BounceTemplate(
                    t_rel_s=t_full,
                    z_m=z_full,
                    vpar_ref_m_per_s=v_full,
                    energy_ref_eV=Eref,
                    v_ref_m_per_s=float(vref),
                    z0_m=float(z0_m),
                    vpar_sign0=int(1 if int(vpar_sign0) >= 0 else -1),
                )

            T *= 2.0

        raise RuntimeError(
            f"Unable to detect a turning point (quarter bounce) up to max_duration_s={max_duration_s}. "
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
        "Try increasing dynamics.template_max_duration_s, loosening dynamics.template_return_z_tol_m, "
        "or use template_build='mirror' if starting near z=0."
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

    t_end = float(t0_s) + float(duration_s)

    t_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    v_out: list[np.ndarray] = []
    E_out: list[np.ndarray] = []

    E = float(energy0_eV)
    t_cursor = float(t0_s)

    t_rel = np.asarray(tpl.t_rel_s, dtype=float)
    vpar_ref = np.asarray(tpl.vpar_ref_m_per_s, dtype=float)
    T_ref = float(tpl.bounce_period_s)

    while t_cursor < t_end - 1e-18:
        _, _, v_cur = gamma_beta_v_from_kinetic(E)
        v_cur = float(np.asarray(v_cur).reshape(()))

        scale_t = float(tpl.v_ref_m_per_s / max(v_cur, 1e-30))

        t_seg = t_cursor + scale_t * t_rel
        z_seg = tpl.z_m
        v_seg = vpar_ref * (v_cur / max(tpl.v_ref_m_per_s, 1e-30))

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
        P = larmor_power_W_array(B_tpl, E, mu)  # array over template points
        E_loss_J = float(scale_t * np.trapz(P, t_rel))
        dE_eV = float(float(energy_loss_scale) * (E_loss_J / float(E_CHARGE)))

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

    Returns a positive loss_rate_eV_per_s.
    """
    if int(n_bounces) < 2:
        raise ValueError("n_bounces must be >= 2 to estimate a linear rate.")

    B_tpl = field.B(float(r0_m), np.asarray(tpl.z_m, dtype=float))
    B0 = float(field.B(float(r0_m), float(z0_m)))

    t_rel = np.asarray(tpl.t_rel_s, dtype=float)
    T_ref = float(tpl.bounce_period_s)

    times = [0.0]
    energies = [float(energy0_eV)]

    E = float(energy0_eV)
    t_cursor = 0.0

    for _ in range(int(n_bounces)):
        _, _, v_cur = gamma_beta_v_from_kinetic(E)
        v_cur = float(np.asarray(v_cur).reshape(()))
        scale_t = float(tpl.v_ref_m_per_s / max(v_cur, 1e-30))

        mu = float(mu_from_pitch(E, float(pitch_angle_deg), B0))
        P = larmor_power_W_array(B_tpl, E, mu)
        E_loss_J = float(scale_t * np.trapz(P, t_rel))
        dE_eV = float(float(energy_loss_scale) * (E_loss_J / float(E_CHARGE)))

        E = max(float(energy_floor_eV), E - dE_eV)
        t_cursor = t_cursor + scale_t * T_ref

        times.append(t_cursor)
        energies.append(E)

    t = np.asarray(times, dtype=float)
    Evec = np.asarray(energies, dtype=float)

    A = np.vstack([np.ones_like(t), t]).T
    coeff, *_ = np.linalg.lstsq(A, Evec, rcond=None)
    b = float(coeff[1])
    return max(0.0, -b)
