from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..field.field_map import FieldMap
from .axial_profile import AxialFieldProfile
from .kinematics import critical_B_from_mu, gamma_beta_v_from_kinetic


@dataclass
class AxialSolver:
    """
    Guiding-center axial integrator on a cached magnetic field line.

    The compact solver still uses z as the integration coordinate, but the guiding
    center itself follows the field line r = r(z). The parallel-motion model is

        dz/dt = s * v_parallel * (Bz / B),

    where s in {+1, -1} is the sign of the parallel motion along the local field.
    Mirror reflection is handled by flipping s when the trial step reaches the
    forbidden region implied by the current (E, µ).

    The returned array under the historical key ``vpar_m_per_s`` is therefore the
    *z-component* of the parallel motion (dz/dt), preserving compatibility with the
    rest of the repo's compact axial-track interfaces.
    """

    field: FieldMap
    r0_m: float
    E0_eV: float
    mu0_J_per_T: float

    dt_max_s: float = 1.0e-8
    dt_min_s: float = 1.0e-11
    safety: float = 0.9
    v_turn_threshold_c: float = 0.02
    max_steps: int = 5_000_000
    axial_profile: Optional[AxialFieldProfile] = None

    mirror_arg_tol: float = 1.0e-12

    @property
    def energy_eV(self) -> float:
        return float(self.E0_eV)

    def B_at_z(self, z_m: np.ndarray | float) -> np.ndarray:
        if self.axial_profile is not None:
            return self.axial_profile.B(z_m)
        return self.field.B(self.r0_m, z_m)

    def r_at_z(self, z_m: np.ndarray | float) -> np.ndarray:
        if self.axial_profile is not None:
            return self.axial_profile.r_at_z(z_m)
        return np.full_like(np.asarray(z_m, dtype=float), float(self.r0_m), dtype=float)

    def bz_over_B_at_z(self, z_m: np.ndarray | float) -> np.ndarray:
        if self.axial_profile is not None:
            return self.axial_profile.bz_over_B(z_m)
        return np.ones_like(np.asarray(z_m, dtype=float), dtype=float)

    def integrate(
        self,
        *,
        t0_s: float,
        duration_s: float,
        z0_m: float,
        vpar_sign: int = 1,
        stop_at_turning: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Integrate z(t) on a generally non-uniform time grid.

        Returns a dict with keys:
          - t_s
          - z_m
          - vpar_m_per_s   (historical name; this is dz/dt)
          - hit_turning
        """
        t0 = float(t0_s)
        dur = float(duration_s)
        if dur <= 0.0:
            return dict(
                t_s=np.asarray([t0], float),
                z_m=np.asarray([float(z0_m)], float),
                vpar_m_per_s=np.asarray([0.0], float),
                hit_turning=np.asarray([False]),
            )

        t_end = t0 + dur

        _, _, v_tot = gamma_beta_v_from_kinetic(self.E0_eV)
        v_tot = float(np.asarray(v_tot).reshape(()))
        c0 = 299_792_458.0

        Bc = float(critical_B_from_mu(self.E0_eV, self.mu0_J_per_T))
        sign = 1.0 if int(vpar_sign) >= 0 else -1.0

        z_min = float(self.field.z[0])
        z_max = float(self.field.z[-1])

        z = float(z0_m)
        B0 = float(self.B_at_z(z))
        arg0 = 1.0 - B0 / (Bc + 1e-300)
        if arg0 <= 0.0 and abs(arg0) > self.mirror_arg_tol:
            raise ValueError(
                "Initial condition is already beyond the mirror condition (B0 > Bc). "
                "This typically indicates an inconsistent (mu0, E0) pair or an extreme pitch angle."
            )

        bz_over_B0 = float(np.asarray(self.bz_over_B_at_z(z)).reshape(()))
        vpar0 = sign * v_tot * float(np.sqrt(max(arg0, 0.0))) * bz_over_B0
        t_list = [t0]
        z_list = [z]
        v_list = [vpar0]
        hit_turning = False

        for _ in range(self.max_steps):
            t = t_list[-1]
            z = z_list[-1]
            if t >= t_end:
                break

            B_here = float(self.B_at_z(z))
            arg_now = float(1.0 - B_here / (Bc + 1e-300))
            vpar_abs = 0.0 if arg_now <= 0.0 else v_tot * float(np.sqrt(arg_now))

            if vpar_abs < self.v_turn_threshold_c * c0:
                dt = self.safety * self.dt_max_s * (vpar_abs / (self.v_turn_threshold_c * c0 + 1e-30))
                dt = max(self.dt_min_s, dt)
            else:
                dt = self.dt_max_s

            dt = min(dt, t_end - t)
            dt = max(self.dt_min_s, dt)

            bz_over_B = float(np.asarray(self.bz_over_B_at_z(z)).reshape(()))
            zdot = sign * vpar_abs * bz_over_B
            z_trial = z + zdot * dt
            B_trial = float(self.B_at_z(z_trial))
            arg_trial = 1.0 - B_trial / (Bc + 1e-300)

            if (arg_now > 0.0) and (arg_trial < -self.mirror_arg_tol):
                n_halve = 0
                while arg_trial < -self.mirror_arg_tol and dt > self.dt_min_s and n_halve < 64:
                    dt = max(self.dt_min_s, 0.5 * dt)
                    z_trial = z + zdot * dt
                    B_trial = float(self.B_at_z(z_trial))
                    arg_trial = 1.0 - B_trial / (Bc + 1e-300)
                    n_halve += 1

            turning_step = (arg_now > 0.0) and (arg_trial <= self.mirror_arg_tol)
            if turning_step:
                hit_turning = True
                z_new = z if arg_trial < -self.mirror_arg_tol else float(z_trial)
                v_new = 0.0
                sign *= -1.0
            else:
                z_new = float(z_trial)
                v_new = float(zdot)

            t_new = t + dt
            if (z_new < z_min or z_new > z_max) and not self.field.clamp_to_grid:
                break

            t_list.append(float(t_new))
            z_list.append(float(z_new))
            v_list.append(float(v_new))

            if stop_at_turning and turning_step:
                break

        return dict(
            t_s=np.asarray(t_list, float),
            z_m=np.asarray(z_list, float),
            vpar_m_per_s=np.asarray(v_list, float),
            hit_turning=np.asarray([hit_turning]),
        )
