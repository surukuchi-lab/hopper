from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..field.field_map import FieldMap
from .kinematics import critical_B_from_mu, gamma_beta_v_from_kinetic


@dataclass
class AxialSolver:
    """
    Fast custom axial integrator based on the notebook's approach:

        dz/dt = ± v_parallel(B(r0, z))
        v_parallel(B) = v * sqrt(max(0, 1 - B/Bc))

    Turning-point handling:
      - If a step would enter B > Bc (forbidden), reduce dt by halving to bracket the turning point.
      - Once a turning point is encountered, flip direction (mirror reflection).

    Key stability features (important at very high pitch angles):
      - adaptive dt shrink when |v_parallel| is small
      - dt never shrinks below dt_min_s
    """
    field: FieldMap
    r0_m: float
    E0_eV: float
    mu0_J_per_T: float

    dt_max_s: float = 1.0e-8
    dt_min_s: float = 1.0e-11
    safety: float = 0.9
    v_turn_threshold_c: float = 0.02  # fraction of c where we start shrinking dt
    max_steps: int = 5_000_000

    # Small tolerance so that tiny interpolation noise doesn't cause immediate "forbidden" classification.
    mirror_arg_tol: float = 1.0e-12

    @property
    def energy_eV(self) -> float:
        # Backwards-compatible alias used by some callers.
        return float(self.E0_eV)

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
        Integrate z(t) on a (generally) non-uniform time grid.

        Returns a dict with standard keys expected by the dynamics subsystem:
          - t_s
          - z_m
          - vpar_m_per_s

        If stop_at_turning=True, the integrator stops immediately after the first turning point.
        The dict also includes:
          - hit_turning (bool)
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

        gamma, _, v_tot = gamma_beta_v_from_kinetic(self.E0_eV)
        v_tot = float(np.asarray(v_tot).reshape(()))
        c0 = 299_792_458.0

        Bc = float(critical_B_from_mu(self.E0_eV, self.mu0_J_per_T))

        sign = 1.0 if int(vpar_sign) >= 0 else -1.0

        # Field map bounds (used only when clamp_to_grid=False).
        z_min = float(self.field.z[0])
        z_max = float(self.field.z[-1])

        # Initial state.
        z = float(z0_m)
        B0 = float(self.field.B(self.r0_m, z))
        arg0 = 1.0 - B0 / (Bc + 1e-300)

        # If we're exactly at the mirror point (or numerically beyond), there is no meaningful bounce.
        if arg0 <= 0.0 and abs(arg0) > self.mirror_arg_tol:
            raise ValueError(
                "Initial condition is already beyond the mirror condition (B0 > Bc). "
                "This typically indicates an inconsistent (mu0, E0) pair or an extreme pitch angle."
            )

        vpar0 = sign * v_tot * float(np.sqrt(max(arg0, 0.0)))

        t_list = [t0]
        z_list = [z]
        v_list = [vpar0]

        hit_turning = False

        for _ in range(self.max_steps):
            t = t_list[-1]
            z = z_list[-1]
            if t >= t_end:
                break

            B_here = float(self.field.B(self.r0_m, z))
            arg_now = 1.0 - B_here / (Bc + 1e-300)
            arg_now = float(arg_now)

            if arg_now <= 0.0:
                # Numerical noise can put us slightly beyond Bc. Treat as at-turning.
                vpar_abs = 0.0
            else:
                vpar_abs = v_tot * float(np.sqrt(arg_now))

            # adaptive dt near turning
            if vpar_abs < self.v_turn_threshold_c * c0:
                dt = self.safety * self.dt_max_s * (vpar_abs / (self.v_turn_threshold_c * c0 + 1e-30))
                dt = max(self.dt_min_s, dt)
            else:
                dt = self.dt_max_s

            dt = min(dt, t_end - t)
            dt = max(self.dt_min_s, dt)

            vpar = sign * vpar_abs

            # trial step
            z_trial = z + vpar * dt
            B_trial = float(self.field.B(self.r0_m, z_trial))
            arg_trial = 1.0 - B_trial / (Bc + 1e-300)

            if (arg_now > 0.0) and (arg_trial < -self.mirror_arg_tol):
                # Crossed into forbidden region; bisect dt to bracket turning point.
                n_halve = 0
                while arg_trial < -self.mirror_arg_tol and dt > self.dt_min_s and n_halve < 64:
                    dt = max(self.dt_min_s, 0.5 * dt)
                    z_trial = z + vpar * dt
                    B_trial = float(self.field.B(self.r0_m, z_trial))
                    arg_trial = 1.0 - B_trial / (Bc + 1e-300)
                    n_halve += 1

            # Decide whether we hit/overstepped the turning point.
            turning_step = (arg_now > 0.0) and (arg_trial <= self.mirror_arg_tol)

            if turning_step:
                hit_turning = True
                # Keep the trial position if it's still valid; otherwise stay put.
                if arg_trial < -self.mirror_arg_tol:
                    z_new = z
                else:
                    z_new = float(z_trial)
                v_new = 0.0
                sign *= -1.0  # reflect after turning
            else:
                z_new = float(z_trial)
                v_new = float(vpar)

            t_new = t + dt

            if (z_new < z_min or z_new > z_max) and not self.field.clamp_to_grid:
                # Stop if we leave the map and clamping is disabled.
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
