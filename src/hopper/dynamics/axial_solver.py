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

    with turning-point detection by checking B>Bc and reflecting.

    Key stability features (important at very high pitch angles):
      - adaptive dt that shrinks when |v_parallel| is small
      - turning-point handling that *never* lets dt shrink below dt_min
        (the original notebook halves dt; here we clamp at dt_min to avoid
        pathological tiny dt values that explode step count)
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
    
    def __getattr__(self, name: str):
    """
    Compatibility aliases between legacy and new solver attributes.

    - legacy: E0_eV
    - new:    energy_eV

    __getattr__ is only called if normal attribute lookup fails, so this does not
    interfere with dataclass field assignment (unlike @property).
    """
    d = self.__dict__
    if name == "energy_eV" and "E0_eV" in d:
        return float(d["E0_eV"])
    if name == "E0_eV" and "energy_eV" in d:
        return float(d["energy_eV"])
    raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")

    def integrate(
        self,
        t0_s: float,
        duration_s: float,
        z0_m: float,
        vpar_sign: int = 1,
    ) -> Dict[str, np.ndarray]:
        t_end = float(t0_s) + float(duration_s)
        if duration_s <= 0:
            return dict(t=np.asarray([t0_s], float), z=np.asarray([z0_m], float), vpar=np.asarray([0.0], float))

        gamma, beta, v_tot = gamma_beta_v_from_kinetic(self.E0_eV)
        v_tot = float(v_tot)
        c0 = 299_792_458.0

        Bc = critical_B_from_mu(self.E0_eV, self.mu0_J_per_T)

        t_list = [float(t0_s)]
        z_list = [float(z0_m)]
        vpar_list = []

        sign = 1.0 if int(vpar_sign) >= 0 else -1.0

        z_min = float(self.field.z[0])
        z_max = float(self.field.z[-1])

        for _ in range(self.max_steps):
            t = t_list[-1]
            z = z_list[-1]
            if t >= t_end:
                break

            B_here = float(self.field.B(self.r0_m, z))
            arg_now = 1.0 - B_here / (Bc + 1e-300)
            if arg_now <= 0.0:
                # If we ever land in the forbidden region, reflect direction and take a tiny step.
                sign *= -1.0
                vpar_abs = 0.0
            else:
                vpar_abs = v_tot * float(np.sqrt(arg_now))

            vpar = sign * vpar_abs
            vpar_list.append(vpar)

            # adaptive dt near turning
            if vpar_abs < self.v_turn_threshold_c * c0:
                dt = self.safety * self.dt_max_s * (vpar_abs / (self.v_turn_threshold_c * c0 + 1e-30))
                dt = max(self.dt_min_s, dt)
            else:
                dt = self.dt_max_s

            dt = min(dt, t_end - t)
            dt = max(self.dt_min_s, dt)

            # trial step
            z_new = z + vpar * dt
            B_new = float(self.field.B(self.r0_m, z_new))
            arg_new = 1.0 - B_new / (Bc + 1e-300)

            if arg_now > 0.0 and arg_new < 0.0:
                # crossed into forbidden region; reduce dt but never below dt_min
                n_halve = 0
                while arg_new < 0.0 and dt > self.dt_min_s and n_halve < 64:
                    dt = max(self.dt_min_s, 0.5 * dt)
                    z_new = z + vpar * dt
                    B_new = float(self.field.B(self.r0_m, z_new))
                    arg_new = 1.0 - B_new / (Bc + 1e-300)
                    n_halve += 1

                # if still slightly outside due to coarse map/interpolation, clamp z_new back
                if arg_new < 0.0:
                    z_new = z  # stay put; reflect and continue
                sign *= -1.0

            t_new = t + dt

            if (z_new < z_min or z_new > z_max) and not self.field.clamp_to_grid:
                break

            t_list.append(t_new)
            z_list.append(z_new)

        # pad vpar to match length
        if vpar_list:
            vpar_arr = np.asarray(vpar_list + [vpar_list[-1]], float)
        else:
            vpar_arr = np.asarray([0.0], float)

        return dict(t=np.asarray(t_list, float), z=np.asarray(z_list, float), vpar=vpar_arr)