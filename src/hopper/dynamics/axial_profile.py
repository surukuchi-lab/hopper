from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..field.field_map import FieldMap


@dataclass(frozen=True)
class AxialFieldProfile:
    """
    Cached guiding-center field-line profile on the axisymmetric (r, z) field map.

    The guiding center follows the background magnetic field line in the meridional
    plane, so the profile is parameterized as r(z) with

        dr/dz = Br / Bz.

    This magnetic geometry is independent of the electron state. Radiative losses
    therefore change only the kinematics on the same cached field line (through the
    evolving mirror condition, cyclotron radius, and local frequency), not the field
    line itself.

    Notes
    -----
    - The implementation is targeted at the repo's axisymmetric trapping fields,
      where Bphi is zero or very small and Bz does not vanish along the relevant
      field line. The meridional projection of the field line is still determined by
      Br / Bz when Bphi is present.
    - The profile is cached on the native field-map z grid plus the exact starting z0.
      All repeated axial/radiative evaluations then use inexpensive 1D interpolation.
    """

    z_m: np.ndarray
    r_m: np.ndarray
    B_T: np.ndarray
    Br_T: np.ndarray
    Bphi_T: np.ndarray
    Bz_T: np.ndarray
    dBdr_T_per_m: np.ndarray
    dBdz_T_per_m: np.ndarray
    b_cross_kappa_phi_per_m: np.ndarray
    z0_m: float

    @classmethod
    def from_field(
        cls,
        field: FieldMap,
        *,
        r0_m: float,
        z0_m: float,
    ) -> "AxialFieldProfile":
        z_grid = np.asarray(field.z, dtype=float)
        if z_grid.ndim != 1 or z_grid.size < 2:
            raise ValueError("Field map z grid must be a 1D array with at least two points.")
        if not np.all(np.diff(z_grid) > 0.0):
            raise ValueError("Field map z grid must be strictly increasing.")

        z0 = float(z0_m)
        z_min = float(z_grid[0])
        z_max = float(z_grid[-1])
        if not (z_min <= z0 <= z_max):
            if not field.clamp_to_grid:
                raise ValueError(f"Initial z0={z0} m is outside field-map z range [{z_min}, {z_max}].")
            z0 = float(np.clip(z0, z_min, z_max))

        r_min = float(field.r[0])
        r_max = float(field.r[-1])
        r0 = float(r0_m)
        if not (r_min <= r0 <= r_max):
            if not field.clamp_to_grid:
                raise ValueError(f"Initial r0={r0} m is outside field-map r range [{r_min}, {r_max}].")
            r0 = float(np.clip(r0, r_min, r_max))

        # Include z0 exactly so the initial condition is represented without interpolation drift.
        z = np.unique(np.concatenate([z_grid, np.asarray([z0], dtype=float)]))
        i0 = int(np.searchsorted(z, z0))
        r = np.empty_like(z, dtype=float)
        r[i0] = r0

        def slope(rr: float, zz: float) -> float:
            Br, _, Bz = field.components(float(rr), float(zz))
            br = float(np.asarray(Br).reshape(()))
            bz = float(np.asarray(Bz).reshape(()))
            bmag = float(np.asarray(field.B(float(rr), float(zz))).reshape(()))
            # Keep the z-parameterized field-line integration well behaved even if the map has extremely small |Bz| at some distant point.
            # This floor is many orders of magnitude below the field values in the production maps
            bz_floor = max(1.0e-15, 1.0e-12 * max(abs(bmag), 1.0))
            if abs(bz) < bz_floor:
                bz = np.copysign(bz_floor, bz if bz != 0.0 else 1.0)
            return br / bz

        # Second-order predictor/corrector integration forward in z
        for i in range(i0, z.size - 1):
            dz = float(z[i + 1] - z[i])
            k1 = slope(float(r[i]), float(z[i]))
            r_pred = float(np.clip(r[i] + k1 * dz, r_min, r_max))
            k2 = slope(r_pred, float(z[i + 1]))
            r[i + 1] = float(np.clip(r[i] + 0.5 * (k1 + k2) * dz, r_min, r_max))

        # Backward in z
        for i in range(i0, 0, -1):
            dz = float(z[i - 1] - z[i])
            k1 = slope(float(r[i]), float(z[i]))
            r_pred = float(np.clip(r[i] + k1 * dz, r_min, r_max))
            k2 = slope(r_pred, float(z[i - 1]))
            r[i - 1] = float(np.clip(r[i] + 0.5 * (k1 + k2) * dz, r_min, r_max))

        B = np.asarray(field.B(r, z), dtype=float)
        Br, Bphi, Bz = field.components(r, z)
        dBdr, dBdz = field.gradB(r, z)

        B_safe = np.maximum(np.abs(B), 1.0e-300)
        br_hat = np.asarray(Br, dtype=float) / B_safe
        bphi_hat = np.asarray(Bphi, dtype=float) / B_safe
        bz_hat = np.asarray(Bz, dtype=float) / B_safe

        # Cache the phi component of b × κ with κ = (b·∇)b
        #
        # Along the cached field line the magnetic geometry is stored as a function of z.
        # Since dz/ds = b_z, the field-line derivative is:
        #
        #     d/ds = b_z d/dz.
        #
        # The production maps are axisymmetric with Bphi ≈ 0, so evaluating the cylindrical components in a fixed local basis is sufficient and avoids recomputing curvature inside the compact solver / RF output loops.
        db_vec_dz = np.stack(
            [
                np.gradient(br_hat, z, edge_order=1),
                np.gradient(bphi_hat, z, edge_order=1),
                np.gradient(bz_hat, z, edge_order=1),
            ],
            axis=-1,
        )
        b_vec = np.stack([br_hat, bphi_hat, bz_hat], axis=-1)
        kappa_vec = bz_hat[:, None] * db_vec_dz
        cross_vec = np.cross(b_vec, kappa_vec)
        b_cross_kappa_phi = np.asarray(cross_vec[:, 1], dtype=float)

        return cls(
            z_m=z,
            r_m=np.asarray(r, dtype=float),
            B_T=np.asarray(B, dtype=float),
            Br_T=np.asarray(Br, dtype=float),
            Bphi_T=np.asarray(Bphi, dtype=float),
            Bz_T=np.asarray(Bz, dtype=float),
            dBdr_T_per_m=np.asarray(dBdr, dtype=float),
            dBdz_T_per_m=np.asarray(dBdz, dtype=float),
            b_cross_kappa_phi_per_m=np.asarray(b_cross_kappa_phi, dtype=float),
            z0_m=float(z0),
        )

    def _interp(self, values: np.ndarray, z: np.ndarray | float) -> np.ndarray:
        zq = np.asarray(z, dtype=float)
        return np.asarray(np.interp(zq, self.z_m, np.asarray(values, dtype=float)), dtype=float)

    def r_at_z(self, z: np.ndarray | float) -> np.ndarray:
        return self._interp(self.r_m, z)

    def B(self, z: np.ndarray | float) -> np.ndarray:
        return self._interp(self.B_T, z)

    def components(self, z: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._interp(self.Br_T, z), self._interp(self.Bphi_T, z), self._interp(self.Bz_T, z)

    def gradB(self, z: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return self._interp(self.dBdr_T_per_m, z), self._interp(self.dBdz_T_per_m, z)

    def b_cross_kappa_phi(self, z: np.ndarray | float) -> np.ndarray:
        """Return the phi component of b × (b·∇)b along the cached field line."""
        return self._interp(self.b_cross_kappa_phi_per_m, z)

    def br_over_B(self, z: np.ndarray | float) -> np.ndarray:
        B = np.maximum(np.abs(self.B(z)), 1.0e-300)
        return self._interp(self.Br_T, z) / B

    def bphi_over_B(self, z: np.ndarray | float) -> np.ndarray:
        B = np.maximum(np.abs(self.B(z)), 1.0e-300)
        return self._interp(self.Bphi_T, z) / B

    def bz_over_B(self, z: np.ndarray | float) -> np.ndarray:
        B = np.maximum(np.abs(self.B(z)), 1.0e-300)
        return self._interp(self.Bz_T, z) / B

    def dr_dz(self, z: np.ndarray | float) -> np.ndarray:
        Br = self._interp(self.Br_T, z)
        Bz = self._interp(self.Bz_T, z)
        out = np.zeros_like(np.asarray(Br, dtype=float))
        np.divide(Br, Bz, out=out, where=np.abs(Bz) > 1.0e-30)
        return np.asarray(out, dtype=float)
