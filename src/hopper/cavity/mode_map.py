"""
Module: hopper.cavity.mode_map

Developer: ehtkarim
Date: April 29, 2026

Loads and evaluates analytic or vector electric-field mode maps for gyro-averaged drive calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .. import constants as const
from .cavity import Cavity

CHI01P: float = 3.8317059702075125  # first root of J1


def _j1(x: np.ndarray) -> np.ndarray:
    """
    Bessel J1 with a SciPy fallback.
    The fallback is a low-order series approximation adequate for small x.
    """
    try:
        from scipy.special import j1 as scipy_j1
        return scipy_j1(x)
    except Exception:
        x = np.asarray(x, float)
        # J1(x) ≈ x/2 - x^3/16 + x^5/384  (good for small x)
        return 0.5 * x - (x**3) / 16.0 + (x**5) / 384.0


class ModeMap(Protocol):
    def __call__(self, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AnalyticTE011ModeMap:
    """
    Analytic signed TE_011 coupling proxy.

    This fallback is useful for geometry validation runs when no measured vector E-field map
    exists.  It preserves the voltage-level sign of the TE_011-like mode instead of using
    absolute values, but it is still not a vector-polarization model and is therefore not
    allowed for stimulated back-reaction when strict vector back-reaction is enabled.
    """
    cavity: Cavity

    is_vector_e_field: bool = False

    def __call__(self, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        r = np.asarray(r_m, float)
        z = np.asarray(z_m, float)
        Cr = _j1(CHI01P * r / self.cavity.radius_m)
        Cz = np.cos(np.pi * z / self.cavity.length_m)
        return Cr * Cz


@dataclass
class VectorElectricFieldModeMap:
    """
    Interpolated cavity electric-field phasor map.

    The supported text ``.fld`` format is the HFSS-style table used in this project:
    ``rho phi z Ex Ey Ez`` with cylindrical sample coordinates.  The component basis can
    be configured as Cartesian or cylindrical because the file header does not always make
    the vector basis unambiguous.  The interpolator is periodic in phi and masks invalid
    rows (NaN/Inf) to zero.

    The map can be used in two ways:
      * ``__call__(r,z)`` returns a dimensionless scalar magnitude proxy for legacy code.
      * ``gyro_drive_coupling_W_per_sqrt_J(...)`` returns the fundamental gyro-harmonic
        work coefficient q <v_perp · E_mode^* exp(-i psi)> for a unit-normalized cavity
        energy.  If the raw map is not normalized to one joule, use ``energy_normalization_J``
        or ``field_unit_scale``/``source_power_scale`` in the config to calibrate it.
    """

    cavity: Cavity
    rho_grid_m: np.ndarray
    phi_grid_rad: np.ndarray
    z_grid_m: np.ndarray
    vector_data: np.ndarray
    component_basis: Literal["cartesian", "cylindrical"] = "cartesian"
    field_unit_scale: float = 1.0
    energy_normalization_J: float = 1.0
    gyro_quadrature_points: int = 16
    scalar_normalization: float = 1.0
    bounds_policy: Literal["error", "warn", "zero"] = "error"

    is_vector_e_field: bool = True
    _runtime_counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    cache_info: dict[str, object] = field(default_factory=dict, repr=False)

    def reset_counters(self) -> None:
        self._runtime_counters.clear()

    def counter_snapshot(self) -> dict[str, int]:
        return dict(self._runtime_counters)

    def _inc_counter(self, name: str, amount: int = 1) -> None:
        self._runtime_counters[name] = int(self._runtime_counters.get(name, 0)) + int(amount)

    @staticmethod
    def _default_cache_path(path: Path) -> Path:
        return path.with_name(path.name + ".npz")

    @staticmethod
    def _source_signature(path: Path) -> tuple[int, int, str]:
        stat = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return int(stat.st_size), int(stat.st_mtime_ns), h.hexdigest()

    @classmethod
    def from_fld(
        cls,
        path: str | Path,
        *,
        cavity: Cavity,
        component_basis: Literal["cartesian", "cylindrical", "auto"] = "cartesian",
        field_unit_scale: float = 1.0,
        energy_normalization_J: float = 1.0,
        gyro_quadrature_points: int = 16,
        normalize_to_peak: bool = True,
        bounds_policy: Literal["error", "warn", "zero"] = "error",
        cache_enabled: bool = True,
        cache_path: str | Path | None = None,
        cache_rebuild: bool = False,
    ) -> "VectorElectricFieldModeMap":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        cache_info: dict[str, object] = {"used": False, "path": None}
        cache_file = Path(cache_path) if cache_path is not None else cls._default_cache_path(path)
        source_size, source_mtime_ns, source_sha256 = cls._source_signature(path)
        rho = phi_base = z = data_unscaled = None

        if bool(cache_enabled) and cache_file.exists() and not bool(cache_rebuild):
            try:
                with np.load(cache_file, allow_pickle=False) as cached:
                    if (
                        int(cached["source_size"]) == source_size
                        and int(cached["source_mtime_ns"]) == source_mtime_ns
                        and str(cached["source_sha256"].item()) == source_sha256
                    ):
                        rho = np.asarray(cached["rho_grid_m"], dtype=float)
                        phi_base = np.asarray(cached["phi_grid_rad"], dtype=float)
                        z = np.asarray(cached["z_grid_m"], dtype=float)
                        data_unscaled = np.asarray(cached["vector_data"], dtype=float)
                        cache_info = {"used": True, "path": str(cache_file), "status": "hit"}
            except Exception:
                rho = phi_base = z = data_unscaled = None

        if rho is None or phi_base is None or z is None or data_unscaled is None:
            raw = np.genfromtxt(path, comments="#", dtype=float)
            if raw.ndim != 2 or raw.shape[1] < 6:
                raise ValueError(f"Vector E-field map {path} must have at least six numeric columns.")

            coords = raw[:, :3]
            vectors = raw[:, 3:6]
            rho = np.unique(coords[:, 0])
            phi_raw = np.unique(coords[:, 1])
            z = np.unique(coords[:, 2])
            # HFSS .fld files may label cylindrical phi in degrees even though the
            # numeric table has no units column.  Treat values spanning 0..360 as
            # degrees; otherwise treat them as radians.
            phi_scale = (np.pi / 180.0) if float(np.nanmax(phi_raw)) > 2.0 * np.pi + 1.0e-8 else 1.0
            coords = coords.copy()
            coords[:, 1] *= phi_scale
            phi = np.unique(coords[:, 1])
            rho.sort(); phi.sort(); z.sort()
            if rho.size < 2 or phi.size < 2 or z.size < 2:
                raise ValueError(f"Vector E-field map {path} must contain at least a 2x2x2 grid.")

            # Remove a duplicated 2π seam if present; periodic interpolation appends it again.
            if phi.size > 1 and np.isclose(phi[-1] - phi[0], 2.0 * np.pi, rtol=0.0, atol=1.0e-10):
                phi_base = phi[:-1]
            else:
                phi_base = phi

            data_unscaled = np.full((rho.size, phi_base.size, z.size, 3), np.nan, dtype=float)
            rho_index = {float(v): i for i, v in enumerate(rho)}
            phi_index = {float(v): i for i, v in enumerate(phi_base)}
            z_index = {float(v): i for i, v in enumerate(z)}
            for row, vec in zip(coords, vectors):
                rr, pp, zz = map(float, row)
                if pp not in phi_index:
                    # Skip the duplicate seam; it is represented by phi=phi0.
                    continue
                data_unscaled[rho_index[rr], phi_index[pp], z_index[zz], :] = vec

            data_unscaled = np.nan_to_num(data_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
            cache_info = {"used": False, "path": str(cache_file), "status": "miss"}
            if bool(cache_enabled):
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_file,
                        source_size=np.asarray(source_size, dtype=np.int64),
                        source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
                        source_sha256=np.asarray(source_sha256),
                        rho_grid_m=np.asarray(rho, dtype=float),
                        phi_grid_rad=np.asarray(phi_base, dtype=float),
                        z_grid_m=np.asarray(z, dtype=float),
                        vector_data=np.asarray(data_unscaled, dtype=float),
                    )
                    cache_info = {"used": False, "path": str(cache_file), "status": "written"}
                except Exception as exc:
                    cache_info = {"used": False, "path": str(cache_file), "status": f"write_failed:{type(exc).__name__}"}

        data = np.nan_to_num(np.asarray(data_unscaled, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) * float(field_unit_scale)
        basis = "cartesian" if component_basis == "auto" else str(component_basis)
        if basis not in {"cartesian", "cylindrical"}:
            raise ValueError("component_basis must be 'cartesian', 'cylindrical', or 'auto'.")

        # Scalar normalization is used only for the legacy scalar coupling fallback.
        mag = np.linalg.norm(data, axis=-1)
        peak = float(np.nanmax(mag)) if mag.size else 0.0
        scalar_norm = peak if normalize_to_peak and peak > 0.0 else 1.0
        return cls(
            cavity=cavity,
            rho_grid_m=rho,
            phi_grid_rad=phi_base,
            z_grid_m=z,
            vector_data=data,
            component_basis=basis,  # type: ignore[arg-type]
            field_unit_scale=1.0,
            energy_normalization_J=max(float(energy_normalization_J), 1.0e-300),
            gyro_quadrature_points=max(int(gyro_quadrature_points), 1),
            scalar_normalization=scalar_norm,
            bounds_policy=bounds_policy,
            cache_info=cache_info,
        )

    @cached_property
    def _periodic_interpolator(self):
        from scipy.interpolate import RegularGridInterpolator

        phi_ext = np.concatenate([self.phi_grid_rad, [self.phi_grid_rad[0] + 2.0 * np.pi]])
        data_ext = np.concatenate([self.vector_data, self.vector_data[:, :1, :, :]], axis=1)
        return RegularGridInterpolator(
            (self.rho_grid_m, phi_ext, self.z_grid_m),
            data_ext,
            bounds_error=False,
            fill_value=0.0,
        )

    def _cartesian_field(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        z = np.asarray(z_m, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        self._inc_counter("n_vector_interp_calls", 1)
        self._inc_counter("n_vector_interp_points", int(np.prod(shape, dtype=np.int64)))
        xb = np.broadcast_to(x, shape)
        yb = np.broadcast_to(y, shape)
        zb = np.broadcast_to(z, shape)
        rho = np.hypot(xb, yb)
        phi = np.mod(np.arctan2(yb, xb) - self.phi_grid_rad[0], 2.0 * np.pi) + self.phi_grid_rad[0]

        in_bounds = (
            (rho >= float(self.rho_grid_m[0]))
            & (rho <= float(self.rho_grid_m[-1]))
            & (zb >= float(self.z_grid_m[0]))
            & (zb <= float(self.z_grid_m[-1]))
        )
        if not bool(np.all(in_bounds)):
            msg = (
                "trajectory samples are outside the vector E-field map domain; "
                "increase the map domain or set mode_map.vector_bounds_policy='warn'/'zero' for diagnostics"
            )
            if self.bounds_policy == "error":
                raise ValueError(msg)
            if self.bounds_policy == "warn":
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        pts = np.column_stack([rho.ravel(), phi.ravel(), zb.ravel()])
        values = np.asarray(self._periodic_interpolator(pts), dtype=float).reshape(shape + (3,))
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if self.component_basis == "cartesian":
            return values

        er = values[..., 0]
        ephi = values[..., 1]
        ez = values[..., 2]
        cp = np.cos(phi)
        sp = np.sin(phi)
        ex = er * cp - ephi * sp
        ey = er * sp + ephi * cp
        return np.stack([ex, ey, ez], axis=-1)

    def __call__(self, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        r = np.asarray(r_m, dtype=float)
        z = np.asarray(z_m, dtype=float)
        x = r
        y = np.zeros_like(r, dtype=float)
        field_vec = self._cartesian_field(x, y, z)
        mag = np.linalg.norm(field_vec, axis=-1)
        return mag / max(float(self.scalar_normalization), 1.0e-300)

    def gyro_drive_coupling_W_per_sqrt_J(
        self,
        *,
        r_gc_m: np.ndarray,
        phi_gc_rad: np.ndarray,
        z_gc_m: np.ndarray,
        B_T: np.ndarray,
        gamma: np.ndarray,
        mu_J_per_T: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        phase_rad: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fundamental gyro-harmonic cavity-drive coefficient.

        The returned coefficient has units of W / sqrt(J) if the field map is scaled to a
        one-joule stored-energy phasor.  The source power used by the compact cavity model is
        approximately ``|drive|^2 * tau_E`` times the resonance response.
        """
        r_gc = np.asarray(r_gc_m, dtype=float)
        self._inc_counter("n_gyro_drive_calls", 1)
        phi_gc = np.asarray(phi_gc_rad, dtype=float)
        z_gc = np.asarray(z_gc_m, dtype=float)
        B = np.asarray(B_T, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float)
        mu = np.asarray(mu_J_per_T, dtype=float)
        shape = np.broadcast_shapes(r_gc.shape, phi_gc.shape, z_gc.shape, B.shape, gamma_arr.shape, mu.shape)
        self._inc_counter("n_gyro_drive_samples", int(np.prod(shape, dtype=np.int64)))
        r_gc = np.broadcast_to(r_gc, shape)
        phi_gc = np.broadcast_to(phi_gc, shape)
        z_gc = np.broadcast_to(z_gc, shape)
        B = np.broadcast_to(B, shape)
        gamma_arr = np.broadcast_to(gamma_arr, shape)
        mu = np.broadcast_to(mu, shape)

        v_perp = np.sqrt(np.maximum(2.0 * mu * B / np.maximum(gamma_arr * const.M_E, 1.0e-300), 0.0))
        rho_larmor = gamma_arr * const.M_E * v_perp / (const.E_CHARGE * np.maximum(B, 1.0e-300))
        x_gc = r_gc * np.cos(phi_gc)
        y_gc = r_gc * np.sin(phi_gc)

        u1_arr = np.broadcast_to(np.asarray(u1, dtype=float), shape + (3,))
        u2_arr = np.broadcast_to(np.asarray(u2, dtype=float), shape + (3,))
        n_gyro = max(int(self.gyro_quadrature_points), 1)
        phase0 = np.zeros(shape, dtype=float) if phase_rad is None else np.broadcast_to(np.asarray(phase_rad, dtype=float), shape)

        # Evaluate the vector field for all gyro nodes in one interpolator call.  This is
        # mathematically identical to looping over gyro phase, but it avoids n_gyro
        # separate RegularGridInterpolator calls.  The validation bundle showed that the
        # vector-map interpolation dominates runtime when back-reaction is enabled.
        gyro_phase = phase0[..., None] + (2.0 * np.pi / float(n_gyro)) * np.arange(n_gyro, dtype=float)
        cos_psi = np.cos(gyro_phase)
        sin_psi = np.sin(gyro_phase)

        u1_q = u1_arr[..., None, :]
        u2_q = u2_arr[..., None, :]
        radial = cos_psi[..., None] * u1_q + sin_psi[..., None] * u2_q
        tangent = -sin_psi[..., None] * u1_q + cos_psi[..., None] * u2_q

        pos = np.stack([x_gc, y_gc, z_gc], axis=-1)[..., None, :] + rho_larmor[..., None, None] * radial
        flat_pos = pos.reshape((-1, 3))
        e_vec = self._cartesian_field(flat_pos[:, 0], flat_pos[:, 1], flat_pos[:, 2]).reshape(pos.shape)
        v_vec = v_perp[..., None, None] * tangent
        acc = np.sum(np.sum(v_vec * np.conjugate(e_vec), axis=-1) * np.exp(-1j * gyro_phase), axis=-1)
        drive = (-const.E_CHARGE) * acc / float(n_gyro)
        return drive / np.sqrt(max(float(self.energy_normalization_J), 1.0e-300))
