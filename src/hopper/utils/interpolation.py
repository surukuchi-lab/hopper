from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

InterpMethod = Literal["bilinear", "linear", "nearest", "cubic_spline"]


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


@dataclass
class Grid2DInterpolator:
    """
    Interpolates a scalar field defined on a tensor-product grid (r_grid, z_grid).

    - values shape must be (Nr, Nz) and correspond to (r, z) axes.
    - Supported methods: bilinear (manual), linear/nearest (SciPy RegularGridInterpolator),
      cubic_spline (SciPy RectBivariateSpline).

    Notes:
      - bilinear is usually fastest and has no SciPy overhead.
      - clamp_to_grid: if True, values outside grid are clamped to edges.
        If False, out-of-bounds points raise ValueError.
    """

    r_grid: np.ndarray
    z_grid: np.ndarray
    values: np.ndarray
    method: InterpMethod = "bilinear"
    clamp_to_grid: bool = True

    _call: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        self.r_grid = np.asarray(self.r_grid, float)
        self.z_grid = np.asarray(self.z_grid, float)
        self.values = np.asarray(self.values, float)

        if self.r_grid.ndim != 1 or self.z_grid.ndim != 1:
            raise ValueError("r_grid and z_grid must be 1D")
        if self.values.shape != (self.r_grid.size, self.z_grid.size):
            raise ValueError(
                f"values must have shape (Nr,Nz)={(self.r_grid.size,self.z_grid.size)}; got {self.values.shape}"
            )
        if not (np.all(np.diff(self.r_grid) > 0) and np.all(np.diff(self.z_grid) > 0)):
            raise ValueError("r_grid and z_grid must be strictly increasing")

        if self.method == "bilinear":
            self._call = self._bilinear
        elif self.method in ("linear", "nearest"):
            self._call = self._scipy_regular(method=self.method)
        elif self.method == "cubic_spline":
            self._call = self._scipy_spline()
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")

    def __call__(self, r: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        assert self._call is not None
        r_arr = np.asarray(r, float)
        z_arr = np.asarray(z, float)
        return self._call(r_arr, z_arr)

    # --- implementations ---

    def _bilinear(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        r0 = float(self.r_grid[0])
        r1 = float(self.r_grid[-1])
        z0 = float(self.z_grid[0])
        z1 = float(self.z_grid[-1])

        if self.clamp_to_grid:
            rq = _clamp(r, r0, r1)
            zq = _clamp(z, z0, z1)
        else:
            if np.any((r < r0) | (r > r1) | (z < z0) | (z > z1)):
                raise ValueError("Query point outside interpolation grid")
            rq, zq = r, z

        # bracket indices
        i = np.searchsorted(self.r_grid, rq, side="right") - 1
        j = np.searchsorted(self.z_grid, zq, side="right") - 1
        i = np.clip(i, 0, self.r_grid.size - 2)
        j = np.clip(j, 0, self.z_grid.size - 2)

        r_lo = self.r_grid[i]
        r_hi = self.r_grid[i + 1]
        z_lo = self.z_grid[j]
        z_hi = self.z_grid[j + 1]

        # weights
        tr = (rq - r_lo) / (r_hi - r_lo + 1e-300)
        tz = (zq - z_lo) / (z_hi - z_lo + 1e-300)

        c00 = self.values[i, j]
        c01 = self.values[i, j + 1]
        c10 = self.values[i + 1, j]
        c11 = self.values[i + 1, j + 1]

        c0 = (1.0 - tr) * c00 + tr * c10
        c1 = (1.0 - tr) * c01 + tr * c11
        return (1.0 - tz) * c0 + tz * c1

    def _scipy_regular(self, method: Literal["linear", "nearest"]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        try:
            from scipy.interpolate import RegularGridInterpolator
        except Exception as e:
            raise ImportError("SciPy is required for linear/nearest interpolation") from e

        interp = RegularGridInterpolator(
            (self.r_grid, self.z_grid),
            self.values,
            method=method,
            bounds_error=not self.clamp_to_grid,
            fill_value=None,
        )

        def call(r: np.ndarray, z: np.ndarray) -> np.ndarray:
            r0 = float(self.r_grid[0])
            r1 = float(self.r_grid[-1])
            z0 = float(self.z_grid[0])
            z1 = float(self.z_grid[-1])

            if self.clamp_to_grid:
                rq = _clamp(r, r0, r1)
                zq = _clamp(z, z0, z1)
            else:
                rq, zq = r, z
            pts = np.stack([rq, zq], axis=-1)
            return interp(pts)

        return call

    def _scipy_spline(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        try:
            from scipy.interpolate import RectBivariateSpline
        except Exception as e:
            raise ImportError("SciPy is required for cubic_spline interpolation") from e

        # RectBivariateSpline expects arrays in increasing order and returns 2D output;
        # use .ev for vector evaluation.
        spline = RectBivariateSpline(self.r_grid, self.z_grid, self.values, kx=3, ky=3)

        def call(r: np.ndarray, z: np.ndarray) -> np.ndarray:
            r0 = float(self.r_grid[0])
            r1 = float(self.r_grid[-1])
            z0 = float(self.z_grid[0])
            z1 = float(self.z_grid[-1])

            if self.clamp_to_grid:
                rq = _clamp(r, r0, r1)
                zq = _clamp(z, z0, z1)
            else:
                rq, zq = r, z

            rq_flat = np.ravel(rq)
            zq_flat = np.ravel(zq)
            out = spline.ev(rq_flat, zq_flat)
            return out.reshape(np.broadcast(rq, zq).shape)

        return call
