from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import InterpMethod
from ..utils.interpolation import Grid2DInterpolator
from .generator import FieldGridSpec, generate_field_map_from_coil_xml, save_field_map_npz


def _placeholder_field(r: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple non-physical placeholder field used only when no map is available.
    Mirrors the notebook's fallback structure.
    """
    rr = r[:, None]
    zz = z[None, :]
    Bmag = 1.0 + 0.02 * rr * rr + 0.05 * zz * zz
    Br = np.zeros_like(Bmag)
    Bphi = np.zeros_like(Bmag)
    Bz = Bmag.copy()
    return Br, Bphi, Bz, Bmag


@dataclass
class FieldMap:
    """
    Axisymmetric magnetic field map on a regular (r,z) grid.

    Supports interpolation of:
      - Bmag (always)
      - Br, Bphi, Bz (optional, if provided by NPZ or generator)
      - gradients dBdr, dBdz computed from Bmag on the grid

    The default file format mirrors the notebook expectation:
      - r: 1D array (meters)
      - z: 1D array (meters)
      - Bmag: (Nr,Nz) array (Tesla)
    Optional:
      - Br, Bphi, Bz: (Nr,Nz) arrays (Tesla)
    """

    r: np.ndarray
    z: np.ndarray
    Bmag: np.ndarray
    Br: Optional[np.ndarray] = None
    Bphi: Optional[np.ndarray] = None
    Bz: Optional[np.ndarray] = None

    method: InterpMethod = "bilinear"
    clamp_to_grid: bool = True

    _Bmag_itp: Optional[Grid2DInterpolator] = None
    _Br_itp: Optional[Grid2DInterpolator] = None
    _Bphi_itp: Optional[Grid2DInterpolator] = None
    _Bz_itp: Optional[Grid2DInterpolator] = None
    _dBdr_itp: Optional[Grid2DInterpolator] = None
    _dBdz_itp: Optional[Grid2DInterpolator] = None

    def __post_init__(self) -> None:
        self.r = np.asarray(self.r, float)
        self.z = np.asarray(self.z, float)
        self.Bmag = np.asarray(self.Bmag, float)

        if self.Br is not None:
            self.Br = np.asarray(self.Br, float)
        if self.Bphi is not None:
            self.Bphi = np.asarray(self.Bphi, float)
        if self.Bz is not None:
            self.Bz = np.asarray(self.Bz, float)

        self._Bmag_itp = Grid2DInterpolator(self.r, self.z, self.Bmag, method=self.method, clamp_to_grid=self.clamp_to_grid)
        if self.Br is not None:
            self._Br_itp = Grid2DInterpolator(self.r, self.z, self.Br, method=self.method, clamp_to_grid=self.clamp_to_grid)
        if self.Bphi is not None:
            self._Bphi_itp = Grid2DInterpolator(self.r, self.z, self.Bphi, method=self.method, clamp_to_grid=self.clamp_to_grid)
        if self.Bz is not None:
            self._Bz_itp = Grid2DInterpolator(self.r, self.z, self.Bz, method=self.method, clamp_to_grid=self.clamp_to_grid)

        dBdr, dBdz = np.gradient(self.Bmag, self.r, self.z, edge_order=2)
        self._dBdr_itp = Grid2DInterpolator(self.r, self.z, dBdr, method=self.method, clamp_to_grid=self.clamp_to_grid)
        self._dBdz_itp = Grid2DInterpolator(self.r, self.z, dBdz, method=self.method, clamp_to_grid=self.clamp_to_grid)

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        *,
        method: InterpMethod = "bilinear",
        clamp_to_grid: bool = True,
        placeholder_if_missing: bool = True,
        grid_if_placeholder: Optional[FieldGridSpec] = None,
    ) -> "FieldMap":
        npz_path = Path(npz_path)
        if npz_path.exists():
            data = np.load(npz_path)
            r = np.asarray(data["r"], float)
            z = np.asarray(data["z"], float)
            Bmag = np.asarray(data["Bmag"], float)
            Br = np.asarray(data["Br"], float) if "Br" in data else None
            Bphi = np.asarray(data["Bphi"], float) if "Bphi" in data else None
            Bz = np.asarray(data["Bz"], float) if "Bz" in data else None
            return cls(r=r, z=z, Bmag=Bmag, Br=Br, Bphi=Bphi, Bz=Bz, method=method, clamp_to_grid=clamp_to_grid)

        if not placeholder_if_missing:
            raise FileNotFoundError(npz_path)

        if grid_if_placeholder is None:
            grid_if_placeholder = FieldGridSpec(r_max_m=0.327, z_min_m=-2.0, z_max_m=2.0, n_r=33, n_z=161)
        r = np.linspace(0.0, float(grid_if_placeholder.r_max_m), int(grid_if_placeholder.n_r), dtype=float)
        z = np.linspace(float(grid_if_placeholder.z_min_m), float(grid_if_placeholder.z_max_m), int(grid_if_placeholder.n_z), dtype=float)
        Br, Bphi, Bz, Bmag = _placeholder_field(r, z)
        return cls(r=r, z=z, Bmag=Bmag, Br=Br, Bphi=Bphi, Bz=Bz, method=method, clamp_to_grid=clamp_to_grid)

    @classmethod
    def from_coil_xml(
        cls,
        coil_xml: str | Path,
        out_npz_path: str | Path,
        grid: FieldGridSpec,
        *,
        method: InterpMethod = "bilinear",
        clamp_to_grid: bool = True,
    ) -> "FieldMap":
        field = generate_field_map_from_coil_xml(coil_xml, grid)
        save_field_map_npz(out_npz_path, field)
        return cls(
            r=field["r"],
            z=field["z"],
            Bmag=field["Bmag"],
            Br=field.get("Br"),
            Bphi=field.get("Bphi"),
            Bz=field.get("Bz"),
            method=method,
            clamp_to_grid=clamp_to_grid,
        )

    def B(self, r: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        assert self._Bmag_itp is not None
        return self._Bmag_itp(r, z)

    def components(self, r: np.ndarray | float, z: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (Br, Bphi, Bz) at (r,z). If components are not present, returns (0,0,Bmag).
        """
        Bmag = self.B(r, z)
        if self._Br_itp is None or self._Bz_itp is None:
            return np.zeros_like(Bmag), np.zeros_like(Bmag), Bmag
        Br = self._Br_itp(r, z)
        Bphi = self._Bphi_itp(r, z) if self._Bphi_itp is not None else np.zeros_like(Br)
        Bz = self._Bz_itp(r, z)
        return Br, Bphi, Bz

    def gradB(self, r: np.ndarray | float, z: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dB/dr, dB/dz) of |B| at (r,z)."""
        assert self._dBdr_itp is not None and self._dBdz_itp is not None
        return self._dBdr_itp(r, z), self._dBdz_itp(r, z)
