from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .coil_xml import LoopCoil, load_loop_coils_from_xml
from .loop_field import loop_field_br_bz_cylindrical


@dataclass(frozen=True)
class FieldGridSpec:
    r_max_m: float
    z_min_m: float
    z_max_m: float
    n_r: int
    n_z: int


def generate_field_map_from_loops(
    loops: List[LoopCoil],
    grid: FieldGridSpec,
    *,
    include_components: bool = True,
) -> dict:
    """
    Generate an axisymmetric field map on an (r,z) grid by summing contributions
    from circular loop coils.

    Returns dict containing r, z, Br, Bphi, Bz, Bmag (arrays).
    """
    r = np.linspace(0.0, float(grid.r_max_m), int(grid.n_r), dtype=float)
    z = np.linspace(float(grid.z_min_m), float(grid.z_max_m), int(grid.n_z), dtype=float)

    rr = r[:, None]  # (Nr,1)
    zz = z[None, :]  # (1,Nz)

    Br = np.zeros((r.size, z.size), dtype=float)
    Bz = np.zeros((r.size, z.size), dtype=float)

    for coil in loops:
        z_rel = zz - float(coil.z0_m)
        I_eff = float(coil.current_A) * int(coil.turns)
        dBr, dBz = loop_field_br_bz_cylindrical(rr, z_rel, float(coil.radius_m), I_eff)
        Br += dBr
        Bz += dBz

    Bphi = np.zeros_like(Br)
    Bmag = np.sqrt(Br * Br + Bphi * Bphi + Bz * Bz)

    out = {"r": r, "z": z, "Bmag": Bmag}
    if include_components:
        out.update({"Br": Br, "Bphi": Bphi, "Bz": Bz})
    return out


def generate_field_map_from_coil_xml(
    xml_path: str | Path,
    grid: FieldGridSpec,
) -> dict:
    loops = load_loop_coils_from_xml(xml_path)
    return generate_field_map_from_loops(loops, grid, include_components=True)


def save_field_map_npz(path: str | Path, field_map: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **field_map)
