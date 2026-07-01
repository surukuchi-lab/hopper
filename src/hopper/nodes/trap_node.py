"""
Module: hopper.nodes.trap_node

Developer: ehtkarim
Date: April 29, 2026

Loads or generates trap magnetic-field maps and stores field metadata in the pipeline context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..config import MainConfig
from ..field.field_map import FieldMap
from ..field.generator import FieldGridSpec


@dataclass
class TrapNode:
    cfg: MainConfig
    name: str = "trap"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        trap = self.cfg.trap
        npz_path = Path(trap.field_map_npz)

        grid = FieldGridSpec(
            r_max_m=trap.grid.r_max_m,
            z_min_m=trap.grid.z_min_m,
            z_max_m=trap.grid.z_max_m,
            n_r=trap.grid.n_r,
            n_z=trap.grid.n_z,
        )

        if npz_path.exists():
            field = FieldMap.from_npz(
                npz_path,
                method=trap.interpolation,
                clamp_to_grid=trap.clamp_to_grid,
                placeholder_if_missing=trap.placeholder_if_missing,
                grid_if_placeholder=grid,
            )
        else:
            if trap.generate_if_missing and trap.coil_xml is not None:
                field = FieldMap.from_coil_xml(
                    trap.coil_xml,
                    out_npz_path=npz_path,
                    grid=grid,
                    method=trap.interpolation,
                    clamp_to_grid=trap.clamp_to_grid,
                )
            else:
                field = FieldMap.from_npz(
                    npz_path,
                    method=trap.interpolation,
                    clamp_to_grid=trap.clamp_to_grid,
                    placeholder_if_missing=trap.placeholder_if_missing,
                    grid_if_placeholder=grid,
                )

        profiler = ctx.get("profiler")
        if profiler is not None:
            profiler.add_note(
                "trap",
                field_map_npz=str(npz_path),
                interpolation=str(trap.interpolation),
                clamp_to_grid=bool(trap.clamp_to_grid),
                grid_shape=f"{field.r.size}x{field.z.size}",
                has_components=bool(field.Br is not None and field.Bz is not None),
            )

        ctx = dict(ctx)
        ctx["field"] = field
        return ctx
