"""
Module: hopper.nodes.mode_map_node

Developer: ehtkarim
Date: April 29, 2026

Builds the cavity mode map selected by configuration and records mode-map metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..cavity.cavity import Cavity
from ..cavity.mode_map import AnalyticTE011ModeMap, VectorElectricFieldModeMap
from ..config import MainConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ModeMapNode:
    cfg: MainConfig
    name: str = "mode_map"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cav_cfg = self.cfg.cavity
        cav = Cavity(radius_m=cav_cfg.radius_m, length_m=cav_cfg.length_m, f0_hz=cav_cfg.f0_hz, Q=cav_cfg.Q)

        mm_cfg = self.cfg.mode_map
        mm_type = str(mm_cfg.type).lower()
        vector_path = Path(mm_cfg.vector_e_field_map) if mm_cfg.vector_e_field_map else None
        wants_vector = mm_type in {"auto", "vector_e_field", "vector"}

        mode_map_kind = "analytic_te011"
        if wants_vector and vector_path is not None and vector_path.exists():
            mode_map = VectorElectricFieldModeMap.from_fld(
                vector_path,
                cavity=cav,
                component_basis=mm_cfg.vector_component_basis,
                field_unit_scale=float(mm_cfg.vector_field_unit_scale),
                energy_normalization_J=float(mm_cfg.vector_energy_normalization_J),
                gyro_quadrature_points=int(mm_cfg.vector_gyro_quadrature_points),
                normalize_to_peak=bool(mm_cfg.vector_normalize_to_peak),
                bounds_policy=mm_cfg.vector_bounds_policy,
                cache_enabled=bool(getattr(mm_cfg, "vector_cache_enabled", True)),
                cache_path=getattr(mm_cfg, "vector_cache_path", None),
                cache_rebuild=bool(getattr(mm_cfg, "vector_cache_rebuild", False)),
            )
            mode_map_kind = "vector_e_field"
            LOGGER.info("loaded vector cavity E-field mode map: %s", vector_path)
        elif mm_type in ("analytic_te011", "te011", "analytic", "auto") or bool(mm_cfg.fallback_to_analytic):
            if wants_vector and vector_path is not None and not vector_path.exists():
                LOGGER.warning("vector E-field mode map %s does not exist; falling back to analytic TE_011 mode map", vector_path)
            elif wants_vector and vector_path is None:
                LOGGER.info("no vector E-field mode map configured; using analytic TE_011 mode map")
            mode_map = AnalyticTE011ModeMap(cavity=cav)
        else:
            raise ValueError(f"Unsupported mode_map.type: {self.cfg.mode_map.type}")

        if bool(self.cfg.cavity.back_reaction_enabled) and bool(self.cfg.cavity.back_reaction_requires_vector_map):
            if not bool(getattr(mode_map, "is_vector_e_field", False)):
                raise ValueError(
                    "cavity.back_reaction_enabled=true requires a vector E-field mode map when "
                    "cavity.back_reaction_requires_vector_map=true. Configure mode_map.vector_e_field_map "
                    "or disable back-reaction for analytic TE_011 fallback runs."
                )

        profiler = ctx.get("profiler")
        if profiler is not None:
            profiler.add_note(
                "mode_map",
                kind=mode_map_kind,
                vector_e_field_map=str(vector_path) if vector_path is not None else None,
                vector_map_exists=bool(vector_path is not None and vector_path.exists()),
                component_basis=str(getattr(mode_map, "component_basis", "n/a")),
                gyro_quadrature_points=int(getattr(mode_map, "gyro_quadrature_points", 0)),
                fallback_to_analytic=bool(mm_cfg.fallback_to_analytic),
                vector_cache_info=dict(getattr(mode_map, "cache_info", {}) or {}),
                vector_counter_snapshot=(mode_map.counter_snapshot() if hasattr(mode_map, "counter_snapshot") else {}),
            )

        ctx = dict(ctx)
        ctx["cavity"] = cav
        ctx["mode_map"] = mode_map
        ctx["mode_map_kind"] = mode_map_kind
        return ctx
