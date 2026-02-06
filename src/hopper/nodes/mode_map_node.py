from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..cavity.cavity import Cavity
from ..cavity.mode_map import AnalyticTE011ModeMap
from ..config import MainConfig


@dataclass
class ModeMapNode:
    cfg: MainConfig
    name: str = "mode_map"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cav_cfg = self.cfg.cavity
        cav = Cavity(radius_m=cav_cfg.radius_m, length_m=cav_cfg.length_m, f0_hz=cav_cfg.f0_hz, Q=cav_cfg.Q)

        mm_type = self.cfg.mode_map.type.lower()
        if mm_type in ("analytic_te011", "te011", "analytic"):
            mode_map = AnalyticTE011ModeMap(cavity=cav)
        else:
            raise ValueError(f"Unsupported mode_map.type: {self.cfg.mode_map.type}")

        ctx = dict(ctx)
        ctx["cavity"] = cav
        ctx["mode_map"] = mode_map
        return ctx
