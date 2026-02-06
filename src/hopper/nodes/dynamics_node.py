from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig
from ..dynamics.track import build_dynamic_track


@dataclass
class DynamicsNode:
    cfg: MainConfig
    name: str = "dynamics"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        field = ctx["field"]
        mode_map = ctx["mode_map"]
        resonance = ctx.get("resonance_curve", ResonanceCurve.unity())

        track_dyn = build_dynamic_track(self.cfg, field=field, mode_map=mode_map, resonance=resonance)

        ctx = dict(ctx)
        ctx["track_dyn"] = track_dyn
        return ctx
