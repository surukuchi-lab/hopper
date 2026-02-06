from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig


@dataclass
class ResonanceNode:
    cfg: MainConfig
    name: str = "resonance"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        res_cfg = self.cfg.resonance
        feat = self.cfg.features

        if not feat.include_resonance or res_cfg.root_file is None:
            curve = ResonanceCurve.unity()
        else:
            curve = ResonanceCurve.from_root(
                res_cfg.root_file,
                object_name=res_cfg.object_name,
                normalize_to_peak=res_cfg.normalize_to_peak,
            )
        ctx = dict(ctx)
        ctx["resonance_curve"] = curve
        return ctx
