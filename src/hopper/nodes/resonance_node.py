"""
Module: hopper.nodes.resonance_node

Developer: ehtkarim
Date: April 29, 2026

Builds optional measured or analytic resonance response objects for downstream signal generation.
"""

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

        need_resonance = bool(feat.include_resonance) or bool(getattr(self.cfg.cavity, "excitation_enabled", False))
        if not need_resonance or res_cfg.resonance_curve is None:
            curve = ResonanceCurve.unity()
        else:
            curve = ResonanceCurve.from_root(
                res_cfg.resonance_curve,
                object_name=res_cfg.object_name,
                normalize_to_peak=res_cfg.normalize_to_peak,
            )
        profiler = ctx.get("profiler")
        if profiler is not None:
            profiler.add_note(
                "resonance",
                enabled=bool(need_resonance),
                source=str(res_cfg.resonance_curve) if res_cfg.resonance_curve is not None else "unity",
                object_name=res_cfg.object_name,
                normalize_to_peak=bool(res_cfg.normalize_to_peak),
            )

        ctx = dict(ctx)
        ctx["resonance_curve"] = curve
        return ctx
