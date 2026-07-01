"""
Module: hopper.nodes.dynamics_node

Developer: ehtkarim
Date: April 29, 2026

Runs the dynamics stage and stores dynamic-track objects in the pipeline context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
        profiler = ctx.get("profiler")

        electron_cfgs = list(self.cfg.tracks) if self.cfg.tracks else [self.cfg.electron]
        tracks = []
        for idx, electron_cfg in enumerate(electron_cfgs):
            cfg_i = replace(self.cfg, electron=electron_cfg)
            if profiler is not None:
                with profiler.step("dynamics.build_track", track_index=idx):
                    track = build_dynamic_track(cfg_i, field=field, mode_map=mode_map, resonance=resonance)
                info = dict(track.solver_info or {})
                profiler.add_note(
                    f"dynamics.track_{idx}",
                    compact_points=int(track.t.size),
                    t_start_s=float(track.t[0]) if track.t.size else None,
                    t_end_s=float(track.t[-1]) if track.t.size else None,
                    z_min_m=float(track.z_gc.min()) if track.z_gc.size else None,
                    z_max_m=float(track.z_gc.max()) if track.z_gc.size else None,
                    energy_start_eV=float(track.energy_eV[0]) if track.energy_eV.size else None,
                    energy_end_eV=float(track.energy_eV[-1]) if track.energy_eV.size else None,
                    mu_start_J_per_T=float(track.mu_J_per_T[0]) if track.mu_J_per_T.size else None,
                    mu_end_J_per_T=float(track.mu_J_per_T[-1]) if track.mu_J_per_T.size else None,
                    solver_info=info,
                )
            else:
                track = build_dynamic_track(cfg_i, field=field, mode_map=mode_map, resonance=resonance)
            tracks.append(track)

        ctx = dict(ctx)
        ctx["track_dyns"] = tracks
        ctx["track_dyn"] = tracks[0]
        solver_info = {
            "n_tracks": len(tracks),
            "pileup_enabled": len(tracks) > 1,
            "coherent_sum": len(tracks) > 1,
        }
        if tracks[0].solver_info is not None:
            solver_info.update(dict(tracks[0].solver_info))
        if "mode_map_kind" in ctx:
            solver_info["mode_map_kind"] = ctx["mode_map_kind"]
        ctx["solver_info"] = solver_info
        return ctx
