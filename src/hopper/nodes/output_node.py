from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..config import MainConfig
from ..io.npz_io import write_iq_npz
from ..io.root_io import write_track_root


@dataclass
class OutputNode:
    cfg: MainConfig
    name: str = "output"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        out_cfg = self.cfg.output
        feat = self.cfg.features

        out_dir = Path(out_cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = out_cfg.basename

        sig_res = ctx["signal_result"]
        track_if = sig_res.track_if

        if out_cfg.write_npz:
            npz_path = out_dir / f"{base}_iq.npz"
            meta = {
                "fs_if_hz": sig_res.fs_if_hz,
                "f_lo_hz": sig_res.f_lo_hz,
                "starting_time_s": float(self.cfg.simulation.starting_time_s),
                "track_length_s": float(self.cfg.simulation.track_length_s),
            }
            write_iq_npz(npz_path, sig_res.t_if, sig_res.iq_if, meta=meta)
            ctx = dict(ctx)
            ctx["npz_path"] = str(npz_path)

        write_root = bool(out_cfg.write_root or feat.write_root)
        if write_root:
            root_path = out_dir / f"{base}_track.root"
            arrays = {
                "time_steps": track_if.t,
                "position_x": track_if.x,
                "position_y": track_if.y,
                "position_z": track_if.z,
                "velocity_x": track_if.vx,
                "velocity_y": track_if.vy,
                "velocity_z": track_if.vz,
                "guiding_center_position_x": track_if.x_gc,
                "guiding_center_position_y": track_if.y_gc,
                "guiding_center_position_z": track_if.z_gc,
                "guiding_center_velocity_x": track_if.vx_gc,
                "guiding_center_velocity_y": track_if.vy_gc,
                "guiding_center_velocity_z": track_if.vz_gc,
                "cyclotron_frequency": track_if.f_c_hz,
                "amplitude_envelope": track_if.amp,
                "phase": track_if.phase_rf,
                "magnetic_field_T": track_if.B_T,
            }
            write_track_root(root_path, arrays)
            ctx = dict(ctx)
            ctx["root_path"] = str(root_path)

        return ctx
