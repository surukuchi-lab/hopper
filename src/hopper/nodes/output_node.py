from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from ..config import MainConfig
from ..dynamics.track import DynamicTrack, sample_dynamic_track
from ..io.npz_io import write_iq_npz
from ..io.root_io import write_track_root, write_track_root_chunks
from ..signal.sampling import estimate_sample_rate_hz, iter_rf_time_grid_index_chunks, rf_time_grid_spec
from ..signal.synth import SignalResult, _apply_sos_filter, _butter_lowpass_sos


def _track_root_arrays(track: DynamicTrack) -> Dict[str, Any]:
    return {
        "time_s": track.t,
        "position_x": track.x,
        "position_y": track.y,
        "position_z": track.z,
        "velocity_x": track.vx,
        "velocity_y": track.vy,
        "velocity_z": track.vz,
        "guiding_center_position_x": track.x_gc,
        "guiding_center_position_y": track.y_gc,
        "guiding_center_position_z": track.z_gc,
        "guiding_center_velocity_x": track.vx_gc,
        "guiding_center_velocity_y": track.vy_gc,
        "guiding_center_velocity_z": track.vz_gc,
        "cyclotron_frequency": track.f_c_hz,
        "amplitude_envelope": track.amp,
        "phase": track.phase_rf,
        "magnetic_field_T": track.B_T,
        "kinetic_energy_eV": track.energy_eV,
        "magnetic_moment_J_per_T": track.mu_J_per_T,
        "fieldline_b_cross_kappa_phi_per_m": track.b_cross_kappa_phi_per_m,
    }


def _concat(parts: list[np.ndarray], *, dtype: Any) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


@dataclass
class OutputNode:
    cfg: MainConfig
    name: str = "output"

    def _write_rf_root_and_npz_one_pass(
        self,
        ctx: Dict[str, Any],
        *,
        root_path: Path,
        npz_path: Path | None,
    ) -> SignalResult:
        """
        Reconstruct the RF-sampled track once in chunks, and use each chunk for both ROOT and IQ/NPZ generation.
        """
        cfg = self.cfg
        sig = cfg.signal
        track_dyn = ctx["track_dyn"]
        field = ctx["field"]
        mode_map = ctx["mode_map"]
        resonance = ctx.get("resonance_curve")

        M = int(sig.if_decim)
        if M < 1:
            raise ValueError("signal.if_decim must be >= 1")

        spec = rf_time_grid_spec(cfg, track_dyn)
        f_lo = float(sig.lo_hz) if sig.lo_hz is not None else float(np.mean(track_dyn.f_c_hz))
        carrier_phase0 = float(sig.carrier_phase0_rad)
        chunk_size = max(1, int(cfg.output.root_chunk_size))

        collect_npz = npz_path is not None
        needs_full_for_filter = bool(
            collect_npz
            and M > 1
            and bool(sig.if_antialias_filter)
            and bool(spec.is_uniform_time)
        )

        t_full_parts: list[np.ndarray] = []
        iq_full_parts: list[np.ndarray] = []
        t_if_parts: list[np.ndarray] = []
        iq_if_parts: list[np.ndarray] = []
        power_sum = 0.0
        power_count = 0

        def root_chunks() -> Iterable[Mapping[str, np.ndarray]]:
            nonlocal power_sum, power_count
            for indices, t_chunk in iter_rf_time_grid_index_chunks(cfg, track_dyn, chunk_size=chunk_size, spec=spec):
                tr = sample_dynamic_track(
                    cfg,
                    track_dyn,
                    field=field,
                    mode_map=mode_map,
                    resonance=resonance,
                    t_new=t_chunk,
                )

                if collect_npz:
                    phi_if = tr.phase_rf - 2.0 * np.pi * f_lo * t_chunk + carrier_phase0
                    iq_chunk = tr.amp * np.exp(1j * phi_if)
                    power_sum += float(np.sum(np.abs(iq_chunk) ** 2))
                    power_count += int(iq_chunk.size)

                    if needs_full_for_filter:
                        t_full_parts.append(np.asarray(t_chunk, dtype=float))
                        iq_full_parts.append(np.asarray(iq_chunk, dtype=np.complex128))
                    else:
                        keep = (np.asarray(indices, dtype=np.int64) % M) == 0
                        t_if_parts.append(np.asarray(t_chunk[keep], dtype=float))
                        iq_if_parts.append(np.asarray(iq_chunk[keep], dtype=np.complex128))

                yield _track_root_arrays(tr)

        write_track_root_chunks(root_path, root_chunks())

        if collect_npz:
            amplitude_normalization = 1.0
            if bool(sig.normalize_power) and power_count > 0:
                rms = float(np.sqrt(power_sum / max(power_count, 1)))
                if rms > 0.0:
                    amplitude_normalization = rms

            if needs_full_for_filter:
                t_full = _concat(t_full_parts, dtype=float)
                iq_full = _concat(iq_full_parts, dtype=np.complex128)
                if amplitude_normalization != 1.0:
                    iq_full = iq_full / amplitude_normalization
                fs_if_nominal = float(spec.fs_hz) / M
                cutoff = float(sig.if_filter_cutoff_ratio) * 0.5 * fs_if_nominal
                sos = _butter_lowpass_sos(fs_hz=float(spec.fs_hz), cutoff_hz=cutoff, order=int(sig.if_filter_order))
                iq_if = _apply_sos_filter(iq_full, sos)[::M]
                t_if = t_full[::M]
                fs_if_out = fs_if_nominal
            else:
                t_if = _concat(t_if_parts, dtype=float)
                iq_if = _concat(iq_if_parts, dtype=np.complex128)
                if amplitude_normalization != 1.0:
                    iq_if = iq_if / amplitude_normalization
                fs_if_out = (float(spec.fs_hz) / M) if spec.is_uniform_time else estimate_sample_rate_hz(t_if)

            meta = {
                "fs_if_hz": fs_if_out,
                "f_lo_hz": f_lo,
                "starting_time_s": float(cfg.simulation.starting_time_s),
                "track_length_s": float(cfg.simulation.track_length_s),
                "rf_grid_kind": spec.kind,
                "rf_grid_is_uniform_time": bool(spec.is_uniform_time),
                "rf_reconstruction": "one_pass_chunked",
            }
            write_iq_npz(npz_path, t_if, iq_if, meta=meta)
        else:
            t_if = np.asarray([], dtype=float)
            iq_if = np.asarray([], dtype=np.complex128)
            fs_if_out = (float(spec.fs_hz) / M) if spec.is_uniform_time else 0.0
            amplitude_normalization = 1.0

        empty = np.asarray([], dtype=float)
        empty_track = sample_dynamic_track(
            cfg,
            track_dyn,
            field=field,
            mode_map=mode_map,
            resonance=resonance,
            t_new=empty,
        )
        return SignalResult(
            t=empty,
            iq=np.asarray([], dtype=np.complex128),
            f_lo_hz=f_lo,
            fs_hz=float(spec.fs_hz),
            rf_grid_kind=spec.kind,
            rf_grid_is_uniform_time=spec.is_uniform_time,
            t_if=t_if,
            iq_if=iq_if,
            fs_if_hz=float(fs_if_out),
            track_rf=None,
            track_if=empty_track,
            amplitude_normalization=amplitude_normalization,
        )

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        out_cfg = self.cfg.output
        out_dir = Path(out_cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_cfg.basename
        sig_res = ctx["signal_result"]

        write_root = bool(out_cfg.write_root)
        track_sampling = str(out_cfg.track_sampling).lower()
        ctx = dict(ctx)

        if write_root and track_sampling == "rf_sampled":
            root_path = out_dir / f"{base}_track.root"
            npz_path = (out_dir / f"{base}_iq.npz") if bool(out_cfg.write_npz) else None
            sig_res = self._write_rf_root_and_npz_one_pass(ctx, root_path=root_path, npz_path=npz_path)
            ctx["signal_result"] = sig_res
            ctx["root_path"] = str(root_path)
            if npz_path is not None:
                ctx["npz_path"] = str(npz_path)
            return ctx

        if out_cfg.write_npz:
            npz_path = out_dir / f"{base}_iq.npz"
            meta = {
                "fs_if_hz": sig_res.fs_if_hz,
                "f_lo_hz": sig_res.f_lo_hz,
                "starting_time_s": float(self.cfg.simulation.starting_time_s),
                "track_length_s": float(self.cfg.simulation.track_length_s),
                "rf_grid_kind": getattr(sig_res, "rf_grid_kind", "uniform_time"),
                "rf_grid_is_uniform_time": bool(getattr(sig_res, "rf_grid_is_uniform_time", True)),
            }
            write_iq_npz(npz_path, sig_res.t_if, sig_res.iq_if, meta=meta)
            ctx["npz_path"] = str(npz_path)

        if write_root:
            root_path = out_dir / f"{base}_track.root"
            if track_sampling == "if_sampled":
                write_track_root(root_path, _track_root_arrays(sig_res.track_if))
            else:
                raise ValueError("output.track_sampling must be 'rf_sampled' or 'if_sampled'.")
            ctx["root_path"] = str(root_path)

        return ctx
