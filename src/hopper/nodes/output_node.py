"""
Module: hopper.nodes.output_node

Developer: ehtkarim
Date: April 29, 2026

Writes configured NPZ and ROOT outputs from generated tracks and IQ signals.
"""

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
from ..signal.synth import (
    SignalResult,
    _apply_sos_filter,
    _butter_lowpass_sos,
    apply_complex_cavity_response_to_track,
    cavity_complex_response_enabled,
    make_cavity_response,
)
from ..cavity.response import integrate_complex_envelope


def _track_root_arrays(track: DynamicTrack) -> Dict[str, Any]:
    arrays = {
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
    if track.cavity_energy_J is not None:
        arrays["cavity_stored_energy_J"] = track.cavity_energy_J
    if track.cavity_power_W is not None:
        arrays["cavity_output_power_W"] = track.cavity_power_W
    if track.cavity_source_power_W is not None:
        arrays["cavity_source_power_W"] = track.cavity_source_power_W
    if getattr(track, "cavity_work_power_W", None) is not None:
        arrays["cavity_work_power_W"] = track.cavity_work_power_W
    if getattr(track, "cavity_amplitude_sqrt_J", None) is not None:
        arrays["cavity_amplitude_real_sqrt_J"] = np.real(track.cavity_amplitude_sqrt_J)
        arrays["cavity_amplitude_imag_sqrt_J"] = np.imag(track.cavity_amplitude_sqrt_J)
    if getattr(track, "cavity_drive_sqrt_J_per_s", None) is not None:
        arrays["cavity_drive_real_sqrt_J_per_s"] = np.real(track.cavity_drive_sqrt_J_per_s)
        arrays["cavity_drive_imag_sqrt_J_per_s"] = np.imag(track.cavity_drive_sqrt_J_per_s)
    return arrays


def _concat(parts: list[np.ndarray], *, dtype: Any) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _slice_track(track: DynamicTrack, start: int) -> DynamicTrack:
    sl = slice(int(start), None)
    return DynamicTrack(
        t=track.t[sl],
        x=track.x[sl],
        y=track.y[sl],
        z=track.z[sl],
        vx=track.vx[sl],
        vy=track.vy[sl],
        vz=track.vz[sl],
        x_gc=track.x_gc[sl],
        y_gc=track.y_gc[sl],
        z_gc=track.z_gc[sl],
        vx_gc=track.vx_gc[sl],
        vy_gc=track.vy_gc[sl],
        vz_gc=track.vz_gc[sl],
        r_gc_m=track.r_gc_m[sl],
        phi_gc_rad=track.phi_gc_rad[sl],
        parallel_sign=track.parallel_sign[sl],
        b_cross_kappa_phi_per_m=track.b_cross_kappa_phi_per_m[sl],
        f_c_hz=track.f_c_hz[sl],
        amp=track.amp[sl],
        phase_rf=track.phase_rf[sl],
        B_T=track.B_T[sl],
        energy_eV=track.energy_eV[sl],
        mu_J_per_T=track.mu_J_per_T[sl],
        axial_profile=track.axial_profile,
        cavity_energy_J=None if track.cavity_energy_J is None else track.cavity_energy_J[sl],
        cavity_power_W=None if track.cavity_power_W is None else track.cavity_power_W[sl],
        cavity_source_power_W=None if track.cavity_source_power_W is None else track.cavity_source_power_W[sl],
        cavity_work_power_W=None if getattr(track, "cavity_work_power_W", None) is None else track.cavity_work_power_W[sl],
        cavity_amplitude_sqrt_J=None if getattr(track, "cavity_amplitude_sqrt_J", None) is None else track.cavity_amplitude_sqrt_J[sl],
        cavity_drive_sqrt_J_per_s=None if getattr(track, "cavity_drive_sqrt_J_per_s", None) is None else track.cavity_drive_sqrt_J_per_s[sl],
        solver_info=track.solver_info,
    )


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
        Reconstruct the RF-sampled track once in chunks, and use each chunk for both ROOT and
        IQ/NPZ generation.  This removes the previous signal-node/output-node double sampling.
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
        prev_t_last: float | None = None
        prev_phi_gc_last: float | None = None
        prev_phase_rf_last: float | None = None
        cavity_state: complex | None = None

        def root_chunks() -> Iterable[Mapping[str, np.ndarray]]:
            nonlocal power_sum, power_count, prev_t_last, prev_phi_gc_last, prev_phase_rf_last, cavity_state
            for indices, t_chunk in iter_rf_time_grid_index_chunks(cfg, track_dyn, chunk_size=chunk_size, spec=spec):
                use_overlap = prev_t_last is not None and t_chunk.size > 0
                if use_overlap:
                    t_eval = np.concatenate([np.asarray([float(prev_t_last)], dtype=float), np.asarray(t_chunk, dtype=float)])
                else:
                    t_eval = t_chunk

                tr = sample_dynamic_track(
                    cfg,
                    track_dyn,
                    field=field,
                    mode_map=mode_map,
                    resonance=resonance,
                    t_new=t_eval,
                    phi_gc_start_rad=prev_phi_gc_last,
                    phase_rf_start_rad=prev_phase_rf_last,
                )

                prev_t_last = float(tr.t[-1])
                prev_phi_gc_last = float(tr.phi_gc_rad[-1])
                prev_phase_rf_last = float(tr.phase_rf[-1])

                if cavity_complex_response_enabled(cfg):
                    initial = cavity_state
                    iq_eval, tr, cavity_amp = apply_complex_cavity_response_to_track(
                        cfg, tr, field=field, mode_map=mode_map, f_lo_hz=f_lo, initial_amplitude_sqrt_J=initial
                    )
                    cavity_state = complex(cavity_amp[-1]) if cavity_amp.size else initial
                else:
                    phi_if_eval = tr.phase_rf - 2.0 * np.pi * f_lo * tr.t + carrier_phase0
                    iq_eval = tr.amp * np.exp(1j * phi_if_eval)

                if use_overlap:
                    tr = _slice_track(tr, 1)
                    iq_chunk = np.asarray(iq_eval[1:], dtype=np.complex128)
                else:
                    iq_chunk = np.asarray(iq_eval, dtype=np.complex128)

                if collect_npz:
                    power_sum += float(np.sum(np.abs(iq_chunk) ** 2))
                    power_count += int(iq_chunk.size)

                    if needs_full_for_filter:
                        t_full_parts.append(np.asarray(tr.t, dtype=float))
                        iq_full_parts.append(np.asarray(iq_chunk, dtype=np.complex128))
                    else:
                        keep = (np.asarray(indices, dtype=np.int64) % M) == 0
                        t_if_parts.append(np.asarray(tr.t[keep], dtype=float))
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

    def _write_rf_root_track_only(self, ctx: Dict[str, Any], *, root_path: Path) -> None:
        """Write an RF-sampled ROOT track without regenerating or overwriting the IQ product."""
        cfg = self.cfg
        track_dyn = ctx.get("track_dyn")
        if track_dyn is None:
            tracks = ctx.get("track_dyns") or []
            if not tracks:
                raise ValueError("RF ROOT output requires at least one dynamic track.")
            track_dyn = tracks[0]
        field = ctx["field"]
        mode_map = ctx["mode_map"]
        resonance = ctx.get("resonance_curve")
        f_lo = float(cfg.signal.lo_hz) if cfg.signal.lo_hz is not None else float(np.mean(track_dyn.f_c_hz))
        chunk_size = max(1, int(cfg.output.root_chunk_size))
        spec = rf_time_grid_spec(cfg, track_dyn)

        prev_t_last: float | None = None
        prev_phi_gc_last: float | None = None
        prev_phase_rf_last: float | None = None
        cavity_state: complex | None = None

        def root_chunks() -> Iterable[Mapping[str, np.ndarray]]:
            nonlocal prev_t_last, prev_phi_gc_last, prev_phase_rf_last, cavity_state
            for _indices, t_chunk in iter_rf_time_grid_index_chunks(cfg, track_dyn, chunk_size=chunk_size, spec=spec):
                use_overlap = prev_t_last is not None and t_chunk.size > 0
                if use_overlap:
                    t_eval = np.concatenate([np.asarray([float(prev_t_last)], dtype=float), np.asarray(t_chunk, dtype=float)])
                else:
                    t_eval = np.asarray(t_chunk, dtype=float)

                tr = sample_dynamic_track(
                    cfg,
                    track_dyn,
                    field=field,
                    mode_map=mode_map,
                    resonance=resonance,
                    t_new=t_eval,
                    phi_gc_start_rad=prev_phi_gc_last,
                    phase_rf_start_rad=prev_phase_rf_last,
                )
                prev_t_last = float(tr.t[-1]) if tr.t.size else prev_t_last
                prev_phi_gc_last = float(tr.phi_gc_rad[-1]) if tr.t.size else prev_phi_gc_last
                prev_phase_rf_last = float(tr.phase_rf[-1]) if tr.t.size else prev_phase_rf_last

                if cavity_complex_response_enabled(cfg) and tr.t.size:
                    _iq_eval, tr, cavity_amp = apply_complex_cavity_response_to_track(
                        cfg, tr, field=field, mode_map=mode_map, f_lo_hz=f_lo, initial_amplitude_sqrt_J=cavity_state
                    )
                    cavity_state = complex(cavity_amp[-1]) if cavity_amp.size else cavity_state

                if use_overlap:
                    tr = _slice_track(tr, 1)
                if tr.t.size:
                    yield _track_root_arrays(tr)

        write_track_root_chunks(root_path, root_chunks())

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        out_cfg = self.cfg.output
        out_dir = Path(out_cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_cfg.basename
        sig_res = ctx["signal_result"]

        write_root = bool(out_cfg.write_root)
        track_sampling = str(out_cfg.track_sampling).lower()
        profiler = ctx.get("profiler")
        ctx = dict(ctx)

        # Use the old one-pass RF reconstruction only when the signal node intentionally
        # returned an empty placeholder.  Cavity/readout and pileup branches already built
        # the coherent IQ product and must not be overwritten by a single-track RF pass.
        if write_root and track_sampling == "rf_sampled" and sig_res.iq_if.size == 0 and sig_res.iq.size == 0:
            root_path = out_dir / f"{base}_track.root"
            npz_path = (out_dir / f"{base}_iq.npz") if bool(out_cfg.write_npz) else None
            sig_res = self._write_rf_root_and_npz_one_pass(ctx, root_path=root_path, npz_path=npz_path)
            ctx["signal_result"] = sig_res
            ctx["root_path"] = str(root_path)
            if npz_path is not None:
                ctx["npz_path"] = str(npz_path)
            if profiler is not None:
                profiler.add_note(
                    "output",
                    out_dir=str(out_dir),
                    write_npz=bool(out_cfg.write_npz),
                    write_root=bool(out_cfg.write_root),
                    requested_track_sampling=track_sampling,
                    actual_track_sampling="rf_sampled_one_pass_chunked",
                    root_path=str(root_path),
                    npz_path=str(npz_path) if npz_path is not None else None,
                )
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
                "cavity_response_model": str(self.cfg.cavity.response_model),
                "readout_model": str(self.cfg.readout.model),
                "n_tracks": len(ctx.get("track_dyns", [ctx.get("track_dyn")])),
            }
            if getattr(sig_res, "readout_meta", None):
                meta = {**meta, **sig_res.readout_meta}
            write_iq_npz(
                npz_path,
                sig_res.t_if,
                sig_res.iq_if,
                meta=meta,
                adc_iq=getattr(sig_res, "adc_iq", None),
                iq_fast=getattr(sig_res, "iq_fast", None),
                t_fast_s=getattr(sig_res, "t_fast", None),
            )
            ctx["npz_path"] = str(npz_path)

        if write_root:
            root_path = out_dir / f"{base}_track.root"
            if track_sampling == "if_sampled":
                write_track_root(root_path, _track_root_arrays(sig_res.track_if))
            elif track_sampling == "rf_sampled" and sig_res.track_rf is None:
                # Readout/pileup paths do not materialize RF-rate track arrays in memory.
                # Honor the requested RF ROOT output by reconstructing the first track in
                # chunks, without touching the already-generated IQ/readout product.
                self._write_rf_root_track_only(ctx, root_path=root_path)
                ctx["root_sampling_actual"] = "rf_sampled_chunked_track_only"
            elif track_sampling == "rf_sampled" and sig_res.track_rf is not None:
                write_track_root(root_path, _track_root_arrays(sig_res.track_rf))
            else:
                raise ValueError("output.track_sampling must be 'rf_sampled' or 'if_sampled'.")
            ctx["root_path"] = str(root_path)

        if profiler is not None:
            profiler.add_note(
                "output",
                out_dir=str(out_dir),
                write_npz=bool(out_cfg.write_npz),
                write_root=bool(out_cfg.write_root),
                requested_track_sampling=track_sampling,
                actual_track_sampling=str(ctx.get("root_sampling_actual", track_sampling if write_root else "none")),
                npz_path=str(ctx.get("npz_path")),
                root_path=str(ctx.get("root_path")),
            )

        return ctx
