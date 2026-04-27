from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..config import MainConfig
from ..dynamics.track import DynamicTrack, sample_dynamic_track
from ..field.field_map import FieldMap
from .sampling import estimate_sample_rate_hz, materialize_rf_time_grid, rf_time_grid_spec


@dataclass
class SignalResult:
    # Full-rate RF-grid IQ arrays. In phase_uniform mode, this grid is uniform in cyclotron phase and therefore usually non-uniform in time.
    t: np.ndarray
    iq: np.ndarray
    f_lo_hz: float
    fs_hz: float
    rf_grid_kind: str
    rf_grid_is_uniform_time: bool

    # Decimated arrays (if decimation used)
    t_if: np.ndarray
    iq_if: np.ndarray
    fs_if_hz: float

    # Tracks sampled onto output grids. track_rf may be None when RF ROOT output is requested, because the output node streams it from the compact dynamics track in chunks to avoid duplicating a very large RF track in memory.
    track_rf: Optional[DynamicTrack]
    track_if: DynamicTrack

    # If signal.normalize_power was applied, track amplitudes written later by a streaming output path should be divided by this same factor.
    amplitude_normalization: float = 1.0


def _butter_lowpass_sos(fs_hz: float, cutoff_hz: float, order: int):
    from scipy.signal import butter

    nyq = 0.5 * fs_hz
    wn = float(cutoff_hz) / nyq
    wn = min(max(wn, 1e-6), 0.999999)
    return butter(order, wn, btype="lowpass", output="sos")


def _apply_sos_filter(x: np.ndarray, sos) -> np.ndarray:
    from scipy.signal import sosfiltfilt

    x = np.asarray(x)
    padlen = 3 * (2 * int(sos.shape[0]) + 1)
    if x.size <= padlen:
        return x.copy()
    return sosfiltfilt(sos, x)


def _normalize_track_amplitude(track: DynamicTrack, scale: float) -> DynamicTrack:
    return DynamicTrack(
        t=track.t,
        x=track.x,
        y=track.y,
        z=track.z,
        vx=track.vx,
        vy=track.vy,
        vz=track.vz,
        x_gc=track.x_gc,
        y_gc=track.y_gc,
        z_gc=track.z_gc,
        vx_gc=track.vx_gc,
        vy_gc=track.vy_gc,
        vz_gc=track.vz_gc,
        r_gc_m=track.r_gc_m,
        phi_gc_rad=track.phi_gc_rad,
        parallel_sign=track.parallel_sign,
        b_cross_kappa_phi_per_m=track.b_cross_kappa_phi_per_m,
        f_c_hz=track.f_c_hz,
        amp=track.amp / scale,
        phase_rf=track.phase_rf,
        B_T=track.B_T,
        energy_eV=track.energy_eV,
        mu_J_per_T=track.mu_J_per_T,
    )


def _root_wants_rf_track(cfg: MainConfig) -> bool:
    sampling = str(getattr(cfg.output, "track_sampling", "rf_sampled")).lower()
    return bool(cfg.output.write_root) and sampling == "rf_sampled"


def synthesize_iq(
    cfg: MainConfig,
    track_dyn: DynamicTrack,
    *,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve,
) -> SignalResult:
    """
    Generate complex IQ time series from a DynamicTrack.

    Analytic convention:
      s_IF(t) = A(t) * exp(i (phi_RF(t) - 2π f_LO t + carrier_phase0))

    phase_uniform is intentionally preserved: when
    dynamics.samples_per_cyclotron_turn is set, the RF grid is sampled at fixed
    accumulated-cyclotron-phase increments. This keeps the requested number of
    samples per cyclotron turn without expanding the compact dynamics track.
    """
    sig = cfg.signal

    M = int(sig.if_decim)
    if M < 1:
        raise ValueError("signal.if_decim must be >= 1")

    grid_spec = rf_time_grid_spec(cfg, track_dyn)

    root_only_rf = _root_wants_rf_track(cfg)
    if root_only_rf:
        empty = np.asarray([], dtype=float)
        empty_track = sample_dynamic_track(
            cfg,
            track_dyn,
            field=field,
            mode_map=mode_map,
            resonance=resonance,
            t_new=empty,
        )
        f_lo = float(sig.lo_hz) if sig.lo_hz is not None else float(np.mean(track_dyn.f_c_hz))
        return SignalResult(
            t=empty,
            iq=np.asarray([], dtype=np.complex128),
            f_lo_hz=f_lo,
            fs_hz=float(grid_spec.fs_hz),
            rf_grid_kind=grid_spec.kind,
            rf_grid_is_uniform_time=grid_spec.is_uniform_time,
            t_if=empty,
            iq_if=np.asarray([], dtype=np.complex128),
            fs_if_hz=float(grid_spec.fs_hz) / max(int(sig.if_decim), 1),
            track_rf=None,
            track_if=empty_track,
            amplitude_normalization=1.0,
        )

    t = materialize_rf_time_grid(cfg, track_dyn, grid_spec)

    track_rf_full = sample_dynamic_track(
        cfg,
        track_dyn,
        field=field,
        mode_map=mode_map,
        resonance=resonance,
        t_new=t,
    )

    if sig.lo_hz is None:
        f_lo = float(np.mean(track_rf_full.f_c_hz))
    else:
        f_lo = float(sig.lo_hz)

    carrier_phase0 = float(sig.carrier_phase0_rad)
    phi_if = track_rf_full.phase_rf - 2.0 * np.pi * f_lo * t + carrier_phase0
    iq = track_rf_full.amp * np.exp(1j * phi_if)

    amplitude_normalization = 1.0
    if sig.normalize_power:
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        if rms > 0.0:
            amplitude_normalization = rms
            iq = iq / rms
            track_rf_full = _normalize_track_amplitude(track_rf_full, rms)

    if M > 1:
        if sig.if_antialias_filter and grid_spec.is_uniform_time:
            fs_if_nominal = float(grid_spec.fs_hz) / M
            cutoff = float(sig.if_filter_cutoff_ratio) * 0.5 * fs_if_nominal
            sos = _butter_lowpass_sos(fs_hz=float(grid_spec.fs_hz), cutoff_hz=cutoff, order=int(sig.if_filter_order))
            iq_f = _apply_sos_filter(iq, sos)
        else:
            # Digital anti-alias filters assume uniform time samples. For phase_uniform the grid is non-uniform, so decimation means stride-thinning the phase samples.
            # Users needing filtered IF data should use phase_bounded or an explicit uniform signal.fs_hz grid.
            iq_f = iq

        iq_if = iq_f[::M]
        t_if = t[::M]
        fs_if_out = (float(grid_spec.fs_hz) / M) if grid_spec.is_uniform_time else estimate_sample_rate_hz(t_if)
        track_if = sample_dynamic_track(
            cfg,
            track_dyn,
            field=field,
            mode_map=mode_map,
            resonance=resonance,
            t_new=t_if,
        )
        if sig.normalize_power and amplitude_normalization > 0.0:
            track_if = _normalize_track_amplitude(track_if, amplitude_normalization)
    else:
        iq_if = iq
        t_if = t
        fs_if_out = float(grid_spec.fs_hz)
        track_if = track_rf_full

    track_rf = track_rf_full if not _root_wants_rf_track(cfg) else None

    return SignalResult(
        t=t,
        iq=iq,
        f_lo_hz=f_lo,
        fs_hz=float(grid_spec.fs_hz),
        rf_grid_kind=grid_spec.kind,
        rf_grid_is_uniform_time=grid_spec.is_uniform_time,
        t_if=t_if,
        iq_if=iq_if,
        fs_if_hz=fs_if_out,
        track_rf=track_rf,
        track_if=track_if,
        amplitude_normalization=amplitude_normalization,
    )
