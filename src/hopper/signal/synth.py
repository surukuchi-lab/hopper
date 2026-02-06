from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MainConfig
from ..dynamics.track import DynamicTrack, resample_dynamic_track


@dataclass
class SignalResult:
    # Full-rate (pre-decimation) arrays
    t: np.ndarray
    iq: np.ndarray
    f_lo_hz: float
    fs_hz: float

    # Decimated arrays (if decimation used)
    t_if: np.ndarray
    iq_if: np.ndarray
    fs_if_hz: float

    # Track resampled onto t_if (for ROOT output)
    track_if: DynamicTrack


def _butter_lowpass_sos(fs_hz: float, cutoff_hz: float, order: int):
    from scipy.signal import butter
    nyq = 0.5 * fs_hz
    wn = float(cutoff_hz) / nyq
    wn = min(max(wn, 1e-6), 0.999999)
    return butter(order, wn, btype="lowpass", output="sos")


def _apply_sos_filter(x: np.ndarray, sos) -> np.ndarray:
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, x)


def synthesize_iq(cfg: MainConfig, track_dyn: DynamicTrack) -> SignalResult:
    """
    Generate complex IQ time series from a DynamicTrack.

    Analytic convention:
      s_IF(t) = A(t) * exp(i (phi_RF(t) - 2π f_LO t + carrier_phase0))
    """
    sig = cfg.signal
    sim = cfg.simulation

    if sig.lo_hz is None:
        f_lo = float(np.mean(track_dyn.f_c_hz))
    else:
        f_lo = float(sig.lo_hz)

    M = int(sig.if_decim)
    if M < 1:
        raise ValueError("signal.if_decim must be >= 1")

    fs_if = float(sig.fs_if_hz)

    if sig.fs_hz is not None:
        fs = float(sig.fs_hz)
        if M > 1:
            fs_if = fs / M
    else:
        fs = fs_if * M

    if fs <= 0 or fs_if <= 0:
        raise ValueError("Sampling rates must be positive")

    t0 = float(sim.starting_time_s)
    duration = float(sim.track_length_s)
    n = int(np.floor(duration * fs))
    if n < 2:
        n = 2
    dt = 1.0 / fs
    t = t0 + dt * np.arange(n, dtype=float)

    track_full = resample_dynamic_track(track_dyn, t)

    carrier_phase0 = float(sig.carrier_phase0_rad)
    phi_if = track_full.phase_rf - 2.0 * np.pi * f_lo * t + carrier_phase0

    iq = track_full.amp * np.exp(1j * phi_if)

    if sig.normalize_power:
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        if rms > 0:
            iq = iq / rms
            track_full = DynamicTrack(
                t=track_full.t,
                x=track_full.x, y=track_full.y, z=track_full.z,
                vx=track_full.vx, vy=track_full.vy, vz=track_full.vz,
                x_gc=track_full.x_gc, y_gc=track_full.y_gc, z_gc=track_full.z_gc,
                vx_gc=track_full.vx_gc, vy_gc=track_full.vy_gc, vz_gc=track_full.vz_gc,
                f_c_hz=track_full.f_c_hz,
                amp=track_full.amp / rms,
                phase_rf=track_full.phase_rf,
            )

    if M > 1:
        if sig.if_antialias_filter:
            cutoff = float(sig.if_filter_cutoff_ratio) * 0.5 * fs_if
            sos = _butter_lowpass_sos(fs_hz=fs, cutoff_hz=cutoff, order=int(sig.if_filter_order))
            iq_f = _apply_sos_filter(iq, sos)
        else:
            iq_f = iq

        iq_if = iq_f[::M]
        t_if = t[::M]
        fs_if_out = fs / M
        track_if = resample_dynamic_track(track_full, t_if)
    else:
        iq_if = iq
        t_if = t
        fs_if_out = fs
        track_if = track_full

    return SignalResult(
        t=t,
        iq=iq,
        f_lo_hz=f_lo,
        fs_hz=fs,
        t_if=t_if,
        iq_if=iq_if,
        fs_if_hz=fs_if_out,
        track_if=track_if,
    )
