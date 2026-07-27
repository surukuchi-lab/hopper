"""
Module: hopper.readout.locust_like

Developer: ehtkarim
Date: April 29, 2026

Applies LOCUST-like readout processing, including gain, noise, digitization, and optional decimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .digitizer import DigitizeResult, digitize_iq
from .noise import complex_white_noise


@dataclass
class LocustReadoutResult:
    t_fast: np.ndarray | None
    iq_fast: np.ndarray | None
    t: np.ndarray
    iq: np.ndarray
    adc_iq: np.ndarray | None
    meta: dict[str, Any]


def fft_brickwall_lpf(x: np.ndarray, *, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    """Complex baseband FFT brick-wall low-pass filter."""
    arr = np.asarray(x, dtype=np.complex128)
    if arr.size == 0:
        return arr.copy()
    freqs = np.fft.fftfreq(arr.size, d=1.0 / float(fs_hz))
    spec = np.fft.fft(arr)
    spec[np.abs(freqs) > float(cutoff_hz)] = 0.0
    return np.fft.ifft(spec)


def fft_brickwall_lpf_windows(
    x: np.ndarray,
    *,
    fs_hz: float,
    cutoff_hz: float,
    n_windows: int = 80,
) -> np.ndarray:
    """Locust-style windowed FFT LPF.

    Locust applies the FFT LPF to a fixed number of windows rather than one global
    transform.  This reproduces that receiver-processing semantics while keeping the
    input as an analytic complex baseband signal, not an undersampled RF waveform.
    """
    arr = np.asarray(x, dtype=np.complex128)
    if arr.size == 0:
        return arr.copy()
    n_win = max(int(n_windows), 1)
    if n_win == 1:
        return fft_brickwall_lpf(arr, fs_hz=fs_hz, cutoff_hz=cutoff_hz)
    out = np.empty_like(arr)
    edges = np.linspace(0, arr.size, n_win + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        out[start:stop] = fft_brickwall_lpf(arr[start:stop], fs_hz=fs_hz, cutoff_hz=cutoff_hz)
    return out


def fir_polyphase_decimate(x: np.ndarray, *, decimation_factor: int) -> np.ndarray:
    from scipy.signal import resample_poly

    D = max(int(decimation_factor), 1)
    if D == 1:
        return np.asarray(x, dtype=np.complex128).copy()
    return np.asarray(resample_poly(np.asarray(x, dtype=np.complex128), up=1, down=D), dtype=np.complex128)


def process_locust_like_readout(
    *,
    t_fast: np.ndarray,
    iq_fast: np.ndarray,
    fs_out_hz: float,
    decimation_factor: int = 10,
    lpf_cutoff_ratio: float = 0.85,
    lpf_mode: Literal["none", "fft_brickwall", "fir_polyphase"] = "none",
    n_windows: int = 80,
    add_noise: bool = False,
    noise_floor_psd_W_per_Hz: float | None = None,
    impedance_ohm: float = 50.0,
    rng: np.random.Generator | None = None,
    digitizer_config: Any | None = None,
    store_fast_iq: bool = False,
    exact_locust: bool = False,
) -> LocustReadoutResult:
    """Apply a Locust-style terminal readout chain to analytic baseband IQ.

    The exact emulation path is: complex baseband fast buffer -> FFT brick-wall LPF
    in ``n_windows`` windows -> stride decimation -> optional time-domain Gaussian
    noise -> I/Q digitizer.  The function intentionally refuses to model a real RF
    waveform; ``iq_fast`` is already the downconverted cavity output.
    """
    t_fast = np.asarray(t_fast, dtype=float)
    iq_fast = np.asarray(iq_fast, dtype=np.complex128)
    if t_fast.ndim != 1 or iq_fast.ndim != 1 or t_fast.size != iq_fast.size:
        raise ValueError("t_fast and iq_fast must be 1D arrays with equal length")
    D = max(int(decimation_factor), 1)
    fs_fast = float(fs_out_hz) * D
    cutoff_hz = float(lpf_cutoff_ratio) * 0.5 * float(fs_out_hz)

    mode = str(lpf_mode).lower()
    if exact_locust and mode == "fir_polyphase":
        mode = "fft_brickwall"
    if mode == "fft_brickwall":
        filtered = fft_brickwall_lpf_windows(iq_fast, fs_hz=fs_fast, cutoff_hz=cutoff_hz, n_windows=n_windows)
        iq = filtered[::D]
    elif mode == "fir_polyphase":
        iq = fir_polyphase_decimate(iq_fast, decimation_factor=D)
        iq = iq[: int(np.ceil(iq_fast.size / D))]
    elif mode == "none":
        iq = iq_fast[::D]
    else:
        raise ValueError("lpf_mode must be 'none', 'fft_brickwall', or 'fir_polyphase'")

    t = t_fast[::D][: iq.size]
    noise_rms = 0.0
    if add_noise:
        if noise_floor_psd_W_per_Hz is None:
            raise ValueError("noise_floor_psd_W_per_Hz is required when add_noise=True")
        noise = complex_white_noise(
            iq.size,
            fs_hz=float(fs_out_hz),
            noise_floor_psd_W_per_Hz=float(noise_floor_psd_W_per_Hz),
            impedance_ohm=float(impedance_ohm),
            rng=rng,
        )
        noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2))) if noise.size else 0.0
        iq = iq + noise

    adc_iq = None
    saturation_count = 0
    if digitizer_config is not None and bool(getattr(digitizer_config, "enabled", False)):
        dig: DigitizeResult = digitize_iq(
            iq,
            v_range=float(getattr(digitizer_config, "v_range", 1.0)),
            v_offset=float(getattr(digitizer_config, "v_offset", 0.0)),
            bit_depth=int(getattr(digitizer_config, "bit_depth", 8)),
            signed=bool(getattr(digitizer_config, "signed", False)),
            strict_range=bool(getattr(digitizer_config, "strict_range", True)),
        )
        adc_iq = dig.adc_iq
        saturation_count = int(dig.saturation_count)

    meta = {
        "readout_model": "locust_exact_baseband" if exact_locust else "locust_like_baseband",
        "generator_chain": "analytic-cavity-signal -> lpf-fft -> decimate-signal -> gaussian-noise(optional) -> digitizer(optional)" if exact_locust else "analytic-cavity-signal -> lpf -> decimate -> noise(optional) -> digitizer(optional)",
        "fs_fast_hz": fs_fast,
        "fs_out_hz": float(fs_out_hz),
        "decimation_factor": D,
        "lpf_mode": mode,
        "lpf_cutoff_hz": cutoff_hz,
        "lpf_cutoff_ratio_of_final_nyquist": float(lpf_cutoff_ratio),
        "lpf_n_windows": int(n_windows),
        "noise_enabled": bool(add_noise),
        "noise_rms": noise_rms,
        "digitizer_enabled": bool(digitizer_config is not None and getattr(digitizer_config, "enabled", False)),
        "adc_saturation_count": saturation_count,
        "analytic_baseband_only": True,
        "rf_undersampling_used": False,
    }
    return LocustReadoutResult(
        t_fast=t_fast if store_fast_iq else None,
        iq_fast=iq_fast if store_fast_iq else None,
        t=t,
        iq=iq,
        adc_iq=adc_iq,
        meta=meta,
    )
