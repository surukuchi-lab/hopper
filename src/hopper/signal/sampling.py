from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..config import MainConfig
from ..dynamics.track import DynamicTrack


@dataclass(frozen=True)
class TimeGridSpec:
    """
    Description of the RF-output sampling grid.

    phase_uniform intentionally samples at fixed cyclotron phase increments, so its
    time values are generally non-uniform. The grid is represented by index ranges
    so large RF-sampled ROOT tracks can be streamed in chunks without first
    materializing the full time vector or all track branches in memory.
    """

    kind: str
    n: int
    t0_s: float
    t_end_s: float
    fs_hz: float
    dt_s: Optional[float] = None
    phase0_rad: Optional[float] = None
    phase_step_rad: Optional[float] = None
    phase_end_rad: Optional[float] = None

    @property
    def is_uniform_time(self) -> bool:
        return self.kind == "uniform_time"


def configured_rf_fs_hz(cfg: MainConfig) -> float:
    """Return the configured full-rate RF sample rate before IF decimation."""
    sig = cfg.signal
    M = int(sig.if_decim)
    if M < 1:
        raise ValueError("signal.if_decim must be >= 1")

    if sig.fs_hz is not None:
        fs = float(sig.fs_hz)
    else:
        fs = float(sig.fs_if_hz) * M

    if fs <= 0.0:
        raise ValueError("RF sampling rate must be positive")
    return fs


def _samples_per_turn(cfg: MainConfig) -> Optional[int]:
    samples_per_turn = getattr(cfg.dynamics, "samples_per_cyclotron_turn", None)
    if samples_per_turn is None:
        return None
    n = int(samples_per_turn)
    if n <= 1:
        raise ValueError("dynamics.samples_per_cyclotron_turn must be >= 2.")
    return n


def _compact_phase_time(track: DynamicTrack) -> tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(track.phase_rf, dtype=float)
    time = np.asarray(track.t, dtype=float)
    if phase.ndim != 1 or time.ndim != 1 or phase.size != time.size:
        raise ValueError("track.t and track.phase_rf must be 1D arrays with equal length")
    if phase.size < 2:
        raise ValueError("track must contain at least two samples")

    keep = np.ones_like(phase, dtype=bool)
    keep[1:] = np.diff(phase) > 0.0
    phase_u = phase[keep]
    time_u = time[keep]
    if phase_u.size < 2 or phase_u[-1] <= phase_u[0]:
        raise ValueError("track.phase_rf must increase to build a phase-uniform sampling grid")
    return phase_u, time_u


def rf_time_grid_spec(cfg: MainConfig, track: DynamicTrack) -> TimeGridSpec:
    """
    Build the RF-output grid specification.

    - axial_adaptive: use the configured uniform-time RF sampling rate.
    - phase_bounded: use a uniform-time RF grid fast enough that the maximum cyclotron phase advance per sample is <= 2π / samples_per_cyclotron_turn.
    - phase_uniform: sample exactly every 2π / samples_per_cyclotron_turn in accumulated cyclotron phase. The resulting time grid is non-uniform.
    """
    t0 = float(cfg.simulation.starting_time_s)
    duration = float(cfg.simulation.track_length_s)
    t_end = t0 + duration
    base_fs = configured_rf_fs_hz(cfg)

    strategy = str(getattr(cfg.dynamics, "time_step_strategy", "axial_adaptive"))
    n_per = _samples_per_turn(cfg)

    if strategy == "phase_uniform" and n_per is not None:
        phase, time = _compact_phase_time(track)
        phase_step = (2.0 * np.pi) / float(n_per)
        phase_span = float(phase[-1] - phase[0])
        n = int(np.floor(phase_span / phase_step)) + 1
        n = max(2, n)
        fs_nominal = (n - 1) / max(float(time[-1] - time[0]), 1e-30)
        return TimeGridSpec(
            kind="uniform_phase",
            n=n,
            t0_s=t0,
            t_end_s=t_end,
            fs_hz=float(fs_nominal),
            phase0_rad=float(phase[0]),
            phase_step_rad=float(phase_step),
            phase_end_rad=float(phase[0] + (n - 1) * phase_step),
        )

    if strategy == "phase_bounded" and n_per is not None:
        fc = np.asarray(track.f_c_hz, dtype=float)
        if fc.size == 0:
            raise ValueError("track.f_c_hz is empty")
        fs_required = float(n_per) * float(np.nanmax(fc))
        fs = max(base_fs, fs_required)
    elif strategy in {"axial_adaptive", "phase_bounded", "phase_uniform"}:
        fs = base_fs
    else:
        raise ValueError(
            f"Unknown dynamics.time_step_strategy={strategy!r}; expected "
            "'axial_adaptive', 'phase_bounded', or 'phase_uniform'."
        )

    n = int(np.floor(duration * fs))
    n = max(2, n)
    return TimeGridSpec(
        kind="uniform_time",
        n=n,
        t0_s=t0,
        t_end_s=t_end,
        fs_hz=float(fs),
        dt_s=1.0 / float(fs),
    )


def _materialize_indices(cfg: MainConfig, track: DynamicTrack, spec: TimeGridSpec, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be 1D")
    if idx.size == 0:
        return np.asarray([], dtype=float)

    if spec.kind == "uniform_time":
        assert spec.dt_s is not None
        return float(spec.t0_s) + float(spec.dt_s) * idx.astype(float)

    if spec.kind == "uniform_phase":
        assert spec.phase0_rad is not None and spec.phase_step_rad is not None
        phase, time = _compact_phase_time(track)
        target_phase = float(spec.phase0_rad) + float(spec.phase_step_rad) * idx.astype(float)
        return np.interp(target_phase, phase, time)

    raise ValueError(f"Unknown time-grid kind {spec.kind!r}")


def materialize_rf_time_grid(cfg: MainConfig, track: DynamicTrack, spec: Optional[TimeGridSpec] = None) -> np.ndarray:
    """Materialize the full RF-output time grid."""
    if spec is None:
        spec = rf_time_grid_spec(cfg, track)
    return _materialize_indices(cfg, track, spec, np.arange(spec.n, dtype=np.int64))


def materialize_if_time_grid(cfg: MainConfig, track: DynamicTrack, spec: Optional[TimeGridSpec] = None) -> np.ndarray:
    """Materialize the decimated IF-output time grid by taking every if_decim-th RF sample."""
    if spec is None:
        spec = rf_time_grid_spec(cfg, track)
    M = int(cfg.signal.if_decim)
    if M < 1:
        raise ValueError("signal.if_decim must be >= 1")
    return _materialize_indices(cfg, track, spec, np.arange(0, spec.n, M, dtype=np.int64))


def iter_rf_time_grid_chunks(
    cfg: MainConfig,
    track: DynamicTrack,
    *,
    chunk_size: int,
    spec: Optional[TimeGridSpec] = None,
) -> Iterator[np.ndarray]:
    """Yield RF-output time samples in chunks."""
    if spec is None:
        spec = rf_time_grid_spec(cfg, track)
    chunk = max(1, int(chunk_size))
    for start in range(0, int(spec.n), chunk):
        stop = min(int(spec.n), start + chunk)
        yield _materialize_indices(cfg, track, spec, np.arange(start, stop, dtype=np.int64))


def iter_rf_time_grid_index_chunks(
    cfg: MainConfig,
    track: DynamicTrack,
    *,
    chunk_size: int,
    spec: Optional[TimeGridSpec] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(indices, time_samples)`` for the RF-output grid in chunks."""
    if spec is None:
        spec = rf_time_grid_spec(cfg, track)
    chunk = max(1, int(chunk_size))
    for start in range(0, int(spec.n), chunk):
        stop = min(int(spec.n), start + chunk)
        idx = np.arange(start, stop, dtype=np.int64)
        yield idx, _materialize_indices(cfg, track, spec, idx)


def estimate_sample_rate_hz(t_s: np.ndarray) -> float:
    """Average sample rate for either a uniform or non-uniform grid."""
    t = np.asarray(t_s, dtype=float)
    if t.size < 2:
        return 0.0
    span = float(t[-1] - t[0])
    if span <= 0.0:
        return 0.0
    return float((t.size - 1) / span)
