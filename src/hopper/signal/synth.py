"""
Module: hopper.signal.synth

Developer: ehtkarim
Date: April 29, 2026

Synthesizes complex IQ time series from dynamic electron tracks and cavity response models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..cavity.cavity import Cavity
from ..cavity.mode_map import ModeMap
from ..cavity.resonance import ResonanceCurve
from ..cavity.response import BasebandCavityResponse, drive_work_power_W, make_cavity_response as make_response_operator
from ..config import MainConfig
from ..dynamics.track import DynamicTrack, sample_dynamic_track, _local_perp_basis_from_field
from ..dynamics.kinematics import gamma_beta_v_from_kinetic
from .. import constants as const
from ..field.field_map import FieldMap
from .sampling import estimate_sample_rate_hz, materialize_rf_time_grid, rf_time_grid_spec
from ..readout.locust_like import process_locust_like_readout


@dataclass
class SignalResult:
    # Full-rate RF-grid IQ arrays. In phase_uniform mode, this grid is uniform in
    # cyclotron phase and therefore usually non-uniform in time.
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

    # Tracks sampled onto output grids. track_rf may be None when RF ROOT output
    # is requested, because the output node streams it from the compact dynamics
    # track in chunks to avoid duplicating a very large RF track in memory.
    track_rf: Optional[DynamicTrack]
    track_if: DynamicTrack

    # Optional terminal readout diagnostics.
    adc_iq: Optional[np.ndarray] = None
    iq_fast: Optional[np.ndarray] = None
    t_fast: Optional[np.ndarray] = None
    readout_meta: Optional[dict[str, Any]] = None

    # If signal.normalize_power was applied, track amplitudes written later by a
    # streaming output path should be divided by this same factor.
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




def cavity_complex_response_enabled(cfg: MainConfig) -> bool:
    return bool(cfg.cavity.excitation_enabled) and str(getattr(cfg.cavity, "response_model", "time_evolution")) in {"time_evolution", "baseband_envelope"}


def make_cavity_response(cfg: MainConfig, f_lo_hz: float) -> BasebandCavityResponse:
    cav = Cavity(
        radius_m=float(cfg.cavity.radius_m),
        length_m=float(cfg.cavity.length_m),
        f0_hz=float(cfg.cavity.f0_hz),
        Q=float(cfg.cavity.Q),
    )
    return make_response_operator(
        response_model=str(getattr(cfg.cavity, "response_model", "time_evolution")),
        cavity=cav,
        lo_hz=float(f_lo_hz),
        output_coupling_fraction=float(getattr(cfg.cavity, "output_coupling_fraction", 1.0)),
        port_phase_rad=float(getattr(cfg.cavity, "port_phase_rad", 0.0)) + float(cfg.signal.carrier_phase0_rad),
        initial_energy_J=float(getattr(cfg.cavity, "initial_stored_energy_J", 0.0)),
        initial_phase_rad=float(getattr(cfg.cavity, "initial_cavity_phase_rad", 0.0)),
    )


def cavity_baseband_drive(
    cfg: MainConfig,
    track_sample: DynamicTrack,
    *,
    field: FieldMap,
    mode_map: ModeMap,
    f_lo_hz: float,
) -> np.ndarray:
    """Build the analytic baseband electron drive d(t) for the cavity mode.

    The production vector-map path evaluates the gyro-averaged q v·E_mode*
    coefficient on the sampled guiding-center state and multiplies it by
    exp(i[phase_rf + phase0 - omega_LO t]).  No RF voltage is materialized.
    """
    t = np.asarray(track_sample.t, dtype=float)
    if t.size == 0:
        return np.asarray([], dtype=np.complex128)

    gamma = np.asarray(gamma_beta_v_from_kinetic(track_sample.energy_eV)[0], dtype=float)
    if bool(getattr(mode_map, "is_vector_e_field", False)):
        if track_sample.axial_profile is not None:
            Br, Bphi, Bz = track_sample.axial_profile.components(track_sample.z_gc)
            B_gc = np.asarray(track_sample.axial_profile.B(track_sample.z_gc), dtype=float)
        else:
            Br, Bphi, Bz = field.components(track_sample.r_gc_m, track_sample.z_gc)
            B_gc = np.asarray(field.B(track_sample.r_gc_m, track_sample.z_gc), dtype=float)
        u1, u2, _ = _local_perp_basis_from_field(track_sample.phi_gc_rad, Br, Bphi, Bz)
        drive = mode_map.gyro_drive_coupling_W_per_sqrt_J(  # type: ignore[attr-defined]
            r_gc_m=np.asarray(track_sample.r_gc_m, dtype=float),
            phi_gc_rad=np.asarray(track_sample.phi_gc_rad, dtype=float),
            z_gc_m=np.asarray(track_sample.z_gc, dtype=float),
            B_T=B_gc,
            gamma=gamma,
            mu_J_per_T=np.asarray(track_sample.mu_J_per_T, dtype=float),
            u1=u1,
            u2=u2,
        ) * np.sqrt(max(float(getattr(cfg.cavity, "source_power_scale", 1.0)), 0.0))
    else:
        # Signed analytic TE011 fallback.  This is a coherent validation path,
        # not the stimulated-backreaction production model.
        coupling = np.asarray(mode_map(track_sample.r_gc_m, track_sample.z_gc), dtype=float)
        B = np.asarray(track_sample.B_T, dtype=float)
        v_perp = np.sqrt(np.maximum(2.0 * track_sample.mu_J_per_T * B / np.maximum(gamma * const.M_E, 1.0e-300), 0.0))
        drive = (-const.E_CHARGE) * v_perp * coupling * float(getattr(cfg.cavity, "source_power_scale", 1.0))

    phase = track_sample.phase_rf + float(cfg.electron.cyclotron_phase0_rad) - 2.0 * np.pi * float(f_lo_hz) * t
    return np.asarray(drive, dtype=np.complex128) * np.exp(1j * phase)


def _replace_track_signal_diagnostics(
    track: DynamicTrack,
    *,
    amp: np.ndarray,
    cavity_drive: np.ndarray,
    cavity_amplitude: np.ndarray,
    cavity_energy: np.ndarray,
    cavity_power: np.ndarray,
    cavity_work: np.ndarray | None = None,
) -> DynamicTrack:
    return DynamicTrack(
        t=track.t, x=track.x, y=track.y, z=track.z, vx=track.vx, vy=track.vy, vz=track.vz,
        x_gc=track.x_gc, y_gc=track.y_gc, z_gc=track.z_gc,
        vx_gc=track.vx_gc, vy_gc=track.vy_gc, vz_gc=track.vz_gc,
        r_gc_m=track.r_gc_m, phi_gc_rad=track.phi_gc_rad, parallel_sign=track.parallel_sign,
        b_cross_kappa_phi_per_m=track.b_cross_kappa_phi_per_m,
        f_c_hz=track.f_c_hz, amp=np.asarray(amp, dtype=float), phase_rf=track.phase_rf, B_T=track.B_T,
        energy_eV=track.energy_eV, mu_J_per_T=track.mu_J_per_T, axial_profile=track.axial_profile,
        cavity_energy_J=np.asarray(cavity_energy, dtype=float),
        cavity_power_W=np.asarray(cavity_power, dtype=float),
        cavity_source_power_W=np.abs(cavity_drive) ** 2,
        cavity_work_power_W=None if cavity_work is None else np.asarray(cavity_work, dtype=float),
        solver_info=track.solver_info,
        cavity_amplitude_sqrt_J=np.asarray(cavity_amplitude, dtype=np.complex128),
        cavity_drive_sqrt_J_per_s=np.asarray(cavity_drive, dtype=np.complex128),
    )


def apply_complex_cavity_response_to_track(
    cfg: MainConfig,
    track_sample: DynamicTrack,
    *,
    field: FieldMap,
    mode_map: ModeMap,
    f_lo_hz: float,
    initial_amplitude_sqrt_J: complex | None = None,
) -> tuple[np.ndarray, DynamicTrack, np.ndarray]:
    response = make_cavity_response(cfg, f_lo_hz)
    drive = cavity_baseband_drive(cfg, track_sample, field=field, mode_map=mode_map, f_lo_hz=f_lo_hz)
    if initial_amplitude_sqrt_J is None:
        initial = response.initial_amplitude_sqrt_J
    else:
        initial = complex(initial_amplitude_sqrt_J)
    from ..cavity.response import integrate_complex_envelope
    a = integrate_complex_envelope(
        track_sample.t,
        drive,
        lambda_per_s=response.lambda_per_s,
        initial_amplitude_sqrt_J=initial,
        update=cfg.signal.cavity_update,
    )
    y = response.output_from_amplitude(a)
    amp = np.abs(y)
    track_out = _replace_track_signal_diagnostics(
        track_sample,
        amp=amp,
        cavity_drive=drive,
        cavity_amplitude=a,
        cavity_energy=np.abs(a) ** 2,
        cavity_power=np.abs(y) ** 2,
        cavity_work=drive_work_power_W(a, drive),
    )
    return y, track_out, a

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
        axial_profile=track.axial_profile,
        cavity_energy_J=track.cavity_energy_J,
        cavity_power_W=track.cavity_power_W,
        cavity_source_power_W=track.cavity_source_power_W,
        cavity_work_power_W=getattr(track, "cavity_work_power_W", None),
        solver_info=track.solver_info,
        cavity_amplitude_sqrt_J=getattr(track, "cavity_amplitude_sqrt_J", None),
        cavity_drive_sqrt_J_per_s=getattr(track, "cavity_drive_sqrt_J_per_s", None),
    )


def _root_wants_rf_track(cfg: MainConfig) -> bool:
    sampling = str(getattr(cfg.output, "track_sampling", "rf_sampled")).lower()
    return bool(cfg.output.write_root) and sampling == "rf_sampled"



def _fast_baseband_grid(cfg: MainConfig) -> tuple[np.ndarray, float, int, float]:
    fs_out = float(cfg.signal.fs_if_hz)
    D = max(int(cfg.readout.fast_decimation_factor), 1)
    fs_fast = fs_out * D
    n_fast = int(np.floor(float(cfg.simulation.track_length_s) * fs_fast))
    t_fast = float(cfg.simulation.starting_time_s) + np.arange(max(n_fast, 0), dtype=float) / fs_fast
    if t_fast.size == 0:
        t_fast = np.asarray([float(cfg.simulation.starting_time_s)], dtype=float)
    return t_fast, fs_out, D, fs_fast


def _default_lo_for_tracks(tracks: Sequence[DynamicTrack]) -> float:
    vals = [np.asarray(track.f_c_hz, dtype=float) for track in tracks if np.asarray(track.f_c_hz).size]
    if not vals:
        return 0.0
    return float(np.mean(np.concatenate(vals)))




def _interp_complex(t_new: np.ndarray, t_old: np.ndarray, values: np.ndarray) -> np.ndarray:
    t_new = np.asarray(t_new, dtype=float)
    t_old = np.asarray(t_old, dtype=float)
    v = np.asarray(values, dtype=np.complex128)
    if t_new.size == 0:
        return np.asarray([], dtype=np.complex128)
    if t_old.size == 0 or v.size == 0:
        return np.zeros(t_new.size, dtype=np.complex128)
    return np.interp(t_new, t_old, np.real(v)) + 1j * np.interp(t_new, t_old, np.imag(v))


def _track_with_readout_diagnostics(
    cfg: MainConfig,
    base_track: DynamicTrack,
    *,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve,
    t_out: np.ndarray,
    t_source: np.ndarray,
    drive_source: np.ndarray,
    amplitude_source: np.ndarray,
    iq_out: np.ndarray,
) -> DynamicTrack:
    """Create an output-grid track without recomputing expensive vector coupling."""
    t_out = np.asarray(t_out, dtype=float)
    if t_out.size == base_track.t.size and np.array_equal(t_out, base_track.t):
        track = base_track
    else:
        track = sample_dynamic_track(cfg, base_track, field=field, mode_map=mode_map, resonance=resonance, t_new=t_out)
    drive = _interp_complex(t_out, t_source, drive_source)
    amp_state = _interp_complex(t_out, t_source, amplitude_source)
    iq_arr = np.asarray(iq_out, dtype=np.complex128)
    if iq_arr.size != t_out.size:
        iq_arr = _interp_complex(t_out, t_out[:iq_arr.size], iq_arr) if iq_arr.size else np.zeros_like(t_out, dtype=np.complex128)
    return _replace_track_signal_diagnostics(
        track,
        amp=np.abs(iq_arr),
        cavity_drive=drive,
        cavity_amplitude=amp_state,
        cavity_energy=np.abs(amp_state) ** 2,
        cavity_power=np.abs(iq_arr) ** 2,
        cavity_work=drive_work_power_W(amp_state, drive),
    )


def synthesize_iq_pileup(
    cfg: MainConfig,
    tracks_dyn: Sequence[DynamicTrack],
    *,
    field: FieldMap,
    mode_map: ModeMap,
    resonance: ResonanceCurve,
) -> SignalResult:
    """Generate a coherent shared-cavity pileup signal from multiple tracks.

    Each electron contributes a complex analytic baseband drive d_j(t).  The drives
    are summed first and then filtered once by the single cavity mode, as required
    by the linear input-output model.  This avoids adding powers or filtering each
    RF-scale track separately.
    """
    tracks = list(tracks_dyn)
    if not tracks:
        raise ValueError("at least one DynamicTrack is required for pileup synthesis")
    if len(tracks) == 1:
        return synthesize_iq(cfg, tracks[0], field=field, mode_map=mode_map, resonance=resonance)
    if str(getattr(cfg.readout, "model", "none")) not in {"locust_exact_baseband", "locust_like_baseband"}:
        raise ValueError("multi-track pileup currently requires readout.model='locust_exact_baseband' or 'locust_like_baseband'")
    if not cavity_complex_response_enabled(cfg):
        raise ValueError("multi-track pileup requires a coherent cavity response model")

    t_fast, fs_out, D, fs_fast = _fast_baseband_grid(cfg)
    f_lo = float(cfg.signal.lo_hz) if cfg.signal.lo_hz is not None else _default_lo_for_tracks(tracks)
    drive_total = np.zeros(t_fast.size, dtype=np.complex128)
    sampled_first: DynamicTrack | None = None
    max_fc_offset_hz = 0.0
    for track in tracks:
        sampled = sample_dynamic_track(cfg, track, field=field, mode_map=mode_map, resonance=resonance, t_new=t_fast)
        if sampled_first is None:
            sampled_first = sampled
        drive_total += cavity_baseband_drive(cfg, sampled, field=field, mode_map=mode_map, f_lo_hz=f_lo)
        if sampled.f_c_hz.size:
            max_fc_offset_hz = max(max_fc_offset_hz, float(np.max(np.abs(np.asarray(sampled.f_c_hz, dtype=float) - f_lo))))

    response = make_cavity_response(cfg, f_lo)
    from ..cavity.response import integrate_complex_envelope
    amp_state = integrate_complex_envelope(
        t_fast,
        drive_total,
        lambda_per_s=response.lambda_per_s,
        initial_amplitude_sqrt_J=response.initial_amplitude_sqrt_J,
        update=cfg.signal.cavity_update,
    )
    iq_fast = response.output_from_amplitude(amp_state)
    usable_band_hz = float(cfg.readout.lpf.cutoff_ratio_of_final_nyquist) * 0.5 * fs_out
    if bool(cfg.signal.require_analytic_baseband_drive) and max_fc_offset_hz > usable_band_hz * (1.0 + float(cfg.signal.if_bandwidth_tolerance)):
        import warnings
        warnings.warn(
            "estimated cyclotron carrier offset exceeds the usable baseband readout band; "
            "increase signal.fs_if_hz, move signal.lo_hz, or use a wider readout filter",
            RuntimeWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(cfg.readout.noise.seed) if cfg.readout.noise.seed is not None else None
    readout_res = process_locust_like_readout(
        t_fast=t_fast,
        iq_fast=iq_fast,
        fs_out_hz=fs_out,
        decimation_factor=D,
        lpf_cutoff_ratio=float(cfg.readout.lpf.cutoff_ratio_of_final_nyquist),
        lpf_mode=cfg.readout.lpf.type,
        n_windows=int(cfg.readout.lpf.n_windows),
        add_noise=bool(cfg.readout.noise.enabled),
        noise_floor_psd_W_per_Hz=cfg.readout.noise.noise_floor_psd_W_per_Hz,
        impedance_ohm=float(cfg.readout.noise.impedance_ohm),
        rng=rng,
        digitizer_config=cfg.readout.digitizer,
        store_fast_iq=bool(cfg.readout.store_fast_iq),
        exact_locust=(str(cfg.readout.model) == "locust_exact_baseband"),
    )

    track_if = sample_dynamic_track(cfg, tracks[0], field=field, mode_map=mode_map, resonance=resonance, t_new=readout_res.t)
    drive_if = np.interp(readout_res.t, t_fast, np.real(drive_total)) + 1j * np.interp(readout_res.t, t_fast, np.imag(drive_total))
    amp_if = np.interp(readout_res.t, t_fast, np.real(amp_state)) + 1j * np.interp(readout_res.t, t_fast, np.imag(amp_state))
    y_if = response.output_from_amplitude(amp_if)
    track_if = _replace_track_signal_diagnostics(
        track_if,
        amp=np.abs(y_if),
        cavity_drive=drive_if,
        cavity_amplitude=amp_if,
        cavity_energy=np.abs(amp_if) ** 2,
        cavity_power=np.abs(y_if) ** 2,
    )
    readout_meta = dict(readout_res.meta)
    readout_meta.update({
        "pileup_tracks": len(tracks),
        "pileup_combination": "coherent_drive_sum_before_single_cavity_filter",
        "cavity_response_model": str(cfg.cavity.response_model),
        "baseband_max_cyclotron_offset_hz": max_fc_offset_hz,
        "baseband_usable_band_hz": usable_band_hz,
        "baseband_carrier_offset_within_band": bool(max_fc_offset_hz <= usable_band_hz * (1.0 + float(cfg.signal.if_bandwidth_tolerance))),
    })
    return SignalResult(
        t=t_fast,
        iq=iq_fast,
        f_lo_hz=f_lo,
        fs_hz=fs_fast,
        rf_grid_kind="locust_exact_baseband_fast" if str(cfg.readout.model) == "locust_exact_baseband" else "locust_like_baseband_fast",
        rf_grid_is_uniform_time=True,
        t_if=readout_res.t,
        iq_if=readout_res.iq,
        fs_if_hz=fs_out,
        track_rf=None,
        track_if=track_if,
        adc_iq=readout_res.adc_iq,
        iq_fast=readout_res.iq_fast,
        t_fast=readout_res.t_fast,
        readout_meta=readout_meta,
        amplitude_normalization=1.0,
    )

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

    if str(getattr(cfg.readout, "model", "none")) in {"locust_exact_baseband", "locust_like_baseband"}:
        if not bool(getattr(cfg.readout, "require_analytic_baseband_drive", True)):
            raise ValueError("Locust-style readout requires analytic baseband drive generation")
        t_fast, fs_out, D, fs_fast = _fast_baseband_grid(cfg)
        track_fast = sample_dynamic_track(
            cfg, track_dyn, field=field, mode_map=mode_map, resonance=resonance, t_new=t_fast
        )
        f_lo = float(sig.lo_hz) if sig.lo_hz is not None else float(np.mean(track_fast.f_c_hz))
        iq_fast, track_fast, amp_state = apply_complex_cavity_response_to_track(
            cfg, track_fast, field=field, mode_map=mode_map, f_lo_hz=f_lo
        )
        usable_band_hz = float(cfg.readout.lpf.cutoff_ratio_of_final_nyquist) * 0.5 * fs_out
        max_fc_offset_hz = float(np.max(np.abs(np.asarray(track_fast.f_c_hz, dtype=float) - f_lo))) if track_fast.f_c_hz.size else 0.0
        if bool(sig.require_analytic_baseband_drive) and max_fc_offset_hz > usable_band_hz * (1.0 + float(sig.if_bandwidth_tolerance)):
            import warnings
            warnings.warn(
                "estimated cyclotron carrier offset exceeds the usable baseband readout band; "
                "increase signal.fs_if_hz, move signal.lo_hz, or use a wider readout filter",
                RuntimeWarning,
                stacklevel=2,
            )

        rng = np.random.default_rng(cfg.readout.noise.seed) if cfg.readout.noise.seed is not None else None
        readout_res = process_locust_like_readout(
            t_fast=t_fast,
            iq_fast=iq_fast,
            fs_out_hz=fs_out,
            decimation_factor=D,
            lpf_cutoff_ratio=float(cfg.readout.lpf.cutoff_ratio_of_final_nyquist),
            lpf_mode=cfg.readout.lpf.type,
            n_windows=int(cfg.readout.lpf.n_windows),
            add_noise=bool(cfg.readout.noise.enabled),
            noise_floor_psd_W_per_Hz=cfg.readout.noise.noise_floor_psd_W_per_Hz,
            impedance_ohm=float(cfg.readout.noise.impedance_ohm),
            rng=rng,
            digitizer_config=cfg.readout.digitizer,
            store_fast_iq=bool(cfg.readout.store_fast_iq),
            exact_locust=(str(cfg.readout.model) == "locust_exact_baseband"),
        )
        track_if = _track_with_readout_diagnostics(
            cfg,
            track_fast,
            field=field,
            mode_map=mode_map,
            resonance=resonance,
            t_out=readout_res.t,
            t_source=t_fast,
            drive_source=track_fast.cavity_drive_sqrt_J_per_s if track_fast.cavity_drive_sqrt_J_per_s is not None else np.zeros_like(t_fast, dtype=np.complex128),
            amplitude_source=amp_state,
            iq_out=readout_res.iq,
        )
        readout_meta = dict(readout_res.meta)
        readout_meta.update({
            "baseband_max_cyclotron_offset_hz": max_fc_offset_hz,
            "baseband_usable_band_hz": usable_band_hz,
            "baseband_carrier_offset_within_band": bool(max_fc_offset_hz <= usable_band_hz * (1.0 + float(sig.if_bandwidth_tolerance))),
            "cavity_response_model": str(cfg.cavity.response_model),
            "drive_grid_samples": int(t_fast.size),
            "drive_grid_hz": float(fs_fast),
            "redundant_cavity_resampling_removed": True,
        })
        return SignalResult(
            t=t_fast,
            iq=iq_fast,
            f_lo_hz=f_lo,
            fs_hz=fs_fast,
            rf_grid_kind=("locust_exact_baseband_fast" if str(cfg.readout.model) == "locust_exact_baseband" else "locust_like_baseband_fast"),
            rf_grid_is_uniform_time=True,
            t_if=readout_res.t,
            iq_if=readout_res.iq,
            fs_if_hz=fs_out,
            track_rf=None,
            track_if=track_if,
            adc_iq=readout_res.adc_iq,
            iq_fast=readout_res.iq_fast,
            t_fast=readout_res.t_fast,
            readout_meta=readout_meta,
            amplitude_normalization=1.0,
        )

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

    if cavity_complex_response_enabled(cfg):
        iq, track_rf_full, _cavity_amp = apply_complex_cavity_response_to_track(
            cfg,
            track_rf_full,
            field=field,
            mode_map=mode_map,
            f_lo_hz=f_lo,
        )
    else:
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
            # Digital anti-alias filters assume uniform time samples. For phase_uniform
            # the grid is non-uniform, so decimation means stride-thinning the phase
            # samples. Users needing filtered IF data should use phase_bounded or an
            # explicit uniform signal.fs_hz grid.
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
