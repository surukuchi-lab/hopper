"""
Module: hopper.config

Developer: ehtkarim
Date: April 29, 2026

Defines typed configuration models, default values, semantic validation, and YAML loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

InterpMethod = Literal["bilinear", "linear", "nearest", "cubic_spline"]
AmplitudeMode = Literal["sqrtP", "P"]
TrackSampling = Literal["rf_sampled", "if_sampled"]


@dataclass
class SimulationConfig:
    starting_time_s: float = 0.0
    track_length_s: float = 1.0e-3




@dataclass
class PhysicsConfig:
    constants_preset: Literal[
        "pdg_2022",
        "pdg_2021",
        "locust_kassiopeia_2021",
        "locust_kassiopeia_2006",
    ] = "pdg_2022"


@dataclass
class FeatureConfig:
    include_true_orbit: bool = False
    include_gradB: bool = True
    # The current curvature-drift implementation is available as an optional correction,
    # but it remains off by default until it is fully validated against the high-fidelity
    # reference tracks for this trap.
    include_curvature_drift: bool = False
    include_resonance: bool = False
    amplitude_mode: AmplitudeMode = "sqrtP"
    normalize_amp_at_t0: bool = False


@dataclass
class TrapGridConfig:
    r_max_m: float = 0.327
    z_min_m: float = -2.0
    z_max_m: float = 2.0
    n_r: int = 201
    n_z: int = 801


@dataclass
class TrapConfig:
    field_map_npz: str = "field_map_rz_components.npz"
    coil_xml: Optional[str] = None
    generate_if_missing: bool = False
    placeholder_if_missing: bool = True
    grid: TrapGridConfig = field(default_factory=TrapGridConfig)
    interpolation: InterpMethod = "bilinear"
    clamp_to_grid: bool = True


@dataclass
class CavityConfig:
    radius_m: float = 0.327
    length_m: float = 4.0
    f0_hz: float = 560.3e6
    Q: float = 500.0

    # Coherent single-mode cavity response. The time_evolution operator is
    # the Locust-compatible damped-resonator default; baseband_envelope is the
    # narrowband energy-normalized production equivalent. Scalar power-envelope
    # response modes are intentionally removed.
    response_model: Literal["time_evolution", "baseband_envelope"] = "time_evolution"
    output_coupling_fraction: float = 1.0
    port_phase_rad: float = 0.0
    initial_cavity_phase_rad: float = 0.0
    extend_ringdown_s: Optional[float] = None

    # Cavity excitation / back-reaction controls.  With a vector E-field map,
    # the source and stimulated work are computed from the complex q v·E_mode*
    # drive.  Scalar source-power approximations are not used for the coherent IQ
    # path and are disallowed for stimulated back-reaction by default.
    excitation_enabled: bool = True
    ringup_enabled: bool = True
    back_reaction_enabled: bool = False
    stimulated_back_reaction: bool = False
    back_reaction_requires_vector_map: bool = True
    mode_volume_m3: Optional[float] = None
    source_power_scale: float = 1.0
    back_reaction_scale: float = 1.0
    initial_stored_energy_J: float = 0.0


@dataclass
class ModeMapConfig:
    # "auto" attempts to load vector_e_field_map when present; otherwise it falls
    # back to the analytic scalar TE_011 map.
    type: Literal["auto", "analytic_te011", "te011", "analytic", "vector_e_field", "vector"] = "auto"
    vector_e_field_map: Optional[str] = None
    vector_component_basis: Literal["cartesian", "cylindrical", "auto"] = "cartesian"
    vector_field_unit_scale: float = 1.0
    vector_energy_normalization_J: float = 1.0
    vector_gyro_quadrature_points: int = 16
    vector_normalize_to_peak: bool = True
    vector_bounds_policy: Literal["error", "warn", "zero"] = "error"
    vector_cache_enabled: bool = True
    vector_cache_path: Optional[str] = None
    vector_cache_rebuild: bool = False
    fallback_to_analytic: bool = True


@dataclass
class ResonanceConfig:
    resonance_curve: Optional[str] = None
    object_name: Optional[str] = None
    normalize_to_peak: bool = True


@dataclass
class ElectronConfig:
    position_reference: Literal["guiding_center", "instantaneous"] = "guiding_center"
    position_coordinates: Literal["auto", "cylindrical", "cartesian"] = "auto"
    energy_eV: float = 18_563.251
    pitch_angle_deg: float = 89.0
    r0_m: float = 0.16
    phi0_rad: float = 0.0
    z0_m: float = 0.0
    x0_m: Optional[float] = None
    y0_m: Optional[float] = None
    vpar_sign: int = 1
    cyclotron_phase0_rad: float = 0.0


@dataclass(frozen=True)
class DynamicsConfig:
    # Existing axial solver controls
    dt_max_s: float = 2e-9
    dt_min_s: float = 1e-12
    safety: float = 0.98
    v_turn_threshold_c: float = 1e-5

    # Axial time evolution strategy
    axial_strategy: Literal["direct", "template_tiling"] = "template_tiling"

    # How to build the single-bounce template (only used when axial_strategy=template_tiling)
    template_build: Literal["auto", "integrate", "mirror", "mirror_quadrature"] = "mirror_quadrature"

    # Template detection controls (integrate build)
    template_return_z_tol_m: float = 1e-7
    template_max_duration_s: float = 5e-3
    template_min_reflections: int = 2

    # Mirror-extension validity checks (mirror build). The mirror builder supports arbitrary
    # starting z by phase-shifting a symmetric midplane-to-turning-point template.
    mirror_z0_tol_m: float = 1e-6
    mirror_symmetry_check: bool = True
    mirror_symmetry_rel_tol: float = 1e-3
    mirror_symmetry_ncheck: int = 25

    # Quadrature/action-time mirror builder controls.  The quadrature builder is the
    # preferred symmetric-template mode because it computes the mirror clock from the
    # regularized axial time-of-flight integral rather than from a time-marched Euler
    # turning step.
    mirror_quadrature_min_theta_nodes: int = 513
    mirror_quadrature_max_theta_nodes: int = 8193
    mirror_template_max_period_rel_error: float = 1.0e-3
    mirror_template_max_phase_slip_rad: float = 5.0e-3
    mirror_template_moving_mirror_tol: float = 1.0e-8

    # Vectorized complex back-reaction controls.
    back_reaction_predictor_corrector: bool = True
    multi_bounce_auto_max_bounces: int = 8

    # Energy loss / chirp model.
    # - "none": constant energy and constant µ
    # - "per_bounce": integrate one full bounce at a time on the cached field line using
    #   the exact local radiation step; the mirror-template machinery is used only to seed
    #   the bounce-period guess, so the radiatively moving mirror remains self-consistent
    # - "linear_fit": fast approximation; fit linear dE/dt and dµ/dt from several
    #   evolving-(γ, µ) bounces, then apply those rates on a tiled template
    # - "analytic": continuously apply the exact local radiation step throughout the
    #   axial integration, evolving both γ and µ and therefore the reference pitch angle
    energy_loss_model: Literal["none", "per_bounce", "linear_fit", "analytic"] = "per_bounce"
    energy_loss_scale: float = 1.0
    energy_floor_eV: float = 1.0

    # Number of full bounce returns to integrate continuously before restarting the
    # per_bounce compact radiative solve. Values >1 reduce the small return-stitching
    # phase error while preserving the cached 1D field-line efficiency.
    per_bounce_block_bounces: int = 1
    back_reaction_block_vectorized: bool = True
    back_reaction_block_max_rel_state_change: float = 2.0e-3
    back_reaction_max_updates_per_bounce: int = 128

    # Optional output grid for the compact dynamics record.  ``null`` keeps the
    # internal solver grid unless Hopper can safely infer a coarser grid from the
    # baseband readout rate.  This does not alter the mirror-template construction;
    # it only compresses the exported compact state used by signal/readout sampling.
    compact_output_dt_s: Optional[float] = None
    compact_output_include_turning_points: bool = True

    # Used only when energy_loss_model="linear_fit"
    energy_loss_fit_bounces: int = 6

    # Frequency/phase reference frame used by the gyro phase, reported B_T/f_c_hz,
    # and signal phase.  "guiding_center" is the asymptotic guiding-center model.
    # "instantaneous" iterates the true orbit locally and uses the orbit-point field
    # consistently for phase/frequency when include_true_orbit=True.
    cyclotron_frequency_reference: Literal["guiding_center", "instantaneous"] = "guiding_center"
    true_orbit_phase_iterations: int = 1

    # RF/output time-grid strategy.  The compact dynamics track is not expanded internally;
    # phase-aware sampling is applied at the signal/ROOT-output boundary.
    # - "axial_adaptive": use the configured uniform-time signal RF sampling rate
    # - "phase_bounded": use a uniform-time RF grid fast enough to keep the maximum
    #   cyclotron phase advance <= 2π / samples_per_cyclotron_turn
    # - "phase_uniform": sample at exactly uniform cyclotron phase increments
    #   Δφ = 2π / samples_per_cyclotron_turn; this time grid is generally non-uniform
    time_step_strategy: Literal["axial_adaptive", "phase_bounded", "phase_uniform"] = "axial_adaptive"

    # Target number of RF/output samples per cyclotron turn for phase_bounded/phase_uniform.
    samples_per_cyclotron_turn: Optional[int] = None



@dataclass
class SignalConfig:
    lo_hz: Optional[float] = None
    fs_if_hz: float = 22.0e6
    if_decim: int = 1
    fs_hz: Optional[float] = None
    if_antialias_filter: bool = True
    if_filter_order: int = 8
    if_filter_cutoff_ratio: float = 0.9
    carrier_phase0_rad: float = 0.0
    normalize_power: bool = False

    # Cavity IQ generation.  The safe production path builds the analytic
    # baseband electron drive, applies the complex cavity response in baseband,
    # and only then performs readout filtering/decimation.
    cavity_filter_grid: Literal["if", "rf"] = "if"
    cavity_update: Literal["zero_order_hold", "first_order_hold"] = "first_order_hold"
    require_analytic_baseband_drive: bool = True
    if_bandwidth_tolerance: float = 1.0e-3


@dataclass
class ReadoutLpfConfig:
    type: Literal["none", "fft_brickwall", "fir_polyphase"] = "none"
    cutoff_ratio_of_final_nyquist: float = 0.85
    n_windows: int = 80


@dataclass
class ReadoutNoiseConfig:
    enabled: bool = False
    noise_floor_psd_W_per_Hz: Optional[float] = None
    impedance_ohm: float = 50.0
    seed: Optional[int] = None


@dataclass
class DigitizerConfig:
    enabled: bool = False
    bit_depth: int = 8
    data_type_size: int = 1
    signed: bool = False
    v_range: float = 1.0
    v_offset: float = 0.0
    strict_range: bool = True


@dataclass
class ReadoutConfig:
    model: Literal["none", "locust_exact_baseband", "locust_like_baseband"] = "none"
    require_analytic_baseband_drive: bool = True
    fast_decimation_factor: int = 1
    store_fast_iq: bool = False
    chunk_size: int = 2_000_000
    lpf: ReadoutLpfConfig = field(default_factory=ReadoutLpfConfig)
    noise: ReadoutNoiseConfig = field(default_factory=ReadoutNoiseConfig)
    digitizer: DigitizerConfig = field(default_factory=DigitizerConfig)


@dataclass
class CampaignConfig:
    enabled: bool = False
    parameter_ranges: Dict[str, list[Any]] = field(default_factory=dict)
    slurm_partition: Optional[str] = None
    slurm_time: str = "01:00:00"
    slurm_cpus_per_task: int = 1
    slurm_mem_per_cpu: Optional[str] = None
    max_jobs: Optional[int] = None


@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    basename: str = "sim"
    write_npz: bool = True
    write_root: bool = True
    track_sampling: TrackSampling = "rf_sampled"
    root_chunk_size: int = 200_000
    write_log: bool = True
    log_file: Optional[str] = None


@dataclass
class MainConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    trap: TrapConfig = field(default_factory=TrapConfig)
    cavity: CavityConfig = field(default_factory=CavityConfig)
    mode_map: ModeMapConfig = field(default_factory=ModeMapConfig)
    resonance: ResonanceConfig = field(default_factory=ResonanceConfig)
    electron: ElectronConfig = field(default_factory=ElectronConfig)
    # Optional pileup tracks. If empty, the single `electron` block is used.
    tracks: list[ElectronConfig] = field(default_factory=list)
    campaign: CampaignConfig = field(default_factory=CampaignConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> MainConfig:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must parse to a dict")

    cfg = MainConfig()

    import dataclasses

    def asdict_dc(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: asdict_dc(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [asdict_dc(x) for x in obj]
        return obj

    defaults_dict = asdict_dc(cfg)

    def validate_keys(defaults: Dict[str, Any], user: Dict[str, Any], prefix: str = "") -> None:
        for k in user.keys():
            if k not in defaults:
                raise ValueError(f"Unknown config key: {prefix}{k}")
            if isinstance(user[k], dict) and isinstance(defaults[k], dict):
                validate_keys(defaults[k], user[k], prefix=f"{prefix}{k}.")

    validate_keys(defaults_dict, data)
    cfg_dict = _deep_update(defaults_dict, data)

    sim = SimulationConfig(**cfg_dict["simulation"])
    physics = PhysicsConfig(**cfg_dict["physics"])
    features = FeatureConfig(**cfg_dict["features"])
    trap_grid = TrapGridConfig(**cfg_dict["trap"]["grid"])
    trap = TrapConfig(**{**{k: v for k, v in cfg_dict["trap"].items() if k != "grid"}, "grid": trap_grid})
    cavity = CavityConfig(**cfg_dict["cavity"])
    mode_map = ModeMapConfig(**cfg_dict["mode_map"])
    resonance = ResonanceConfig(**cfg_dict["resonance"])
    electron = ElectronConfig(**cfg_dict["electron"])
    tracks = [ElectronConfig(**item) for item in cfg_dict.get("tracks", [])]
    campaign = CampaignConfig(**cfg_dict.get("campaign", {}))
    dynamics = DynamicsConfig(**cfg_dict["dynamics"])
    signal = SignalConfig(**cfg_dict["signal"])
    ro_lpf = ReadoutLpfConfig(**cfg_dict["readout"]["lpf"])
    ro_noise = ReadoutNoiseConfig(**cfg_dict["readout"]["noise"])
    ro_digitizer = DigitizerConfig(**cfg_dict["readout"]["digitizer"])
    readout = ReadoutConfig(**{
        **{k: v for k, v in cfg_dict["readout"].items() if k not in {"lpf", "noise", "digitizer"}},
        "lpf": ro_lpf,
        "noise": ro_noise,
        "digitizer": ro_digitizer,
    })
    output = OutputConfig(**cfg_dict["output"])

    cfg_obj = MainConfig(
        simulation=sim,
        physics=physics,
        features=features,
        trap=trap,
        cavity=cavity,
        mode_map=mode_map,
        resonance=resonance,
        electron=electron,
        tracks=tracks,
        campaign=campaign,
        dynamics=dynamics,
        signal=signal,
        readout=readout,
        output=output,
    )
    _validate_semantic_compatibility(cfg_obj, data)
    return cfg_obj


def _key_present(data: Dict[str, Any], *path: str) -> bool:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _validate_semantic_compatibility(cfg: MainConfig, raw_data: Dict[str, Any]) -> None:
    dyn = cfg.dynamics
    feat = cfg.features
    cav = cfg.cavity
    mm = cfg.mode_map
    mm_type = str(mm.type).lower()

    if dyn.time_step_strategy in {"phase_uniform", "phase_bounded"}:
        if dyn.samples_per_cyclotron_turn is None or int(dyn.samples_per_cyclotron_turn) <= 0:
            raise ValueError(
                "dynamics.samples_per_cyclotron_turn must be a positive integer when "
                "dynamics.time_step_strategy is phase_uniform or phase_bounded."
            )

    if dyn.axial_strategy == "direct":
        if dyn.energy_loss_model in {"per_bounce", "linear_fit"}:
            raise ValueError(
                "dynamics.energy_loss_model='per_bounce' or 'linear_fit' requires "
                "dynamics.axial_strategy='template_tiling'. Use energy_loss_model='analytic' "
                "for direct continuous radiative integration."
            )
        if _key_present(raw_data, "dynamics", "template_build"):
            raise ValueError(
                "dynamics.template_build is only meaningful when dynamics.axial_strategy='template_tiling'."
            )

    if dyn.axial_strategy == "template_tiling" and dyn.energy_loss_model == "analytic":
        raise ValueError(
            "dynamics.energy_loss_model='analytic' is a direct continuous solver. "
            "Set dynamics.axial_strategy='direct', or use 'per_bounce'/'linear_fit' with template_tiling."
        )

    if dyn.energy_loss_model == "linear_fit" and int(dyn.energy_loss_fit_bounces) < 2:
        raise ValueError("dynamics.energy_loss_fit_bounces must be >= 2 for linear_fit.")

    if int(dyn.per_bounce_block_bounces) < 1:
        raise ValueError("dynamics.per_bounce_block_bounces must be >= 1.")
    if int(dyn.multi_bounce_auto_max_bounces) < 1:
        raise ValueError("dynamics.multi_bounce_auto_max_bounces must be >= 1.")
    if int(dyn.mirror_quadrature_min_theta_nodes) < 3:
        raise ValueError("dynamics.mirror_quadrature_min_theta_nodes must be >= 3.")
    if int(dyn.mirror_quadrature_max_theta_nodes) < int(dyn.mirror_quadrature_min_theta_nodes):
        raise ValueError("dynamics.mirror_quadrature_max_theta_nodes must be >= mirror_quadrature_min_theta_nodes.")
    if float(dyn.mirror_template_max_period_rel_error) <= 0.0:
        raise ValueError("dynamics.mirror_template_max_period_rel_error must be positive.")
    if float(dyn.mirror_template_max_phase_slip_rad) <= 0.0:
        raise ValueError("dynamics.mirror_template_max_phase_slip_rad must be positive.")
    if float(dyn.mirror_template_moving_mirror_tol) <= 0.0:
        raise ValueError("dynamics.mirror_template_moving_mirror_tol must be positive.")
    if float(dyn.back_reaction_block_max_rel_state_change) <= 0.0:
        raise ValueError("dynamics.back_reaction_block_max_rel_state_change must be positive.")
    if int(dyn.back_reaction_max_updates_per_bounce) < 1:
        raise ValueError("dynamics.back_reaction_max_updates_per_bounce must be >= 1.")
    if dyn.compact_output_dt_s is not None and float(dyn.compact_output_dt_s) <= 0.0:
        raise ValueError("dynamics.compact_output_dt_s must be positive or null.")

    if dyn.cyclotron_frequency_reference == "instantaneous" and not feat.include_true_orbit:
        raise ValueError(
            "dynamics.cyclotron_frequency_reference='instantaneous' requires features.include_true_orbit=true."
        )

    if int(dyn.true_orbit_phase_iterations) < 0:
        raise ValueError("dynamics.true_orbit_phase_iterations must be >= 0.")

    if not cav.excitation_enabled:
        if _key_present(raw_data, "cavity", "ringup_enabled") and cav.ringup_enabled:
            raise ValueError("cavity.ringup_enabled=true requires cavity.excitation_enabled=true.")
        if _key_present(raw_data, "cavity", "back_reaction_enabled") and cav.back_reaction_enabled:
            raise ValueError("cavity.back_reaction_enabled=true requires cavity.excitation_enabled=true.")
    if str(cav.response_model) not in {"time_evolution", "baseband_envelope"}:
        raise ValueError("cavity.response_model must be 'time_evolution' or 'baseband_envelope'.")

    if cav.response_model in {"time_evolution", "baseband_envelope"} and not cav.excitation_enabled:
        raise ValueError("coherent cavity response models require cavity.excitation_enabled=true.")
    if cav.response_model in {"time_evolution", "baseband_envelope"} and bool(cav.back_reaction_enabled):
        if not (mm_type in {"auto", "vector_e_field", "vector"} and mm.vector_e_field_map):
            raise ValueError(
                "cavity.back_reaction_enabled=true with a coherent cavity response requires "
                "a configured vector E-field map. Scalar/analytic mode maps are allowed for "
                "one-way signal generation but not for stimulated back-reaction."
            )
        if cfg.dynamics.energy_loss_model == "none":
            raise ValueError(
                "cavity.back_reaction_enabled=true requires dynamics.energy_loss_model to be "
                "'per_bounce' or 'analytic' so the electron state can be updated by the "
                "complex cavity work."
            )
    if not (0.0 <= float(cav.output_coupling_fraction) <= 1.0):
        raise ValueError("cavity.output_coupling_fraction must lie in [0, 1].")
    if cav.extend_ringdown_s is not None and float(cav.extend_ringdown_s) < 0.0:
        raise ValueError("cavity.extend_ringdown_s must be non-negative or null.")

    if cfg.signal.cavity_filter_grid == "if" and not bool(cfg.signal.require_analytic_baseband_drive):
        raise ValueError("signal.cavity_filter_grid='if' requires signal.require_analytic_baseband_drive=true.")
    if float(cfg.signal.if_bandwidth_tolerance) < 0.0:
        raise ValueError("signal.if_bandwidth_tolerance must be non-negative.")

    readout = cfg.readout
    if int(readout.fast_decimation_factor) < 1:
        raise ValueError("readout.fast_decimation_factor must be >= 1.")
    if int(readout.chunk_size) < 1:
        raise ValueError("readout.chunk_size must be >= 1.")
    if not (0.0 < float(readout.lpf.cutoff_ratio_of_final_nyquist) <= 1.0):
        raise ValueError("readout.lpf.cutoff_ratio_of_final_nyquist must lie in (0, 1].")
    if int(readout.digitizer.bit_depth) < 1:
        raise ValueError("readout.digitizer.bit_depth must be positive.")
    if readout.digitizer.v_range <= 0.0:
        raise ValueError("readout.digitizer.v_range must be positive.")
    if readout.model in {"locust_exact_baseband", "locust_like_baseband"} and not bool(readout.require_analytic_baseband_drive):
        raise ValueError("Locust-style readout requires analytic baseband drive generation.")
    if int(readout.lpf.n_windows) < 1:
        raise ValueError("readout.lpf.n_windows must be >= 1.")

    if int(mm.vector_gyro_quadrature_points) < 1:
        raise ValueError("mode_map.vector_gyro_quadrature_points must be >= 1.")
    if float(mm.vector_energy_normalization_J) <= 0.0:
        raise ValueError("mode_map.vector_energy_normalization_J must be positive.")
    if mm.vector_bounds_policy not in {"error", "warn", "zero"}:
        raise ValueError("mode_map.vector_bounds_policy must be 'error', 'warn', or 'zero'.")
    if mm_type in {"vector_e_field", "vector"} and not mm.vector_e_field_map and not mm.fallback_to_analytic:
        raise ValueError(
            "mode_map.type='vector_e_field' requires mode_map.vector_e_field_map unless "
            "mode_map.fallback_to_analytic=true."
        )
    if cav.back_reaction_enabled and cav.back_reaction_requires_vector_map:
        if not (mm_type in {"auto", "vector_e_field", "vector"} and mm.vector_e_field_map):
            raise ValueError(
                "cavity.back_reaction_enabled with cavity.back_reaction_requires_vector_map=true "
                "requires a configured vector E-field map."
            )

    for i, trk in enumerate(cfg.tracks):
        if trk.vpar_sign not in {-1, 1}:
            raise ValueError(f"tracks[{i}].vpar_sign must be +1 or -1")
    if cfg.campaign.enabled and int(cfg.campaign.slurm_cpus_per_task) < 1:
        raise ValueError("campaign.slurm_cpus_per_task must be >= 1")

