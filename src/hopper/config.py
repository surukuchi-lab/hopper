"""
Configuration models + YAML loader.

Naming requested in config:
- starting_time_s == "starting time"
- track_length_s  == "track length"
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
class FeatureConfig:
    include_true_orbit: bool = True
    include_gradB: bool = True
    include_curvature_drift: bool = True
    include_resonance: bool = True
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


@dataclass
class ModeMapConfig:
    type: str = "analytic_te011"


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
    template_build: Literal["auto", "integrate", "mirror"] = "mirror"

    # Template detection controls (integrate build)
    template_return_z_tol_m: float = 1e-7
    template_max_duration_s: float = 5e-3
    template_min_reflections: int = 2

    # Mirror-extension validity checks (mirror build). The mirror builder supports arbitrary starting z by phase-shifting a symmetric midplane-to-turning-point template.
    mirror_z0_tol_m: float = 1e-6
    mirror_symmetry_check: bool = True
    mirror_symmetry_rel_tol: float = 1e-3
    mirror_symmetry_ncheck: int = 25

    # Energy loss / chirp model.
    # - "none": constant energy and constant µ
    # - "per_bounce": integrate one full bounce at a time on the cached field line using the exact local radiation step; the mirror-template machinery is used only to seed the bounce-period guess, so the radiatively moving mirror remains self-consistent
    # - "linear_fit": fast approximation; fit linear dE/dt and dµ/dt from several evolving-(γ, µ) bounces, then apply those rates on a tiled template
    # - "analytic": continuously apply the exact local radiation step throughout the axial integration, evolving both γ and µ and therefore the reference pitch angle
    energy_loss_model: Literal["none", "per_bounce", "linear_fit", "analytic"] = "per_bounce"
    energy_loss_scale: float = 1.0
    energy_floor_eV: float = 1.0

    # Used only when energy_loss_model="linear_fit"
    energy_loss_fit_bounces: int = 6

    # RF/output time-grid strategy.  The compact dynamics track is not expanded internally;
    # phase-aware sampling is applied at the signal/ROOT-output boundary.
    # - "axial_adaptive": use the configured uniform-time signal RF sampling rate
    # - "phase_bounded": use a uniform-time RF grid fast enough to keep the maximum cyclotron phase advance <= 2π / samples_per_cyclotron_turn
    # - "phase_uniform": sample at exactly uniform cyclotron phase increments Δφ = 2π / samples_per_cyclotron_turn; this time grid is generally non-uniform
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



@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    basename: str = "sim"
    write_npz: bool = True
    write_root: bool = True
    track_sampling: TrackSampling = "rf_sampled"
    root_chunk_size: int = 200_000


@dataclass
class MainConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    trap: TrapConfig = field(default_factory=TrapConfig)
    cavity: CavityConfig = field(default_factory=CavityConfig)
    mode_map: ModeMapConfig = field(default_factory=ModeMapConfig)
    resonance: ResonanceConfig = field(default_factory=ResonanceConfig)
    electron: ElectronConfig = field(default_factory=ElectronConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
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
    features = FeatureConfig(**cfg_dict["features"])
    trap_grid = TrapGridConfig(**cfg_dict["trap"]["grid"])
    trap = TrapConfig(**{**{k: v for k, v in cfg_dict["trap"].items() if k != "grid"}, "grid": trap_grid})
    cavity = CavityConfig(**cfg_dict["cavity"])
    mode_map = ModeMapConfig(**cfg_dict["mode_map"])
    resonance = ResonanceConfig(**cfg_dict["resonance"])
    electron = ElectronConfig(**cfg_dict["electron"])
    dynamics = DynamicsConfig(**cfg_dict["dynamics"])
    signal = SignalConfig(**cfg_dict["signal"])
    output = OutputConfig(**cfg_dict["output"])

    return MainConfig(
        simulation=sim,
        features=features,
        trap=trap,
        cavity=cavity,
        mode_map=mode_map,
        resonance=resonance,
        electron=electron,
        dynamics=dynamics,
        signal=signal,
        output=output,
    )
