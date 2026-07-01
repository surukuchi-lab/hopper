"""
Hopper integration and regression tests.

Developer: ehtkarim
Date: April 29, 2026

Covers configuration validation, field/cavity utilities, dynamics paths, signal synthesis, and output behavior.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hopper.config import load_config
from hopper.cavity.cavity import Cavity
from hopper.cavity.interaction import CavityInteraction
from hopper.dynamics.axial_profile import AxialFieldProfile
from hopper.dynamics.kinematics import (
    beta_parallel2_from_B_gamma_mu,
    gamma_beta_v_from_kinetic,
    gamma_mu_after_radiation_step_fixed_upar,
    larmor_radius_m_array,
    mu_from_pitch,
    parallel_u2_from_B_gamma_mu,
)
from hopper.dynamics.track import _local_perp_basis_from_field, sample_dynamic_track
from hopper.dynamics.drifts import curvature_drift_vphi, gradB_drift_vphi
from hopper.nodes.output_node import _slice_track
from hopper.nodes.pipeline import run_pipeline


REAL_FIELD_MAP = Path(__file__).resolve().parents[1] / "resources" / "field_map_rz_components.npz"


def _write_cfg(tmp_path: Path, body: str, *, name: str = "cfg.yaml") -> Path:
    cfg_path = tmp_path / name
    cfg_path.write_text(body.format(out_dir=str(tmp_path / "out")))
    return cfg_path


def test_smoke_runs_with_placeholder(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-5
electron:
  pitch_angle_deg: 89.0
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{out_dir}"
  basename: "t"
  write_npz: true
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text))
    ctx = run_pipeline(cfg)

    assert "signal_result" in ctx
    assert Path(ctx["npz_path"]).exists()


def _assert_instantaneous_start_geometry(ctx, cfg, *, use_true_orbit: bool):
    tr = ctx["track_dyn"]
    field = ctx["field"]
    psi0 = float(cfg.electron.cyclotron_phase0_rad)

    P = np.asarray([float(tr.x[0]), float(tr.y[0]), float(tr.z[0])], dtype=float)
    G = np.asarray([float(tr.x_gc[0]), float(tr.y_gc[0]), float(tr.z_gc[0])], dtype=float)
    d = P - G

    # The gyro-orbit geometry is always a guiding-center construction. In
    # true-orbit instantaneous-start mode, only the initial magnetic moment is
    # inferred from the actual particle point; the Larmor radius and perpendicular
    # basis are evaluated at the inferred guiding center.
    r_ref = float(np.hypot(G[0], G[1]))
    z_ref = float(G[2])
    phi_ref = float(np.arctan2(G[1], G[0]))

    if use_true_orbit:
        r_pitch = float(np.hypot(P[0], P[1]))
        z_pitch = float(P[2])
        B_pitch = float(field.B(r_pitch, z_pitch))
        mu_expected = float(np.asarray(mu_from_pitch(cfg.electron.energy_eV, cfg.electron.pitch_angle_deg, B_pitch)).reshape(()))
        assert np.isclose(float(tr.mu_J_per_T[0]), mu_expected, rtol=1.0e-12, atol=0.0)

    B_ref = float(field.B(r_ref, z_ref))
    Br_ref, Bphi_ref, Bz_ref = field.components(r_ref, z_ref)
    u1, u2, b = _local_perp_basis_from_field(phi_ref, Br_ref, Bphi_ref, Bz_ref)
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    b = np.asarray(b, dtype=float)

    rho0 = float(
        larmor_radius_m_array(
            B_ref,
            np.asarray([float(tr.energy_eV[0])], dtype=float),
            np.asarray([float(tr.mu_J_per_T[0])], dtype=float),
            q_C=-1.602176634e-19,
        )[0]
    )

    assert np.isclose(np.linalg.norm(d), rho0, rtol=0.0, atol=5.0e-11)
    assert abs(float(np.dot(d, b))) < 5.0e-13
    assert np.isclose(float(np.dot(d, u1)) / rho0, math.cos(psi0), rtol=0.0, atol=2.0e-11)
    assert np.isclose(float(np.dot(d, u2)) / rho0, math.sin(psi0), rtol=0.0, atol=2.0e-11)




def _assert_guiding_center_start_geometry(ctx, cfg) -> None:
    tr = ctx["track_dyn"]
    field = ctx["field"]
    psi0 = float(cfg.electron.cyclotron_phase0_rad)

    G_expected = np.asarray(
        [
            float(cfg.electron.r0_m * math.cos(cfg.electron.phi0_rad)),
            float(cfg.electron.r0_m * math.sin(cfg.electron.phi0_rad)),
            float(cfg.electron.z0_m),
        ],
        dtype=float,
    )
    G = np.asarray([float(tr.x_gc[0]), float(tr.y_gc[0]), float(tr.z_gc[0])], dtype=float)
    P = np.asarray([float(tr.x[0]), float(tr.y[0]), float(tr.z[0])], dtype=float)
    d = P - G

    assert np.allclose(G, G_expected, rtol=0.0, atol=1.0e-12)

    r_ref = float(np.hypot(G[0], G[1]))
    z_ref = float(G[2])
    phi_ref = float(np.arctan2(G[1], G[0]))
    B_ref = float(field.B(r_ref, z_ref))
    Br_ref, Bphi_ref, Bz_ref = field.components(r_ref, z_ref)
    u1, u2, b = _local_perp_basis_from_field(phi_ref, Br_ref, Bphi_ref, Bz_ref)
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    b = np.asarray(b, dtype=float)

    rho0 = float(
        larmor_radius_m_array(
            B_ref,
            np.asarray([float(tr.energy_eV[0])], dtype=float),
            np.asarray([float(tr.mu_J_per_T[0])], dtype=float),
            q_C=-1.602176634e-19,
        )[0]
    )

    assert np.isclose(np.linalg.norm(d), rho0, rtol=0.0, atol=5.0e-11)
    assert abs(float(np.dot(d, b))) < 5.0e-13
    assert np.isclose(float(np.dot(d, u1)) / rho0, math.cos(psi0), rtol=0.0, atol=2.0e-11)
    assert np.isclose(float(np.dot(d, u2)) / rho0, math.sin(psi0), rtol=0.0, atol=2.0e-11)


class _UniformFieldMap:
    def __init__(self):
        self.r = np.linspace(0.0, 1.0, 8)
        self.z = np.linspace(-1.0, 1.0, 16)
        self.clamp_to_grid = True

    def B(self, r, z):
        r_arr = np.asarray(r, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        return np.full(np.broadcast(r_arr, z_arr).shape, 2.0, dtype=float)

    def components(self, r, z):
        r_arr = np.asarray(r, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        shape = np.broadcast(r_arr, z_arr).shape
        return (
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            np.full(shape, 2.0, dtype=float),
        )

    def gradB(self, r, z):
        r_arr = np.asarray(r, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        shape = np.broadcast(r_arr, z_arr).shape
        return np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)


def test_instantaneous_start_position_cylindrical_is_self_consistent_without_true_orbit(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-6
electron:
  position_reference: instantaneous
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.7
  z0_m: 0.0
  cyclotron_phase0_rad: 0.3
features:
  include_true_orbit: false
  include_gradB: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: true
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_inst_cyl.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]

    x0 = cfg.electron.r0_m * math.cos(cfg.electron.phi0_rad)
    y0 = cfg.electron.r0_m * math.sin(cfg.electron.phi0_rad)
    assert abs(float(tr.x[0]) - x0) < 1e-9
    assert abs(float(tr.y[0]) - y0) < 1e-9
    assert abs(float(tr.z[0]) - cfg.electron.z0_m) < 1e-12
    _assert_instantaneous_start_geometry(ctx, cfg, use_true_orbit=False)


def test_instantaneous_start_position_cartesian_is_self_consistent_with_true_orbit(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-6
electron:
  position_reference: instantaneous
  position_coordinates: cartesian
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  x0_m: 0.1205
  y0_m: -0.08025
  z0_m: 0.0
  cyclotron_phase0_rad: 1.1
features:
  include_true_orbit: true
  include_gradB: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: true
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_inst_xy.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]

    assert abs(float(tr.x[0]) - cfg.electron.x0_m) < 1e-9
    assert abs(float(tr.y[0]) - cfg.electron.y0_m) < 1e-9
    assert abs(float(tr.z[0]) - cfg.electron.z0_m) < 1e-12
    _assert_instantaneous_start_geometry(ctx, cfg, use_true_orbit=True)


def test_guiding_center_follows_cached_field_line_on_real_field_map(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 5.0e-6
electron:
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.3
  z0_m: 0.0
features:
  include_true_orbit: false
  include_gradB: true
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_real_map.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]
    field = ctx["field"]

    profile = AxialFieldProfile.from_field(field, r0_m=cfg.electron.r0_m, z0_m=cfg.electron.z0_m)
    rgc = np.hypot(tr.x_gc, tr.y_gc)
    assert np.allclose(rgc, profile.r_at_z(tr.z_gc), rtol=0.0, atol=1e-12)

    dz = np.diff(tr.z_gc)
    dr = np.diff(rgc)
    mask = np.abs(dz) > 1.0e-9
    slope = dr[mask] / dz[mask]
    z_mid = 0.5 * (tr.z_gc[:-1] + tr.z_gc[1:])[mask]
    assert np.allclose(slope, profile.dr_dz(z_mid), rtol=0.0, atol=2.0e-6)


def test_phase_uniform_keeps_internal_grid_compact_and_samples_per_cyclotron_turn(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-8
electron:
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  z0_m: 0.0
features:
  include_true_orbit: true
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  time_step_strategy: phase_uniform
  samples_per_cyclotron_turn: 15
signal:
  fs_if_hz: 4.0e6
  if_decim: 2
output:
  out_dir: "{out_dir}"
  basename: "t"
  write_npz: false
  write_root: false
  track_sampling: rf_sampled
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_phase_uniform.yaml"))
    ctx = run_pipeline(cfg)

    tr_dyn = ctx["track_dyn"]
    sig = ctx["signal_result"]

    assert len(tr_dyn.t) < 10000
    assert sig.rf_grid_kind == "uniform_phase"
    assert sig.track_rf is not None

    dphi = np.diff(np.unwrap(sig.track_rf.phase_rf))
    target = 2.0 * np.pi / cfg.dynamics.samples_per_cyclotron_turn
    assert np.allclose(dphi, target, rtol=2e-3, atol=2e-3)
    assert len(sig.track_if.t) == math.ceil(len(sig.t) / cfg.signal.if_decim)


def test_exact_local_radiation_step_preserves_parallel_u_and_reduces_mu():
    B_T = 1.0
    E0_eV = 18_563.251
    pitch_deg = 80.0

    gamma0 = float(np.asarray(gamma_beta_v_from_kinetic(E0_eV)[0]).reshape(()))
    mu0 = float(np.asarray(mu_from_pitch(E0_eV, pitch_deg, B_T)).reshape(()))
    upar2_0 = float(np.asarray(parallel_u2_from_B_gamma_mu(B_T, gamma0, mu0)).reshape(()))

    gamma1, mu1 = gamma_mu_after_radiation_step_fixed_upar(
        gamma0,
        mu0,
        B_T,
        1.0e-6,
        energy_loss_scale=1.0e6,
    )
    gamma1 = float(np.asarray(gamma1).reshape(()))
    mu1 = float(np.asarray(mu1).reshape(()))
    upar2_1 = float(np.asarray(parallel_u2_from_B_gamma_mu(B_T, gamma1, mu1)).reshape(()))

    assert gamma1 < gamma0
    assert mu1 < mu0
    assert np.isclose(upar2_1, upar2_0, rtol=1e-10, atol=1e-12)


def test_energy_loss_models_are_monotone_and_evolve_mu(tmp_path: Path):
    base = """
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-6
electron:
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  z0_m: 0.0
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: {model}
  energy_loss_scale: 1.0e3
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{out_dir}"
  basename: "t"
  write_npz: false
  write_root: false
"""

    cfg_pb = load_config(_write_cfg(tmp_path, base.format(model="per_bounce", out_dir=str(tmp_path / "out")), name="cfg_pb.yaml"))
    ctx_pb = run_pipeline(cfg_pb)
    tr_pb = ctx_pb["track_dyn"]
    E_pb = np.asarray(tr_pb.energy_eV, dtype=float)
    mu_pb = np.asarray(tr_pb.mu_J_per_T, dtype=float)

    cfg_an_text = base.format(model="analytic", out_dir=str(tmp_path / "out")).replace(
        "  axial_strategy: template_tiling\n  template_build: mirror\n",
        "  axial_strategy: direct\n",
    )
    cfg_an = load_config(_write_cfg(tmp_path, cfg_an_text, name="cfg_an.yaml"))
    ctx_an = run_pipeline(cfg_an)
    tr_an = ctx_an["track_dyn"]
    E_an = np.asarray(tr_an.energy_eV, dtype=float)
    mu_an = np.asarray(tr_an.mu_J_per_T, dtype=float)

    assert np.all(np.diff(E_pb) <= 1e-9)
    assert np.all(np.diff(E_an) <= 1e-9)
    assert np.all(np.diff(mu_pb) <= 1e-18)
    assert np.all(np.diff(mu_an) <= 1e-18)
    assert mu_pb[0] > mu_pb[-1]
    assert mu_an[0] > mu_an[-1]
    assert np.unique(np.round(mu_an, 18)).size > 2


def test_true_orbit_does_not_feed_orbit_point_field_back_into_gyrophase(tmp_path: Path):
    base = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.4
  z0_m: 0.1
  cyclotron_phase0_rad: 0.9
features:
  include_true_orbit: {{true_orbit}}
  include_gradB: true
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg_true = load_config(_write_cfg(tmp_path, base.format(true_orbit="true", out_dir=str(tmp_path / "out_true")), name="cfg_true_orbit.yaml"))
    cfg_false = load_config(_write_cfg(tmp_path, base.format(true_orbit="false", out_dir=str(tmp_path / "out_false")), name="cfg_false_orbit.yaml"))

    tr_true = run_pipeline(cfg_true)["track_dyn"]
    tr_false = run_pipeline(cfg_false)["track_dyn"]

    # With the default guiding-center frequency reference, true-orbit reconstruction
    # changes the instantaneous particle branch but not the phase/frequency reference
    # products.  This keeps phase_rf and f_c_hz in the same frame.
    assert np.allclose(tr_true.phase_rf, tr_false.phase_rf, rtol=0.0, atol=5.0e-9)
    assert np.allclose(tr_true.f_c_hz, tr_false.f_c_hz, rtol=0.0, atol=1.0e-6)
    assert np.allclose(tr_true.B_T, tr_false.B_T, rtol=0.0, atol=1.0e-15)


def test_root_branch_schema_keeps_only_time_s():
    from hopper.dynamics.track import DynamicTrack
    from hopper.nodes.output_node import _track_root_arrays

    t = np.asarray([0.0, 1.0e-9], dtype=float)
    zeros = np.zeros_like(t)
    track = DynamicTrack(
        t=t,
        x=zeros,
        y=zeros,
        z=zeros,
        vx=zeros,
        vy=zeros,
        vz=zeros,
        x_gc=zeros,
        y_gc=zeros,
        z_gc=zeros,
        vx_gc=zeros,
        vy_gc=zeros,
        vz_gc=zeros,
        r_gc_m=zeros,
        phi_gc_rad=zeros,
        parallel_sign=np.ones_like(t),
        b_cross_kappa_phi_per_m=zeros,
        f_c_hz=zeros,
        amp=np.ones_like(t),
        phase_rf=zeros,
        B_T=np.ones_like(t),
        energy_eV=np.full_like(t, 18_563.251),
        mu_J_per_T=np.full_like(t, 1.0e-17),
    )

    arrays = _track_root_arrays(track)
    assert np.array_equal(arrays["time_s"], t)
    assert "time_steps" not in arrays
    assert "time_step" not in arrays


def test_run_from_config_keeps_output_dir_relative_to_cwd(tmp_path: Path, monkeypatch):
    resource_src = Path(__file__).resolve().parents[1] / "resources" / "field_map_rz_components.npz"
    cfg_dir = tmp_path / "configs"
    res_dir = tmp_path / "resources"
    cfg_dir.mkdir()
    res_dir.mkdir()
    (res_dir / "field_map_rz_components.npz").write_bytes(resource_src.read_bytes())

    cfg_path = cfg_dir / "cfg.yaml"
    cfg_path.write_text(
        """
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-6
electron:
  pitch_angle_deg: 89.0
trap:
  field_map_npz: ../resources/field_map_rz_components.npz
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: run_outputs
  basename: t
  write_npz: true
  write_root: false
"""
    )

    monkeypatch.chdir(tmp_path)
    from hopper.nodes.pipeline import run_from_config

    ctx = run_from_config(cfg_path)
    npz_path = Path(ctx["npz_path"])
    assert npz_path.resolve() == (tmp_path / "run_outputs" / "t_iq.npz").resolve()
    assert npz_path.exists()


def test_mirror_template_handles_off_midplane_start_without_flat_tail(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 8.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.3
  z0_m: 0.2
  vpar_sign: -1
features:
  include_true_orbit: false
  include_gradB: true
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_mirror_off_midplane.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]

    assert np.isclose(float(tr.t[0]), 0.0)
    assert np.isclose(float(tr.t[-1]), cfg.simulation.track_length_s)
    assert np.isclose(float(tr.z_gc[0]), cfg.electron.z0_m, atol=1.0e-12)
    assert float(np.ptp(tr.z_gc[-min(100, len(tr.z_gc)) :])) > 1.0e-3
    sign_changes = np.flatnonzero(np.signbit(tr.vz_gc[1:]) != np.signbit(tr.vz_gc[:-1]))
    assert sign_changes.size >= 2


def test_instantaneous_off_midplane_phase_uses_mirror_template(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 3.0e-6
electron:
  position_reference: instantaneous
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.7
  z0_m: 0.2
  vpar_sign: 1
  cyclotron_phase0_rad: 1.2
features:
  include_true_orbit: true
  include_gradB: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_inst_off_midplane.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]

    x0 = cfg.electron.r0_m * math.cos(cfg.electron.phi0_rad)
    y0 = cfg.electron.r0_m * math.sin(cfg.electron.phi0_rad)
    assert abs(float(tr.x[0]) - x0) < 1e-9
    assert abs(float(tr.y[0]) - y0) < 1e-9
    assert abs(float(tr.z[0]) - cfg.electron.z0_m) < 1e-12
    assert np.isclose(float(tr.t[-1]), cfg.simulation.track_length_s)
    assert float(np.ptp(tr.z_gc)) > 0.1


def test_curvature_drift_branch_is_available_and_finite_on_real_field_map(tmp_path: Path):
    base = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 3.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.3
  z0_m: 0.0
features:
  include_true_orbit: false
  include_gradB: true
  include_curvature_drift: {{curv}}
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg_on = load_config(_write_cfg(tmp_path, base.format(curv="true", out_dir=str(tmp_path / "out_on")), name="cfg_curv_on.yaml"))
    cfg_off = load_config(_write_cfg(tmp_path, base.format(curv="false", out_dir=str(tmp_path / "out_off")), name="cfg_curv_off.yaml"))

    tr_on = run_pipeline(cfg_on)["track_dyn"]
    tr_off = run_pipeline(cfg_off)["track_dyn"]

    assert np.all(np.isfinite(tr_on.b_cross_kappa_phi_per_m))
    from hopper.nodes.output_node import _track_root_arrays

    assert "fieldline_b_cross_kappa_phi_per_m" in _track_root_arrays(tr_on)
    # The correction is intentionally small for this field map, but it must be wired into
    # the guiding-center azimuthal evolution rather than merely exposed as an unused branch.
    assert not np.allclose(tr_on.phi_gc_rad, tr_off.phi_gc_rad, rtol=0.0, atol=1.0e-15)


def test_per_bounce_radiative_time_axis_responds_inside_each_bounce(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-6
electron:
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  z0_m: 0.0
features:
  include_true_orbit: false
  include_gradB: false
  include_curvature_drift: false
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
  energy_loss_scale: 1.0e3
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{out_dir}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_pb_time_axis.yaml"))
    tr = run_pipeline(cfg)["track_dyn"]

    assert np.isclose(float(tr.t[-1]), cfg.simulation.track_length_s)
    assert np.all(np.diff(tr.t) > 0.0)
    assert np.all(np.diff(tr.energy_eV) <= 1e-9)
    assert np.all(np.diff(tr.mu_J_per_T) <= 1e-18)



def test_per_bounce_guiding_center_stays_inside_the_moving_mirror_on_real_field_map(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-5
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.0
  z0_m: 0.0
  vpar_sign: 1
  cyclotron_phase0_rad: 0.0
features:
  include_true_orbit: false
  include_gradB: false
  include_curvature_drift: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_pb_mirror_bound.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]
    field = ctx["field"]

    rgc = np.hypot(tr.x_gc, tr.y_gc)
    Bgc = np.asarray(field.B(rgc, tr.z_gc), dtype=float)
    gamma = np.asarray(gamma_beta_v_from_kinetic(tr.energy_eV)[0], dtype=float)
    beta_par2 = beta_parallel2_from_B_gamma_mu(Bgc, gamma, tr.mu_J_per_T)

    # The compact guiding-center track must remain inside the moving mirror implied by the
    # evolving (gamma, mu) state; tiny negative values at the exact turning node are allowed
    # only at numerical roundoff level.
    assert float(np.min(beta_par2)) > -5.0e-12



def test_per_bounce_radiative_track_matches_analytic_short_reference_on_real_field_map(tmp_path: Path):
    base = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 5.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.0
  z0_m: 0.0
  vpar_sign: 1
  cyclotron_phase0_rad: 0.0
features:
  include_true_orbit: false
  include_gradB: false
  include_curvature_drift: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_scale: 1.0
  per_bounce_block_bounces: 8
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""

    cfg_pb_text = base.replace("dynamics:\n", "dynamics:\n  energy_loss_model: per_bounce\n", 1)
    cfg_an_text = base.replace("dynamics:\n", "dynamics:\n  energy_loss_model: analytic\n", 1).replace(
        "  axial_strategy: template_tiling\n  template_build: mirror\n",
        "  axial_strategy: direct\n",
    )
    cfg_pb = load_config(_write_cfg(tmp_path, cfg_pb_text, name="cfg_pb_short_ref.yaml"))
    cfg_an = load_config(_write_cfg(tmp_path, cfg_an_text, name="cfg_an_short_ref.yaml"))

    tr_pb = run_pipeline(cfg_pb)["track_dyn"]
    tr_an = run_pipeline(cfg_an)["track_dyn"]

    t_ref = np.linspace(0.0, cfg_pb.simulation.track_length_s, 500)
    z_pb = np.interp(t_ref, tr_pb.t, tr_pb.z_gc)
    z_an = np.interp(t_ref, tr_an.t, tr_an.z_gc)
    B_pb = np.interp(t_ref, tr_pb.t, tr_pb.B_T)
    B_an = np.interp(t_ref, tr_an.t, tr_an.B_T)

    # per_bounce is the production-speed mode, but it should still track the fully continuous
    # analytic reference closely over a short window on the real field map.
    assert float(np.max(np.abs(z_pb - z_an))) < 3.0e-4
    assert float(np.max(np.abs(B_pb - B_an))) < 1.0e-8


def test_per_bounce_radiative_track_matches_analytic_longer_reference_on_real_field_map(tmp_path: Path):
    base = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-5
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.0
  z0_m: 0.0
  vpar_sign: 1
  cyclotron_phase0_rad: 0.0
features:
  include_true_orbit: false
  include_gradB: true
  include_curvature_drift: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_scale: 1.0
  per_bounce_block_bounces: 8
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""

    cfg_pb_text = base.replace("dynamics:\n", "dynamics:\n  energy_loss_model: per_bounce\n", 1)
    cfg_an_text = base.replace("dynamics:\n", "dynamics:\n  energy_loss_model: analytic\n", 1).replace(
        "  axial_strategy: template_tiling\n  template_build: mirror\n",
        "  axial_strategy: direct\n",
    )
    cfg_pb = load_config(_write_cfg(tmp_path, cfg_pb_text, name="cfg_pb_long_ref.yaml"))
    cfg_an = load_config(_write_cfg(tmp_path, cfg_an_text, name="cfg_an_long_ref.yaml"))

    tr_pb = run_pipeline(cfg_pb)["track_dyn"]
    tr_an = run_pipeline(cfg_an)["track_dyn"]

    t_ref = np.linspace(0.0, cfg_pb.simulation.track_length_s, 1200)
    x_pb = np.interp(t_ref, tr_pb.t, tr_pb.x_gc)
    x_an = np.interp(t_ref, tr_an.t, tr_an.x_gc)
    y_pb = np.interp(t_ref, tr_pb.t, tr_pb.y_gc)
    y_an = np.interp(t_ref, tr_an.t, tr_an.y_gc)
    z_pb = np.interp(t_ref, tr_pb.t, tr_pb.z_gc)
    z_an = np.interp(t_ref, tr_an.t, tr_an.z_gc)
    B_pb = np.interp(t_ref, tr_pb.t, tr_pb.B_T)
    B_an = np.interp(t_ref, tr_an.t, tr_an.B_T)

    assert float(np.max(np.abs(x_pb - x_an))) < 1.0e-7
    assert float(np.max(np.abs(y_pb - y_an))) < 5.0e-7
    assert float(np.max(np.abs(z_pb - z_an))) < 3.0e-5
    assert float(np.max(np.abs(B_pb - B_an))) < 1.0e-8


def test_mirror_quadrature_template_reports_turning_diagnostics_on_real_field_map(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 4.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.0
  z0_m: 0.0
  vpar_sign: 1
  cyclotron_phase0_rad: 0.0
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror_quadrature
  energy_loss_model: none
  mirror_quadrature_min_theta_nodes: 513
  mirror_quadrature_max_theta_nodes: 1025
  mirror_template_max_period_rel_error: 1.0e-3
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{tmp_path.as_posix()}"
  basename: "qmirror"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_qmirror_diag.yaml"))
    track = run_pipeline(cfg)["track_dyn"]
    info = track.solver_info

    assert info["template_build"] == "mirror_quadrature"
    assert info["template_method"] == "mirror_quadrature"
    assert info["template_theta_node_count"] >= 513
    assert info["template_z_turn_positive_m"] > 0.0
    assert info["bounce_period_s"] > 0.0
    assert info["template_period_rel_error_estimate"] <= 1.0e-3
    assert np.max(np.abs(track.z_gc)) <= info["template_z_turn_positive_m"] * (1.0 + 1.0e-6)


def test_chunked_sampling_matches_one_shot_sampling_with_phase_reintegration(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 5.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.0
  z0_m: 0.0
  vpar_sign: 1
  cyclotron_phase0_rad: 0.4
features:
  include_true_orbit: true
  include_gradB: true
  include_curvature_drift: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_chunk_sample.yaml"))
    ctx = run_pipeline(cfg)
    tr_dyn = ctx["track_dyn"]
    field = ctx["field"]
    mode_map = ctx["mode_map"]
    resonance = ctx.get("resonance_curve")

    t_eval = np.linspace(0.0, cfg.simulation.track_length_s, 2048)
    tr_full = sample_dynamic_track(cfg, tr_dyn, field=field, mode_map=mode_map, resonance=resonance, t_new=t_eval)

    pieces = []
    prev_t = None
    prev_phi = None
    prev_phase = None
    for start in range(0, t_eval.size, 257):
        chunk = t_eval[start : start + 257]
        if prev_t is None:
            t_call = chunk
        else:
            t_call = np.concatenate([np.asarray([float(prev_t)], dtype=float), chunk])

        tr_chunk = sample_dynamic_track(
            cfg,
            tr_dyn,
            field=field,
            mode_map=mode_map,
            resonance=resonance,
            t_new=t_call,
            phi_gc_start_rad=prev_phi,
            phase_rf_start_rad=prev_phase,
        )
        prev_t = float(tr_chunk.t[-1])
        prev_phi = float(tr_chunk.phi_gc_rad[-1])
        prev_phase = float(tr_chunk.phase_rf[-1])
        if start > 0:
            tr_chunk = _slice_track(tr_chunk, 1)
        pieces.append(tr_chunk)

    def cat(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(p, name), dtype=float) for p in pieces])

    assert np.allclose(cat("t"), tr_full.t, rtol=0.0, atol=0.0)
    assert np.allclose(cat("x_gc"), tr_full.x_gc, rtol=0.0, atol=1.0e-12)
    assert np.allclose(cat("y_gc"), tr_full.y_gc, rtol=0.0, atol=1.0e-12)
    assert np.allclose(cat("z_gc"), tr_full.z_gc, rtol=0.0, atol=1.0e-12)
    assert np.allclose(cat("phi_gc_rad"), tr_full.phi_gc_rad, rtol=0.0, atol=1.0e-12)
    assert np.allclose(cat("phase_rf"), tr_full.phase_rf, rtol=0.0, atol=1.0e-9)
    assert np.allclose(cat("B_T"), tr_full.B_T, rtol=0.0, atol=1.0e-14)


def test_guiding_center_start_phase_is_respected(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-6
electron:
  position_reference: guiding_center
  position_coordinates: cylindrical
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.7
  z0_m: 0.0
  cyclotron_phase0_rad: 1.1
features:
  include_true_orbit: false
  include_gradB: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_gc_phase.yaml"))
    ctx = run_pipeline(cfg)
    _assert_guiding_center_start_geometry(ctx, cfg)


def test_gradB_drift_matches_repo_mu_convention_and_is_gamma_independent():
    mu = np.asarray([1.7e-15], dtype=float)
    Bmag = np.asarray([2.3], dtype=float)
    Br = np.asarray([0.4], dtype=float)
    Bz = np.asarray([2.1], dtype=float)
    dBdr = np.asarray([0.12], dtype=float)
    dBdz = np.asarray([-0.08], dtype=float)
    q = -1.602176634e-19

    expected = mu * (Bz * dBdr - Br * dBdz) / (q * Bmag * Bmag)
    got_no_gamma = gradB_drift_vphi(mu, q, Bmag, Br, Bz, dBdr, dBdz)
    got_with_gamma = gradB_drift_vphi(mu, q, Bmag, Br, Bz, dBdr, dBdz, gamma=np.asarray([1.25]))

    assert np.allclose(got_no_gamma, expected, rtol=0.0, atol=1.0e-18)
    assert np.allclose(got_with_gamma, expected, rtol=0.0, atol=1.0e-18)

    got_positive_q = gradB_drift_vphi(mu, -q, Bmag, Br, Bz, dBdr, dBdz)
    assert np.allclose(got_positive_q, -expected, rtol=0.0, atol=1.0e-18)


def test_axial_field_profile_curvature_is_zero_for_straight_uniform_field():
    profile = AxialFieldProfile.from_field(_UniformFieldMap(), r0_m=0.2, z0_m=0.0)
    assert np.allclose(profile.r_m, 0.2, rtol=0.0, atol=1.0e-15)
    assert np.allclose(profile.b_cross_kappa_phi_per_m, 0.0, rtol=0.0, atol=1.0e-15)


def test_curvature_drift_helper_matches_formula():
    gamma = np.asarray([1.04, 1.06], dtype=float)
    vpar = np.asarray([2.0e7, 1.8e7], dtype=float)
    B = np.asarray([1.2, 1.4], dtype=float)
    b_cross_kappa = np.asarray([5.0e-4, -4.0e-4], dtype=float)
    q = -1.602176634e-19

    from hopper import constants as const

    expected = gamma * const.M_E * vpar * vpar * b_cross_kappa / (q * B)
    got = curvature_drift_vphi(gamma, vpar, q, B, b_cross_kappa)
    assert np.allclose(got, expected, rtol=0.0, atol=1.0e-18)



def test_cavity_interaction_ringup_and_backreaction_controls():
    cavity = Cavity(radius_m=0.3, length_m=1.0, f0_hz=560.0e6, Q=1000.0)
    interaction = CavityInteraction(
        cavity=cavity,
        ringup_enabled=True,
        back_reaction_enabled=True,
        stimulated_back_reaction=True,
        source_power_scale=1.0,
        back_reaction_scale=1.0,
    )

    B = np.asarray([0.02075], dtype=float)
    gamma = np.asarray([1.036], dtype=float)
    mu = np.asarray([2.0e-16], dtype=float)
    source_power = interaction.source_power_W(B, gamma, mu, coupling=0.8, resonance_response=0.5)
    assert float(source_power[0]) > 0.0

    field_work = interaction.field_work_power_W(source_power, stored_energy_J=0.0)
    stored = interaction.advance_stored_energy_J(0.0, float(field_work[0]), cavity.tau_E)
    assert stored > 0.0
    assert interaction.output_power_W(stored) > 0.0

    off = CavityInteraction(cavity=cavity, back_reaction_enabled=False)
    assert np.allclose(off.back_reaction_power_W(source_power, stored), 0.0)


def test_cavity_source_power_scales_with_q():
    low_q = CavityInteraction(cavity=Cavity(radius_m=0.3, length_m=1.0, f0_hz=560.0e6, Q=500.0))
    high_q = CavityInteraction(cavity=Cavity(radius_m=0.3, length_m=1.0, f0_hz=560.0e6, Q=1000.0))

    B = np.asarray([0.02075], dtype=float)
    gamma = np.asarray([1.036], dtype=float)
    mu = np.asarray([2.0e-16], dtype=float)
    p_low = low_q.source_power_W(B, gamma, mu, coupling=1.0, resonance_response=1.0)
    p_high = high_q.source_power_W(B, gamma, mu, coupling=1.0, resonance_response=1.0)
    assert np.allclose(p_high, 2.0 * p_low, rtol=1.0e-13, atol=0.0)


def test_config_rejects_incoherent_dynamics_modes(tmp_path: Path):
    cfg_text = """
simulation:
  track_length_s: 1.0e-8
trap:
  field_map_npz: does_not_exist.npz
  placeholder_if_missing: true
dynamics:
  axial_strategy: direct
  template_build: mirror
  energy_loss_model: per_bounce
output:
  out_dir: "{out_dir}"
  write_npz: false
  write_root: false
"""
    try:
        load_config(_write_cfg(tmp_path, cfg_text, name="bad_direct_per_bounce.yaml"))
    except ValueError as exc:
        assert "per_bounce" in str(exc) or "template_build" in str(exc)
    else:
        raise AssertionError("Expected incoherent direct/per_bounce config to be rejected")


def test_config_rejects_instantaneous_frequency_reference_without_true_orbit(tmp_path: Path):
    cfg_text = """
features:
  include_true_orbit: false
trap:
  field_map_npz: does_not_exist.npz
  placeholder_if_missing: true
dynamics:
  cyclotron_frequency_reference: instantaneous
output:
  out_dir: "{out_dir}"
  write_npz: false
  write_root: false
"""
    try:
        load_config(_write_cfg(tmp_path, cfg_text, name="bad_ref.yaml"))
    except ValueError as exc:
        assert "cyclotron_frequency_reference" in str(exc)
    else:
        raise AssertionError("Expected instantaneous reference without true orbit to be rejected")


def test_constants_presets_are_selectable_and_affect_mec2():
    from hopper import constants as const

    pdg = const.configure_constants("pdg_2022")
    locust_2006 = const.configure_constants("locust_kassiopeia_2006")
    try:
        assert const.active_constants_name() == "locust_kassiopeia_2006"
        assert not np.isclose(pdg.mec2_ev, locust_2006.mec2_ev, rtol=0.0, atol=1.0e-7)
    finally:
        const.configure_constants("pdg_2022")


def test_phase_and_reported_frequency_use_same_reference_frame(tmp_path: Path):
    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-7
electron:
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.16
  phi0_rad: 0.4
  z0_m: 0.0
features:
  include_true_orbit: true
  include_gradB: false
  include_curvature_drift: false
trap:
  field_map_npz: "{REAL_FIELD_MAP.as_posix()}"
  generate_if_missing: false
  placeholder_if_missing: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
  cyclotron_frequency_reference: instantaneous
  true_orbit_phase_iterations: 1
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{{out_dir}}"
  basename: "t"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_phase_ref.yaml"))
    tr = run_pipeline(cfg)["track_dyn"]
    if tr.t.size > 4:
        phase_slope_hz = np.diff(tr.phase_rf) / (2.0 * np.pi * np.diff(tr.t))
        fc_mid = 0.5 * (tr.f_c_hz[:-1] + tr.f_c_hz[1:])
        assert np.allclose(phase_slope_hz, fc_mid, rtol=5.0e-4, atol=2.0e3)


def test_vector_e_field_mode_map_loads_and_drives_cavity(tmp_path: Path):
    from hopper.cavity.mode_map import VectorElectricFieldModeMap

    # Tiny periodic cylindrical-coordinate map with Cartesian Ex,Ey,Ez components.
    fld = tmp_path / "tiny_vector.fld"
    rows = [
        "#Rho, Phi, Z, Vector data \"Vector_E\"\n",
    ]
    for rho in (0.0, 0.01):
        for phi in (0.0, math.pi, 2.0 * math.pi):
            for z in (-0.01, 0.01):
                rows.append(f"{rho:.12e} {phi:.12e} {z:.12e} 1.0 0.0 0.0\n")
    fld.write_text("".join(rows))

    cavity = Cavity(radius_m=0.1, length_m=0.2, f0_hz=560.0e6, Q=400.0)
    mode_map = VectorElectricFieldModeMap.from_fld(
        fld,
        cavity=cavity,
        component_basis="cartesian",
        gyro_quadrature_points=8,
    )
    scalar = mode_map(np.asarray([0.005]), np.asarray([0.0]))
    assert np.isfinite(scalar[0])
    assert scalar[0] > 0.0

    # Horizontal B field basis at phi=0: u1=x, u2=y.  A nonzero vector E field must
    # produce a finite fundamental work coefficient for a gyrating electron.
    drive = mode_map.gyro_drive_coupling_W_per_sqrt_J(
        r_gc_m=np.asarray([0.005]),
        phi_gc_rad=np.asarray([0.0]),
        z_gc_m=np.asarray([0.0]),
        B_T=np.asarray([0.02075]),
        gamma=np.asarray([1.036]),
        mu_J_per_T=np.asarray([2.0e-16]),
        u1=np.asarray([[1.0, 0.0, 0.0]]),
        u2=np.asarray([[0.0, 1.0, 0.0]]),
    )
    assert np.isfinite(drive[0])
    assert abs(drive[0]) > 0.0

    interaction = CavityInteraction(cavity=cavity)
    source_power = interaction.source_power_from_drive_W(drive, resonance_response=np.asarray([1.0]))
    assert source_power[0] > 0.0


def test_missing_vector_mode_map_falls_back_to_analytic(tmp_path: Path):
    from hopper.cavity.mode_map import AnalyticTE011ModeMap
    from hopper.nodes.mode_map_node import ModeMapNode

    cfg_text = """
mode_map:
  type: auto
  vector_e_field_map: missing_vector_map.fld
  fallback_to_analytic: true
output:
  out_dir: "{out_dir}"
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_vector_fallback.yaml"))
    node = ModeMapNode(cfg)
    ctx = node.run({})
    assert isinstance(ctx["mode_map"], AnalyticTE011ModeMap)
    assert ctx["mode_map_kind"] == "analytic_te011"


def test_complex_cavity_response_exact_detuned_drive_phase():
    from hopper.cavity.response import ComplexCavityResponse, integrate_complex_envelope

    cavity = Cavity(radius_m=0.1, length_m=0.2, f0_hz=560.0e6, Q=500.0)
    lo_hz = 559.0e6
    response = ComplexCavityResponse(cavity=cavity, lo_hz=lo_hz, output_coupling_fraction=0.4)
    omega_drive = 2.0 * np.pi * 560.5e6
    omega_lo = 2.0 * np.pi * lo_hz
    t = np.linspace(0.0, 8.0e-6, 4000)
    d0 = 3.0e-9 + 2.0e-9j
    drive = d0 * np.exp(1j * (omega_drive - omega_lo) * t)
    a = integrate_complex_envelope(t, drive, lambda_per_s=response.lambda_per_s, update="first_order_hold")
    expected = d0 / (0.5 * response.kappa_rad_per_s + 1j * (omega_drive - response.omega0_rad_per_s))
    ratio = a[-500:] / np.exp(1j * (omega_drive - omega_lo) * t[-500:])
    assert np.allclose(np.mean(ratio), expected, rtol=2.0e-3, atol=1.0e-16)


def test_locust_like_baseband_lpf_rejects_out_of_band_tone():
    from hopper.readout.locust_like import process_locust_like_readout

    fs_out = 22.0e6
    D = 10
    fs_fast = fs_out * D
    t = np.arange(0, 4400) / fs_fast
    pass_tone = np.exp(2j * np.pi * 5.0e6 * t)
    reject_tone = np.exp(2j * np.pi * 12.0e6 * t)
    pass_res = process_locust_like_readout(
        t_fast=t,
        iq_fast=pass_tone,
        fs_out_hz=fs_out,
        decimation_factor=D,
        lpf_cutoff_ratio=0.85,
        lpf_mode="fft_brickwall",
    )
    reject_res = process_locust_like_readout(
        t_fast=t,
        iq_fast=reject_tone,
        fs_out_hz=fs_out,
        decimation_factor=D,
        lpf_cutoff_ratio=0.85,
        lpf_mode="fft_brickwall",
    )
    assert np.mean(np.abs(pass_res.iq)) > 0.8
    assert np.mean(np.abs(reject_res.iq)) < 1.0e-6


def test_backreaction_requires_vector_map_when_requested(tmp_path: Path):
    cfg_text = """
cavity:
  back_reaction_enabled: true
  back_reaction_requires_vector_map: true
mode_map:
  type: analytic_te011
output:
  out_dir: "{out_dir}"
"""
    try:
        load_config(_write_cfg(tmp_path, cfg_text, name="cfg_strict_vector_backreaction.yaml"))
    except ValueError as exc:
        assert "requires a configured vector E-field map" in str(exc)
    else:
        raise AssertionError("Expected strict vector back-reaction validation to fail")



def test_time_evolution_uses_underdamped_frequency():
    from hopper.cavity.response import BasebandCavityResponse, TimeEvolutionCavityResponse

    cavity = Cavity(radius_m=0.1, length_m=0.2, f0_hz=560.0e6, Q=50.0)
    time_evolution = TimeEvolutionCavityResponse(cavity=cavity, lo_hz=560.0e6)
    baseband = BasebandCavityResponse(cavity=cavity, lo_hz=560.0e6)
    assert time_evolution.omega_prime_rad_per_s < baseband.omega_prime_rad_per_s
    assert time_evolution.lambda_per_s != baseband.lambda_per_s


def test_multi_track_pileup_sums_drives_before_cavity_filter(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-7
tracks:
  - energy_eV: 18563.251
    pitch_angle_deg: 89.0
    r0_m: 0.04
    phi0_rad: 0.0
    z0_m: 0.0
    cyclotron_phase0_rad: 0.0
  - energy_eV: 18563.251
    pitch_angle_deg: 89.0
    r0_m: 0.04
    phi0_rad: 0.1
    z0_m: 0.0
    cyclotron_phase0_rad: 0.3
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
features:
  include_true_orbit: false
  include_gradB: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
readout:
  model: locust_exact_baseband
  fast_decimation_factor: 2
  lpf:
    type: fft_brickwall
    cutoff_ratio_of_final_nyquist: 1.0
    n_windows: 2
output:
  out_dir: "{out_dir}"
  basename: "pileup"
  write_npz: true
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_pileup.yaml"))
    ctx = run_pipeline(cfg)
    sig = ctx["signal_result"]
    assert ctx["solver_info"]["n_tracks"] == 2
    assert sig.readout_meta["pileup_tracks"] == 2
    assert sig.readout_meta["pileup_combination"] == "coherent_drive_sum_before_single_cavity_filter"
    assert Path(ctx["npz_path"]).exists()


def test_runtime_profile_out_file_is_written(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-7
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
features:
  include_gradB: false
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
output:
  out_dir: "{out_dir}"
  basename: "timed"
  write_npz: false
  write_root: false
  write_log: true
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_profile.yaml"))
    ctx = run_pipeline(cfg)
    log_path = Path(ctx["log_path"])
    text = log_path.read_text()
    assert "Hopper runtime profile" in text
    assert "dynamics.build_track" in text
    assert "signal.synthesize" in text


def test_time_evolution_model_name_and_fast_grid_default_are_efficient(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-6
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
features:
  include_true_orbit: false
  include_gradB: false
cavity:
  response_model: time_evolution
  back_reaction_enabled: false
mode_map:
  type: analytic_te011
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
readout:
  model: locust_like_baseband
  fast_decimation_factor: 1
  lpf:
    type: none
output:
  out_dir: "{out_dir}"
  basename: "fast"
  write_npz: true
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_fast_readout.yaml"))
    ctx = run_pipeline(cfg)
    sig = ctx["signal_result"]
    assert sig.readout_meta["cavity_response_model"] == "time_evolution"
    assert sig.readout_meta["decimation_factor"] == 1
    assert sig.readout_meta["lpf_mode"] == "none"
    assert sig.readout_meta["drive_grid_samples"] == len(sig.t_if)
    assert sig.readout_meta["redundant_cavity_resampling_removed"] is True


def test_config_rejects_old_named_time_evolution_model(tmp_path: Path):
    cfg_text = """
cavity:
  response_model: rick_time_evolution
output:
  out_dir: "{out_dir}"
  write_npz: false
  write_root: false
"""
    try:
        load_config(_write_cfg(tmp_path, cfg_text, name="cfg_old_response_name.yaml"))
    except ValueError:
        pass
    else:
        raise AssertionError("old response-model name should not be accepted by the config schema")


def test_coherent_cavity_work_power_is_reported_from_time_evolution(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 5.0e-7
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
features:
  include_true_orbit: false
  include_gradB: false
cavity:
  response_model: time_evolution
  back_reaction_enabled: false
mode_map:
  type: analytic_te011
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
signal:
  fs_if_hz: 2.0e6
  if_decim: 1
readout:
  model: locust_like_baseband
  fast_decimation_factor: 1
  lpf:
    type: none
output:
  out_dir: "{out_dir}"
  basename: "work"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_work.yaml"))
    sig = run_pipeline(cfg)["signal_result"]
    assert sig.track_if.cavity_work_power_W is not None
    assert sig.track_if.cavity_work_power_W.shape == sig.track_if.t.shape
    assert np.all(np.isfinite(sig.track_if.cavity_work_power_W))


def test_legacy_response_symbol_alias_does_not_break_imports():
    from hopper.cavity.response import RickTimeEvolutionResponse, TimeEvolutionCavityResponse

    assert RickTimeEvolutionResponse is TimeEvolutionCavityResponse


def test_runtime_log_reports_pathway_details_and_compaction(tmp_path: Path):
    cfg_text = """
simulation:
  starting_time_s: 0.0
  track_length_s: 2.0e-6
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
features:
  include_true_orbit: false
  include_gradB: false
cavity:
  response_model: time_evolution
  back_reaction_enabled: false
mode_map:
  type: analytic_te011
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: none
  compact_output_dt_s: 1.0e-7
signal:
  fs_if_hz: 1.0e6
  if_decim: 1
readout:
  model: locust_like_baseband
  fast_decimation_factor: 1
  lpf:
    type: none
output:
  out_dir: "{out_dir}"
  basename: "log_detail"
  write_npz: false
  write_root: false
  write_log: true
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_log_detail.yaml"))
    ctx = run_pipeline(cfg)
    text = Path(ctx["log_path"]).read_text()
    assert "Simulation pathway details" in text
    assert "[dynamics.track_0]" in text
    assert "compact_compression_applied: True" in text
    assert "energy_loss_model_effective: none" in text
    assert "[signal]" in text
    assert "rf_undersampling_used: False" in text


def test_vector_time_evolution_backreaction_runs_and_updates_state(tmp_path: Path):
    fld = tmp_path / "tiny_vector_backreaction.fld"
    rows = ["# rho phi z Ex Ey Ez\n"]
    for rho in (0.0, 0.05):
        for phi in (0.0, math.pi, 2.0 * math.pi):
            for z in (-1.0, 0.0, 1.0):
                rows.append(f"{rho:.12e} {phi:.12e} {z:.12e} 1.0e8 0.0 0.0\n")
    fld.write_text("".join(rows))

    cfg_text = f"""
simulation:
  starting_time_s: 0.0
  track_length_s: 1.0e-8
electron:
  position_reference: guiding_center
  energy_eV: 18563.251
  pitch_angle_deg: 89.0
  r0_m: 0.01
  phi0_rad: 0.0
  z0_m: 0.0
features:
  include_true_orbit: false
  include_gradB: false
cavity:
  response_model: time_evolution
  back_reaction_enabled: true
  back_reaction_requires_vector_map: true
  initial_stored_energy_J: 1.0e-18
  initial_cavity_phase_rad: 0.1
  source_power_scale: 1.0e16
  back_reaction_scale: 1.0
mode_map:
  type: vector_e_field
  vector_e_field_map: "{fld.as_posix()}"
  vector_component_basis: cartesian
  vector_gyro_quadrature_points: 4
  vector_bounds_policy: error
trap:
  field_map_npz: "does_not_exist.npz"
  generate_if_missing: false
  placeholder_if_missing: true
dynamics:
  axial_strategy: template_tiling
  template_build: mirror
  energy_loss_model: per_bounce
  dt_max_s: 1.0e-10
  per_bounce_block_bounces: 2
signal:
  fs_if_hz: 1.0e6
  lo_hz: 560.3e6
  if_decim: 1
readout:
  model: locust_like_baseband
  fast_decimation_factor: 1
  lpf:
    type: none
output:
  out_dir: "{{out_dir}}"
  basename: "br"
  write_npz: false
  write_root: false
"""
    cfg = load_config(_write_cfg(tmp_path, cfg_text, name="cfg_vector_backreaction.yaml"))
    ctx = run_pipeline(cfg)
    tr = ctx["track_dyn"]
    assert tr.energy_eV.size > 1
    assert np.all(np.isfinite(tr.energy_eV))
    assert np.all(np.isfinite(tr.mu_J_per_T))
    assert np.all(np.isfinite(tr.cavity_energy_J))
    assert not np.allclose(tr.energy_eV, tr.energy_eV[0], rtol=0.0, atol=1.0e-12)
    assert ctx["solver_info"]["cavity_back_reaction_model"] == "complex_time_evolution_signed_work"
