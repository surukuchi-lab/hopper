from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hopper.config import load_config
from hopper.dynamics.axial_profile import AxialFieldProfile
from hopper.dynamics.kinematics import (
    beta_parallel2_from_B_gamma_mu,
    gamma_beta_v_from_kinetic,
    gamma_mu_after_radiation_step_fixed_upar,
    larmor_radius_m_array,
    mu_from_pitch,
    parallel_u2_from_B_gamma_mu,
)
from hopper.dynamics.track import _local_perp_basis_from_field
from hopper.dynamics.drifts import curvature_drift_vphi, gradB_drift_vphi
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
  track_length_s: 2.0e-5
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
  energy_loss_scale: 1.0e6
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

    cfg_an = load_config(_write_cfg(tmp_path, base.format(model="analytic", out_dir=str(tmp_path / "out")), name="cfg_an.yaml"))
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

    # include_true_orbit changes where B/coupling are sampled, not the guiding-center
    # gyro-orbit geometry itself. Reintroducing the old fixed-point phase feedback
    # makes these arrays diverge and over-speeds the x/y cyclotron motion.
    assert np.allclose(tr_true.x, tr_false.x, rtol=0.0, atol=5.0e-13)
    assert np.allclose(tr_true.y, tr_false.y, rtol=0.0, atol=5.0e-13)
    assert np.allclose(tr_true.z, tr_false.z, rtol=0.0, atol=5.0e-13)
    assert not np.allclose(tr_true.B_T, tr_false.B_T, rtol=0.0, atol=1.0e-15)


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
  track_length_s: 2.0e-5
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
  energy_loss_scale: 1.0e6
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
    cfg_an_text = base.replace("dynamics:\n", "dynamics:\n  energy_loss_model: analytic\n", 1)
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

    expected = gamma * 9.1093837015e-31 * vpar * vpar * b_cross_kappa / (q * B)
    got = curvature_drift_vphi(gamma, vpar, q, B, b_cross_kappa)
    assert np.allclose(got, expected, rtol=0.0, atol=1.0e-18)

