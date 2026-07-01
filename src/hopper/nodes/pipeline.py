"""
Module: hopper.nodes.pipeline

Developer: ehtkarim
Date: April 29, 2026

Assembles and executes the end-to-end Hopper simulation pipeline from a YAML configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..config import MainConfig, load_config
from .. import constants as const
from ..utils.profiling import SimulationProfiler
from .trap_node import TrapNode
from .mode_map_node import ModeMapNode
from .resonance_node import ResonanceNode
from .dynamics_node import DynamicsNode
from .signal_node import SignalNode
from .output_node import OutputNode


def _default_log_path(cfg: MainConfig) -> Path:
    if cfg.output.log_file:
        return Path(cfg.output.log_file)
    return Path(cfg.output.out_dir) / f"{cfg.output.basename}.out"


def run_pipeline(cfg: MainConfig) -> Dict[str, Any]:
    """Run the standard node pipeline with optional structured runtime profiling."""
    active_constants = const.configure_constants(cfg.physics.constants_preset)
    profiler = SimulationProfiler(enabled=bool(getattr(cfg.output, "write_log", True)))
    profiler.add_note(
        "config",
        constants_preset=cfg.physics.constants_preset,
        track_length_s=float(cfg.simulation.track_length_s),
        starting_time_s=float(cfg.simulation.starting_time_s),
        n_tracks=len(cfg.tracks) if cfg.tracks else 1,
        axial_strategy=cfg.dynamics.axial_strategy,
        energy_loss_model=cfg.dynamics.energy_loss_model,
        template_build=cfg.dynamics.template_build,
        compact_output_dt_s=cfg.dynamics.compact_output_dt_s,
        cavity_response_model=cfg.cavity.response_model,
        cavity_back_reaction_enabled=cfg.cavity.back_reaction_enabled,
        readout_model=cfg.readout.model,
        readout_fast_decimation_factor=int(cfg.readout.fast_decimation_factor),
        output_track_sampling=cfg.output.track_sampling,
    )
    ctx: Dict[str, Any] = {"cfg": cfg, "constants": active_constants, "profiler": profiler}

    nodes = [
        TrapNode(cfg),
        ModeMapNode(cfg),
        ResonanceNode(cfg),
        DynamicsNode(cfg),
        SignalNode(cfg),
        OutputNode(cfg),
    ]

    for node in nodes:
        with profiler.step(f"node.{node.name}"):
            ctx = node.run(ctx)

    if profiler.enabled:
        log_path = _default_log_path(cfg)
        profiler.write(log_path)
        ctx = dict(ctx)
        ctx["log_path"] = str(log_path)
    return ctx


def run_from_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_config(config_path)

    # Resolve input file paths relative to the config file location so runs are
    # independent of the current working directory.  Keep output.out_dir unchanged
    # so relative outputs preserve the longstanding working-directory behavior.
    base_dir = config_path.resolve().parent
    trap = cfg.trap
    field_map_path = Path(trap.field_map_npz)
    if not field_map_path.is_absolute():
        trap.field_map_npz = str(base_dir / field_map_path)

    if trap.coil_xml is not None:
        coil_xml_path = Path(trap.coil_xml)
        if not coil_xml_path.is_absolute():
            trap.coil_xml = str(base_dir / coil_xml_path)

    mode_map = cfg.mode_map
    if mode_map.vector_e_field_map is not None:
        vector_path = Path(mode_map.vector_e_field_map)
        if not vector_path.is_absolute():
            mode_map.vector_e_field_map = str(base_dir / vector_path)

    res = cfg.resonance
    if res.resonance_curve is not None:
        resonance_path = Path(res.resonance_curve)
        if not resonance_path.is_absolute():
            res.resonance_curve = str(base_dir / resonance_path)

    return run_pipeline(cfg)
