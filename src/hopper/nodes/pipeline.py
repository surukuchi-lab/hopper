from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from copy import deepcopy

from ..config import MainConfig, ElectronConfig, load_config
from .trap_node import TrapNode
from .mode_map_node import ModeMapNode
from .resonance_node import ResonanceNode
from .dynamics_node import DynamicsNode
from .signal_node import SignalNode
from .output_node import OutputNode

def build_config(base_cfg: MainConfig, electrons: dict[int, dict]) -> MainConfig:
    cfg = deepcopy(base_cfg)
    cfg.electrons = { eid: ElectronConfig(**edata) for eid, edata in electrons.items() }
    return cfg
    

def run_pipeline_scriptable(config_path: str | Path, electrons: dict[int, dict]) -> Dict[str, Any]:
    
    cfg = load_config(config_path)
    cfg = build_config(cfg, electrons)
    ctx: Dict[str, Any] = {"cfg": cfg}
    nodes = [
        TrapNode(cfg),
        ModeMapNode(cfg),
        ResonanceNode(cfg),
        DynamicsNode(cfg),
        SignalNode(cfg),
    ]

    for node in nodes:
        ctx = node.run(ctx)

    config = ctx["cfg"]
    signal = ctx["signal_result"]
    # The latter two are to intercept the dynamics at different stages in the processing
    track_if = ctx["signal_result"].track_if
    track_dyn = ctx["track_dyn"]
    field = ctx["field"]
    return config, signal, track_if, track_dyn, field

def run_pipeline(cfg: MainConfig) -> Dict[str, Any]:
    """
    Run the standard node pipeline.

    Node order:
      trap -> mode_map -> resonance -> dynamics -> signal -> output
    """
    ctx: Dict[str, Any] = {"cfg": cfg}

    nodes = [
        TrapNode(cfg),
        ModeMapNode(cfg),
        ResonanceNode(cfg),
        DynamicsNode(cfg),
        SignalNode(cfg),
        OutputNode(cfg),
    ]

    for node in nodes:
        ctx = node.run(ctx)

    return ctx


def run_from_config(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    return run_pipeline(cfg)

def run_from_config(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    return run_pipeline(cfg)
