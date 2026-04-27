from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..config import MainConfig, load_config
from .trap_node import TrapNode
from .mode_map_node import ModeMapNode
from .resonance_node import ResonanceNode
from .dynamics_node import DynamicsNode
from .signal_node import SignalNode
from .output_node import OutputNode


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
    config_path = Path(config_path)
    cfg = load_config(config_path)
    
    # Resolve file paths relative to the config file location so runs are independent of the current working directory.
    base_dir = config_path.resolve().parent
    trap = cfg.trap
    field_map_path = Path(trap.field_map_npz)
    if not field_map_path.is_absolute():
        trap.field_map_npz = str(base_dir / field_map_path)

    if trap.coil_xml is not None:
        coil_xml_path = Path(trap.coil_xml)
        if not coil_xml_path.is_absolute():
            trap.coil_xml = str(base_dir / coil_xml_path)

    res = cfg.resonance
    if res.resonance_curve is not None:
        resonance_path = Path(res.resonance_curve)
        if not resonance_path.is_absolute():
            res.resonance_curve = str(base_dir / resonance_path)

    return run_pipeline(cfg)
