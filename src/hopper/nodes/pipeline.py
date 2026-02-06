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
    cfg = load_config(config_path)
    return run_pipeline(cfg)
