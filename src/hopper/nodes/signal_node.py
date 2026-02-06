from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import MainConfig
from ..signal.synth import synthesize_iq


@dataclass
class SignalNode:
    cfg: MainConfig
    name: str = "signal"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        track_dyn = ctx["track_dyn"]
        sig_res = synthesize_iq(self.cfg, track_dyn)

        ctx = dict(ctx)
        ctx["signal_result"] = sig_res
        return ctx
