from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import MainConfig
from ..signal.synth import synthesize_iq, pileup_add

@dataclass
class SignalNode:
    cfg: MainConfig
    name: str = "signal"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print("[NODE]: SignalNode")
        
        track_dyn = ctx["track_dyn"]
        
        signals = {}
        for idx, electron_cfg in self.cfg.active_electrons():
            sig_res = synthesize_iq(self.cfg, track_dyn[idx])
            signals[idx] = sig_res
            print(f"[NSIG]: Electron {idx} processed") 
        ctx = dict(ctx)
        ctx["individual_signals"] = signals
        ctx["signal_result"] = pileup_add(signals)
        return ctx
