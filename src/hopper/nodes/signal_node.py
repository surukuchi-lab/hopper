from __future__ import annotations

from dataclasses import dataclass, replace
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
        save_ind_signals = ctx["cfg"].signal.save_ind_signals 
        ind_signals = {}
        signal = None
        
        for idx, electron_cfg in self.cfg.active_electrons():
            sig_res = synthesize_iq(self.cfg, track_dyn[idx], electron_cfg)
        
            if save_ind_signals:
                ind_signals[idx] = sig_res
        
            if signal is None:
                signal = replace(
                    sig_res,
                    iq=sig_res.iq.copy(),
                    iq_if=sig_res.iq_if.copy(),
                )
            else:
                signal.iq += sig_res.iq
                signal.iq_if += sig_res.iq_if

            print(f"[NSIG]: Electron {idx} processed") 
        ctx = dict(ctx)
        ctx["individual_signals"] = None
        # Saving individual signals (npileup * ~2e6 entries) gives us RAM headaches, so this should not be switched on carelessly
        if save_ind_signals:
            ctx["individual_signals"] = ind_signals
        ctx["signal_result"] = signal
        return ctx
