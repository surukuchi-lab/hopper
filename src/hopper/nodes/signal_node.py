"""
Module: hopper.nodes.signal_node

Developer: ehtkarim
Date: April 29, 2026

Synthesizes IQ signals, including optional pileup handling, from dynamic tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict

from ..config import MainConfig
from ..signal.synth import synthesize_iq, synthesize_iq_pileup

@dataclass
class SignalNode:
    cfg: MainConfig
    name: str = "signal"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        #JK: Signal add loop from feature/pileup_integration
        #print("[NODE]: SignalNode")
        #
        #track_dyn = ctx["track_dyn"]
        #save_ind_signals = ctx["cfg"].signal.save_ind_signals 
        #ind_signals = {}
        #signal = None
        #
        #for idx, electron_cfg in self.cfg.active_electrons():
        #    sig_res = synthesize_iq(self.cfg, track_dyn[idx], electron_cfg)
        #
        #    if save_ind_signals:
        #        ind_signals[idx] = sig_res
        #
        #    if signal is None:
        #        signal = replace(
        #            sig_res,
        #            iq=sig_res.iq.copy(),
        #            iq_if=sig_res.iq_if.copy(),
        #        )
        #    else:
        #        signal.iq += sig_res.iq
        #        signal.iq_if += sig_res.iq_if
        tracks_dyn = ctx.get("track_dyns") or [ctx["track_dyn"]]
        field = ctx["field"]
        mode_map = ctx["mode_map"]
        resonance = ctx.get("resonance_curve")
        profiler = ctx.get("profiler")

        if profiler is not None:
            with profiler.step("signal.synthesize", n_tracks=len(tracks_dyn), readout_model=self.cfg.readout.model):
                sig_res = synthesize_iq_pileup(self.cfg, tracks_dyn, field=field, mode_map=mode_map, resonance=resonance)
            profiler.add_note(
                "signal",
                readout_model=str(self.cfg.readout.model),
                cavity_response_model=str(self.cfg.cavity.response_model),
                rf_grid_kind=str(getattr(sig_res, "rf_grid_kind", "unknown")),
                rf_grid_is_uniform_time=bool(getattr(sig_res, "rf_grid_is_uniform_time", True)),
                fast_samples=int(sig_res.t.size),
                if_samples=int(sig_res.t_if.size),
                fs_hz=float(sig_res.fs_hz),
                fs_if_hz=float(sig_res.fs_if_hz),
                f_lo_hz=float(sig_res.f_lo_hz),
                amplitude_normalization=float(getattr(sig_res, "amplitude_normalization", 1.0)),
                readout_meta=dict(getattr(sig_res, "readout_meta", {}) or {}),
                mode_map_counters_after_signal=(mode_map.counter_snapshot() if hasattr(mode_map, "counter_snapshot") else {}),
            )
        else:
            sig_res = synthesize_iq_pileup(self.cfg, tracks_dyn, field=field, mode_map=mode_map, resonance=resonance)

            print(f"[NSIG]: Electron {idx} processed") 
        ctx = dict(ctx)
        ctx["individual_signals"] = None
        # Saving individual signals (npileup * ~2e6 entries) gives us RAM headaches, so this should not be switched on carelessly
        if save_ind_signals:
            ctx["individual_signals"] = ind_signals
        ctx["signal_result"] = signal
        return ctx
