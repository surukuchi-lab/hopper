import time
import sys 
sys.path.append("../")
import matplotlib.pyplot as plt
import src.hopper as hopper
import hopper.nodes.pipeline as pipe
import numpy as np
plt.style.use("ibm")

electrons = {
    0: {
        "energy_eV": 18563.251,
        "pitch_angle_deg": 89.9,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 0.0,
        "track_length_e": 1e-6,
        "cyclotron_phase0_rad": 0,
    },
    1: {"energy_eV": 18563.251,
        "pitch_angle_deg": 89.9,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 0.25/(560e6),
        "track_length_e": 1e-6,
        "cyclotron_phase0_rad": 0,
    },
    2: {"energy_eV": 18563.251,
        "pitch_angle_deg": 89.9,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 4.1/(560e6),
        "track_length_e": 1e-6,
        "cyclotron_phase0_rad": 0,
    },
}

plot_ds = 1
start = time.perf_counter()
meta, signal, signal_dict, _, track, field = pipe.run_pipeline_scriptable("../configs/lfa_pileup.yaml", electrons)
end = time.perf_counter()
print(f"run_pipeline_scriptable timing benchmark is {end - start:.6f} s")

output = {"meta": meta, "signal": signal, "signal_dict": signal_dict}
np.savez('integration', output)

#plt.plot(signal.t_if, signal.iq_if, label=f"iq_if")
#plt.scatter(signal.t, signal.iq, marker=".", label=f"iq {i}")


#window = (0, track.t[-1])#(100/(560e6), 500/(560e6)) 
window = (0, 20/(560e6))

nplots = 4
plt.figure(figsize=(10, 8))
plt.subplot(nplots, 1, 1)
#plt.axvline(1/(560e6), label="First cyclotron orbit")
plt.plot(signal.t[::plot_ds], signal.iq[::plot_ds], linewidth=2, alpha=0.8, label="iq [V]")
plt.xlim(window)
plt.legend(loc=1)
for i in signal_dict:
    plt.subplot(nplots, 1, i+2)
    #plt.axvline(electrons[i]["starting_time_e"])
    plt.plot(signal_dict[i].t[::plot_ds], signal_dict[i].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq {i} [V]")
    plt.xlim(window)
    plt.legend(loc=1)
plt.tight_layout()
plt.savefig("../imgs/3_el.png")
plt.show()
