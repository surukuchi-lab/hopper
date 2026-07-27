import time
import sys 
sys.path.append("../")
import matplotlib.pyplot as plt
import src.hopper as hopper
import hopper.nodes.pipeline as pipe
import numpy as np
plt.style.use("ibm")


f_c_prior = 560334450

electrons = {
    0: {
        "energy_eV": 18563.251,
        "pitch_angle_deg": 87,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 0.0,
        "track_length_e": 1e-5,
        "cyclotron_phase0_rad": 0,
    },
    1: {"energy_eV": 18563.251,
        "pitch_angle_deg": 87,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 560/f_c_prior, #1e-6
        "track_length_e": 1e-5,
        "cyclotron_phase0_rad": np.pi/2,
    },
    2: {"energy_eV": 18563.251,
        "pitch_angle_deg": 87,
        "r0_m": 0.16,
        "phi0_rad": 0.0,
        "z0_m": 0.0,
        "vpar_sign": 1,
        "starting_time_e": 1120/f_c_prior, #2e-6
        "track_length_e": 1e-5,
        "cyclotron_phase0_rad": np.pi,
    },
}


plot_ds = 1
start = time.perf_counter()
meta, signal, signal_dict, _, track, field = pipe.run_pipeline_scriptable("../configs/lfa_pileup.yaml", electrons)
end = time.perf_counter()
print(f"run_pipeline_scriptable timing benchmark is {end - start:.6f} s")

output = {"meta": meta, "signal": signal, "signal_dict": signal_dict}
np.savez('integration_no_ringup', output)

#plt.plot(signal.t_if, signal.iq_if, label=f"iq_if")
#plt.scatter(signal.t, signal.iq, marker=".", label=f"iq {i}")


window = (-1e-8, 3e-6)#signal.t[-1])#(100/(560e6), 500/(560e6)) 
#window = (0, 20/(560e6))

nplots = 4
plt.figure(figsize=(10, 8))
plt.subplot(nplots, 1, 1)
#plt.axvline(1/(560e6), label="First cyclotron orbit")
plt.plot(signal.t[::plot_ds], signal.iq[::plot_ds], linewidth=2, alpha=0.8, label="iq [V]")
plt.xlim(window)
plt.legend(loc=1)
for i in signal_dict:
    el = signal_dict[i]
    track_info = el.track_if
    f_c = np.mean(track_info.f_c_hz[:100])
    print(f_c)
    plt.subplot(nplots, 2, 2*i+3)
    #plt.axvline(electrons[i]["starting_time_e"])
    starting_t = electrons[i]["starting_time_e"]
    phi_glob = electrons[i]["cyclotron_phase0_rad"]
    plt.title(f"omega_c = {f_c:.2e} Hz, phi = {(phi_glob%(2*np.pi))/(2*np.pi)*360:.2f}°")
    plt.plot(signal_dict[i].t[::plot_ds], signal_dict[i].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq {i} [V]")
    plt.xlim(window)
    plt.legend(loc=1)
    plt.subplot(nplots, 2, 2*i+4)
    #plt.axvline(electrons[i]["starting_time_e"])
    plt.title(f"Starting time {starting_t:.2e} zoomed in")
    plt.plot(signal_dict[0].t[::plot_ds], signal_dict[0].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq 0 [V]")
    plt.plot(signal_dict[i].t[::plot_ds], signal_dict[i].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq {i} [V]")
    plt.xlim(starting_t-3e-9, starting_t+1e-8)
    plt.tight_layout()
plt.savefig("../imgs/3_el_no_ringup_87.png")
plt.show()
