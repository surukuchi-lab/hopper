import time
import sys 
sys.path.append("../")
import matplotlib.pyplot as plt
import src.hopper as hopper
import hopper.nodes.pipeline as pipe
import numpy as np
plt.style.use("ibm")


f_c_prior = 560334450

plot_ds = 10
start = time.perf_counter()
cfg = pipe.load_config("../configs/lfa_pileup.yaml")
meta, signal, signal_dict, drives_dict, tracks, field = pipe.run_pipeline_scriptable(cfg)
end = time.perf_counter()
print(f"run_pipeline_scriptable timing benchmark is {end - start:.6f} s")

output = {"meta": meta, "signal": signal, "signal_dict": signal_dict}
np.savez('integration_no_ringup', output)

#plt.plot(signal.t_if, signal.iq_if, label=f"iq_if")
#plt.scatter(signal.t, signal.iq, marker=".", label=f"iq {i}")


window = (-1e-8, 2e-6)#signal.t[-1])#(100/(560e6), 500/(560e6)) 
#window = (0, 20/(560e6))

nplots = 5
plt.figure(figsize=(10, 8))
plt.subplot(nplots, 1, 1)
#plt.axvline(1/(560e6), label="First cyclotron orbit")
plt.plot(signal.t[::plot_ds], signal.iq[::plot_ds], linewidth=2, alpha=0.8, label="iq [V]")
plt.xlim(window)
total_sum = np.zeros(len(signal_dict[0].iq), dtype=np.complex128)

for idx, sig in enumerate(signal_dict):
    if sig == "total": 
        continue
    el = signal_dict[sig]
    track_info = tracks[sig]
    f_c = np.mean(track_info.f_c_hz[:100])
    print(f_c)
    plt.subplot(nplots, 1, 1)
    plt.plot(signal_dict[sig].t[::plot_ds], signal_dict[sig].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq {idx} [V]")
    total_sum += signal_dict[sig].iq
    plt.subplot(nplots, 2, 2*idx+3)
    #plt.axvline(electrons[i]["starting_time_e"])
    #starting_t = meta["tracks"][i]["starting_time_e"]
    #phi_glob = meta["tracks"][i]["cyclotron_phase0_rad"]
    #plt.title(f"omega_c = {f_c:.2e} Hz, phi = {(phi_glob%(2*np.pi))/(2*np.pi)*360:.2f}°")
    plt.plot(signal_dict[sig].t[::plot_ds], signal_dict[sig].iq[::plot_ds], linewidth=2, alpha=0.8, label=f"iq {idx} [V]")
    plt.xlim(window)
    plt.legend(loc=1)
    plt.subplot(nplots, 2, 2*idx+4)
    #plt.title(f"Starting time {starting_t:.2e} zoomed in")
    #plt.plot(signal_dict[0].t[::plot_ds], drives_dict["total"][::plot_ds], linewidth=2, alpha=0.8, label=f"iq 0 [V]")
    #plt.plot(signal_dict[sig].t[::plot_ds], drives_dict[sig][::plot_ds], linewidth=2, alpha=0.8, label=f"iq {idx} [V]")
    plt.plot(track_info.t[::plot_ds], track_info.x[::plot_ds], linewidth=2, alpha=0.8, label=f"z {idx} [V]")
    plt.xlim(window)
    plt.legend()
    #plt.xlim(starting_t-3e-9, starting_t+1e-8)
    
plt.subplot(nplots, 1, 5)
plt.plot(signal.t[::plot_ds], signal.iq[::plot_ds]-total_sum[::plot_ds], label="Residuals")
plt.legend(loc=1)
plt.subplot(nplots, 1, 1)
plt.legend(loc=1)
plt.tight_layout()

plt.savefig("../imgs/3_el_no_ringup_87.png")
plt.show()
