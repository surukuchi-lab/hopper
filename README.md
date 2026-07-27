# Hopper

This package generates synthetic single/multi-electron CRES tracks and complex IQ time series.

Key features:
- Axial bounce motion from adiabatic invariant μ
- Cyclotron motion with evolving cyclotron frequency
- Grad-B drift using ∂B/∂r from finite-difference field gradients
- Magnetic field interpolation (bilinear/linear/nearest/cubic spline)
- Trap field map generation from coil XML (axisymmetric loops)
- Analytic TE_011 mode map
- Optional cavity resonance curve from ROOT via uproot
- Outputs:
  - NPZ (time, iq)
  - Optional ROOT (track arrays)

## Install
conda env create -f environment.yml
conda activate hopper
pip install -e .

## Run
hopper-sim configs/example.yaml

## Branch-specific features

The pileup-integration branch features
- Slight increase in verbosity (Mainly due to debug purposes)

- Additional Configurations

- Configuration refactor "electron" -> "electrons" 
  - Dictionary with one or multiple keys, mapping int->ElectronConfig
  - Downwards compatible with the Electron configuration. A property in config.py is added, which automatically maps this configuration tree electron -> electrons = {0: electron}
  - Track length parameter is split up in track_length_s (simulation duration, should rename that on closer look ^^) and track_length_e (in the ElectronConfig). Sanity check built in that an electrontrack cannot go past the simulation duration 
  - signal.cyclotron_phase0_rad and electron.cyclotron_phase0_rad are merged as they should be the same thing

- Scriptable function "run_pipeline_scriptable" in pipeline.py
  - Takes the configuration of the detector from YAML, the configuration of particles from the second argument, which is the mentioned "electrons" dictionary

- Pileup capability according to mentioned electrons configuration tree
  - Processing of the electrons is done in a for loop on the node level, both in the "dynamics" and "signal" node
  - Output of the nodes does not change for the original run_pipeline. Next to ctx["signal_result"], there is a possibility to switch on a feature extracting the waveforms generated from each particle via ctx["individual_signals"], this has to be enabled in the configuration though (signal["save_ind_signals"])

