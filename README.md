# Hopper

This package generates synthetic single-electron CRES tracks and complex IQ time series.

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
cres-sim configs/example.yaml
