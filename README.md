# Hopper

Hopper is a Python package for synthetic single/multi-electron CRES track and IQ-signal simulation.

## Author and license

Author: E. Karim — University of Pittsburgh.

License: MIT. See `LICENSE` for the full license text and `AUTHORLIST.md` for the author list.

## Install

```bash
conda env create -f environment.yml
conda activate hopper-sim
pip install -e .
```

## Run

```bash
hopper-sim configs/example.yaml
```

## Validate

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

The test suite exercises configuration validation, field-map interpolation and generation, cavity response models, dynamic tracks, signal synthesis, output writing, and the mirror-quadrature radiation path.

## Repository layout

- `configs/` — example and smoke-test YAML configurations.
- `resources/` — compact field-map and mode-map resources used by examples and tests.
- `src/hopper/` — simulator source package.
- `tests/` — regression tests for package behavior.

See the `README.md` file in each subfolder for a short description of that directory.
