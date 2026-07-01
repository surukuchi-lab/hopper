"""
Module: hopper.constants

Developer: ehtkarim
Date: April 29, 2026

Stores physical constants and named constants presets used consistently across the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

ConstantsPresetName = Literal[
    "pdg_2022",
    "pdg_2021",
    "locust_kassiopeia_2021",
    "locust_kassiopeia_2006",
]


@dataclass(frozen=True)
class PhysicalConstants:
    name: str
    c0: float
    e_charge: float
    m_e: float
    eps0: float
    mu0: float

    @property
    def mec2_ev(self) -> float:
        return (self.m_e * self.c0 * self.c0) / self.e_charge


# Current default: CODATA 2022 / latest PDG-style SI constants.
# The 2021 preset is the 2018 CODATA set used in the 2021 PDG constants review.
# The Locust/Kassiopeia presets intentionally mirror common constants snapshots used
# in those code comparisons.  They are exposed separately so small convention-driven
# shifts in cyclotron frequency can be isolated during HF comparisons.
CONSTANT_PRESETS: Dict[str, PhysicalConstants] = {
    "pdg_2022": PhysicalConstants(
        name="pdg_2022",
        c0=299_792_458.0,
        e_charge=1.602_176_634e-19,
        m_e=9.109_383_7139e-31,
        eps0=8.854_187_8188e-12,
        mu0=1.256_637_06127e-6,
    ),
    "pdg_2021": PhysicalConstants(
        name="pdg_2021",
        c0=299_792_458.0,
        e_charge=1.602_176_634e-19,
        m_e=9.109_383_7015e-31,
        eps0=8.854_187_8128e-12,
        mu0=1.256_637_06212e-6,
    ),
    "locust_kassiopeia_2021": PhysicalConstants(
        name="locust_kassiopeia_2021",
        c0=299_792_458.0,
        e_charge=1.602_176_634e-19,
        m_e=9.109_383_7015e-31,
        eps0=8.854_187_8128e-12,
        mu0=4.0e-7 * np.pi,
    ),
    "locust_kassiopeia_2006": PhysicalConstants(
        name="locust_kassiopeia_2006",
        c0=299_792_458.0,
        e_charge=1.602_176_487e-19,
        m_e=9.109_382_15e-31,
        eps0=8.854_187_817e-12,
        mu0=4.0e-7 * np.pi,
    ),
}

_ACTIVE_PRESET_NAME: str = "pdg_2022"

C0: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].c0
E_CHARGE: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].e_charge
M_E: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].m_e
EPS0: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].eps0
MU0: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].mu0
MEC2_EV: float = CONSTANT_PRESETS[_ACTIVE_PRESET_NAME].mec2_ev


def configure_constants(preset: str = "pdg_2022") -> PhysicalConstants:
    """Select the active constants preset for subsequent simulation work."""
    global _ACTIVE_PRESET_NAME, C0, E_CHARGE, M_E, EPS0, MU0, MEC2_EV
    key = str(preset)
    if key not in CONSTANT_PRESETS:
        allowed = ", ".join(sorted(CONSTANT_PRESETS))
        raise ValueError(f"Unknown constants preset {preset!r}; allowed values: {allowed}")
    values = CONSTANT_PRESETS[key]
    _ACTIVE_PRESET_NAME = key
    C0 = float(values.c0)
    E_CHARGE = float(values.e_charge)
    M_E = float(values.m_e)
    EPS0 = float(values.eps0)
    MU0 = float(values.mu0)
    MEC2_EV = float(values.mec2_ev)
    return values


def active_constants() -> PhysicalConstants:
    """Return a value object describing the currently active constants."""
    return CONSTANT_PRESETS[_ACTIVE_PRESET_NAME]


def active_constants_name() -> str:
    return _ACTIVE_PRESET_NAME
