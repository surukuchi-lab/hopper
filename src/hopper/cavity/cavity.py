from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cavity:
    """Cavity geometry and a single TE011-like resonance parameterization."""
    radius_m: float = 0.327
    length_m: float = 4.0
    f0_hz: float = 560.3e6
    Q: float = 500.0

    @property
    def w0(self) -> float:
        return 2.0 * np.pi * self.f0_hz

    @property
    def tau_A(self) -> float:
        # amplitude time constant = 2Q / ω0
        return 2.0 * self.Q / self.w0

    @property
    def tau_E(self) -> float:
        # energy time constant = Q / ω0
        return self.Q / self.w0
