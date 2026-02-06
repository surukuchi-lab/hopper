"""Physical constants used across the simulator (SI units)."""
from __future__ import annotations

import numpy as np

C0: float = 299_792_458.0
E_CHARGE: float = 1.602_176_634e-19
M_E: float = 9.109_383_7015e-31
EPS0: float = 8.854_187_8128e-12
MU0: float = 4.0e-7 * np.pi

MEC2_EV: float = (M_E * C0 * C0) / E_CHARGE
