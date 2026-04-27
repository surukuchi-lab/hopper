"""Physical constants used across the simulator (SI units).
Temporary Locust_MC / Kassiopeia-2006 comparison constants.
"""
from __future__ import annotations

import numpy as np

C0: float = 299_792_458.0 # unchanged
E_CHARGE: float = 1.602_176_53e-19 # 1.602_176_634e-19 in Hopper previously
M_E: float = 9.109_382_6e-31 # 9.109_383_7015e-31 in Hopper previously
EPS0: float = 8.854_187_817e-12 # 8.854_187_8128e-12 previously
MU0: float = 4.0e-7 * np.pi

MEC2_EV: float = 510.998918e3 # Hard-coded in Locust_MC; Hopper had (M_E * C0 * C0) / E_CHARGE

# If Kassiopeia 2021 is used, use these values instead:
# MU0 = 4.0e-7 * np.pi * 1.00000000055
# MEC2_EV = 510.99895000e3
