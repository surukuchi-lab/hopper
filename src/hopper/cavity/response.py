"""
Module: hopper.cavity.response

Developer: ehtkarim
Date: April 29, 2026

Implements baseband and time-evolution cavity response models for converting mode drive into signal work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cavity import Cavity

UpdateKind = Literal["zero_order_hold", "first_order_hold", "zoh", "foh"]
ResponseKind = Literal["time_evolution", "baseband_envelope"]


@dataclass(frozen=True)
class BasebandCavityResponse:
    """Energy-normalized single-mode complex-envelope cavity response.

    Hopper uses the positive analytic-IQ convention

        V_RF(t) = Re{IQ(t) exp(+i omega_LO t)}.

    The mode amplitude ``a`` has units sqrt(J), so ``abs(a)**2`` is stored
    energy.  The baseband production equation is

        da/dt = (-kappa/2 + i(omega0 - omega_LO)) a + d(t).
    """

    cavity: Cavity
    lo_hz: float
    output_coupling_fraction: float = 1.0
    port_phase_rad: float = 0.0
    initial_energy_J: float = 0.0
    initial_phase_rad: float = 0.0

    @property
    def omega0_rad_per_s(self) -> float:
        return float(self.cavity.w0)

    @property
    def omega_prime_rad_per_s(self) -> float:
        # Baseband model uses the cavity eigenfrequency in the rotating frame.
        return self.omega0_rad_per_s

    @property
    def omega_lo_rad_per_s(self) -> float:
        return 2.0 * np.pi * float(self.lo_hz)

    @property
    def kappa_rad_per_s(self) -> float:
        return self.omega0_rad_per_s / max(float(self.cavity.Q), 1.0e-300)

    @property
    def decay_gamma_rad_per_s(self) -> float:
        return 0.5 * self.kappa_rad_per_s

    @property
    def kappa_out_rad_per_s(self) -> float:
        frac = min(max(float(self.output_coupling_fraction), 0.0), 1.0)
        return frac * self.kappa_rad_per_s

    @property
    def kappa_internal_rad_per_s(self) -> float:
        return max(self.kappa_rad_per_s - self.kappa_out_rad_per_s, 0.0)

    @property
    def lambda_per_s(self) -> complex:
        return self.decay_gamma_rad_per_s - 1j * (self.omega_prime_rad_per_s - self.omega_lo_rad_per_s)

    @property
    def initial_amplitude_sqrt_J(self) -> complex:
        amp = np.sqrt(max(float(self.initial_energy_J), 0.0))
        return complex(amp * np.exp(1j * float(self.initial_phase_rad)))

    def output_from_amplitude(self, a: np.ndarray | complex) -> np.ndarray:
        return np.exp(1j * float(self.port_phase_rad)) * np.sqrt(max(self.kappa_out_rad_per_s, 0.0)) * np.asarray(a, dtype=np.complex128)


@dataclass(frozen=True)
class TimeEvolutionCavityResponse(BasebandCavityResponse):
    """O(N) damped-resonator time-evolution operator.

    This is the efficient time-evolution form of the causal DHO Green-function
    convolution. In the baseband Hopper implementation the RF carrier is removed
    analytically, but the resonant phase evolves with the underdamped DHO frequency
    omega' = sqrt(omega0^2 - gamma^2), gamma = omega0/(2Q).
    """

    @property
    def omega_prime_rad_per_s(self) -> float:
        g = self.decay_gamma_rad_per_s
        return float(np.sqrt(max(self.omega0_rad_per_s * self.omega0_rad_per_s - g * g, 0.0)))


ComplexCavityResponse = BasebandCavityResponse

# Internal compatibility alias for worktrees where an older __init__.py still imports
# this symbol after patch application.  The public configuration name remains
# ``time_evolution`` and config validation rejects the retired development-only response-model name.
RickTimeEvolutionResponse = TimeEvolutionCavityResponse


def make_cavity_response(
    *,
    response_model: ResponseKind,
    cavity: Cavity,
    lo_hz: float,
    output_coupling_fraction: float = 1.0,
    port_phase_rad: float = 0.0,
    initial_energy_J: float = 0.0,
    initial_phase_rad: float = 0.0,
) -> BasebandCavityResponse:
    cls = TimeEvolutionCavityResponse if str(response_model) == "time_evolution" else BasebandCavityResponse
    return cls(
        cavity=cavity,
        lo_hz=float(lo_hz),
        output_coupling_fraction=float(output_coupling_fraction),
        port_phase_rad=float(port_phase_rad),
        initial_energy_J=float(initial_energy_J),
        initial_phase_rad=float(initial_phase_rad),
    )


def _foh_i1(lambda_per_s: complex, h_s: float, one_minus_e: complex) -> complex:
    z = lambda_per_s * h_s
    if abs(z) < 1.0e-5:
        return 0.5 * h_s - (lambda_per_s * h_s * h_s) / 6.0 + (lambda_per_s * lambda_per_s * h_s**3) / 24.0
    return 1.0 / lambda_per_s - one_minus_e / (lambda_per_s * lambda_per_s * h_s)


def integrate_complex_envelope(
    t_s: np.ndarray,
    drive_sqrt_J_per_s: np.ndarray,
    *,
    lambda_per_s: complex,
    initial_amplitude_sqrt_J: complex = 0.0j,
    update: UpdateKind = "first_order_hold",
) -> np.ndarray:
    """Integrate a first-order cavity mode exactly per time step.

    The function supports nonuniform grids and implements both ZOH and FOH exact
    updates using expm1 safeguards.  It is used by both preferred response models:
    the DHO time-evolution operator and Hopper's energy-normalized baseband
    operator.
    """
    t = np.asarray(t_s, dtype=float)
    d = np.asarray(drive_sqrt_J_per_s, dtype=np.complex128)
    if t.ndim != 1 or d.ndim != 1 or t.size != d.size:
        raise ValueError("t_s and drive_sqrt_J_per_s must be 1D arrays with equal length")
    if t.size == 0:
        return np.asarray([], dtype=np.complex128)
    if t.size > 1 and not np.all(np.diff(t) >= 0.0):
        raise ValueError("t_s must be monotonically increasing")

    lam = complex(lambda_per_s)
    if abs(lam) <= 0.0:
        raise ValueError("lambda_per_s must be nonzero for a damped cavity mode")
    kind = "first_order_hold" if str(update).lower() in {"foh", "first_order_hold"} else "zero_order_hold"
    a = np.empty(t.size, dtype=np.complex128)
    state = complex(initial_amplitude_sqrt_J)
    a[0] = state
    for i in range(t.size - 1):
        h = float(t[i + 1] - t[i])
        if h <= 0.0:
            a[i + 1] = state
            continue
        e = np.exp(-lam * h)
        one_minus_e = -np.expm1(-lam * h)
        i0 = one_minus_e / lam
        if kind == "first_order_hold":
            i1 = _foh_i1(lam, h, one_minus_e)
            state = e * state + i0 * d[i] + i1 * (d[i + 1] - d[i])
        else:
            state = e * state + i0 * d[i]
        a[i + 1] = state
    return a


def integrate_response(
    t_s: np.ndarray,
    drive_sqrt_J_per_s: np.ndarray,
    response: BasebandCavityResponse,
    *,
    update: UpdateKind = "first_order_hold",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mode amplitude, port output, stored energy, and port power."""
    a = integrate_complex_envelope(
        t_s,
        drive_sqrt_J_per_s,
        lambda_per_s=response.lambda_per_s,
        initial_amplitude_sqrt_J=response.initial_amplitude_sqrt_J,
        update=update,
    )
    y = response.output_from_amplitude(a)
    stored = np.abs(a) ** 2
    port_power = np.abs(y) ** 2
    return a, y, stored, port_power


def drive_work_power_W(amplitude_sqrt_J: np.ndarray, drive_sqrt_J_per_s: np.ndarray) -> np.ndarray:
    """Coherent work rate from electron drive into the cavity mode.

    For da/dt = -lambda a + d, the drive contribution to stored energy is
    d|a|^2/dt = 2 Re(a* d). Positive values correspond to work done by the
    electron on the cavity; negative values represent stimulated absorption.
    """
    a = np.asarray(amplitude_sqrt_J, dtype=np.complex128)
    d = np.asarray(drive_sqrt_J_per_s, dtype=np.complex128)
    return 2.0 * np.real(np.conjugate(a) * d)
