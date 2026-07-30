"""
Module: hopper.cavity.interaction

Developer: ehtkarim
Date: April 29, 2026

Computes analytic cavity-coupling quantities such as radiated power and mode amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as const
from .cavity import Cavity


@dataclass(frozen=True)
class CavityInteraction:
    """Compact-grid cavity back-reaction helper.

    The coherent IQ path is handled by :mod:`hopper.cavity.response`.  This class is
    restricted to the compact dynamics/radiation grid, where it supplies the local
    vector-map source power and the optional stimulated back-reaction work used to
    update the electron state.  It deliberately avoids the old scalar Larmor/Purcell
    source model for vector-map production runs.
    """

    cavity: Cavity
    excitation_enabled: bool = True
    ringup_enabled: bool = True
    back_reaction_enabled: bool = True
    stimulated_back_reaction: bool = True
    mode_volume_m3: float | None = None
    source_power_scale: float = 1.0
    back_reaction_scale: float = 1.0
    initial_energy_J: float = 0.0

    # Complex coherent back-reaction state.  These parameters mirror the signal-side
    # time-evolution operator so the compact trajectory solver and final IQ readout
    # use the same cavity dynamics.
    response_model: str = "time_evolution"
    lo_hz: float | None = None
    initial_phase_rad: float = 0.0
    cyclotron_phase0_rad: float = 0.0

    @property
    def tau_energy_s(self) -> float:
        return max(float(self.cavity.tau_E), 1.0e-30)

    @property
    def kappa_rad_per_s(self) -> float:
        return self.cavity.w0 / max(float(self.cavity.Q), 1.0e-300)

    @property
    def decay_gamma_rad_per_s(self) -> float:
        return 0.5 * self.kappa_rad_per_s

    @property
    def omega_lo_rad_per_s(self) -> float:
        return 2.0 * np.pi * float(self.cavity.f0_hz if self.lo_hz is None else self.lo_hz)

    @property
    def omega_prime_rad_per_s(self) -> float:
        if str(self.response_model) == "time_evolution":
            g = self.decay_gamma_rad_per_s
            return float(np.sqrt(max(self.cavity.w0 * self.cavity.w0 - g * g, 0.0)))
        return float(self.cavity.w0)

    @property
    def lambda_per_s(self) -> complex:
        return self.decay_gamma_rad_per_s - 1j * (self.omega_prime_rad_per_s - self.omega_lo_rad_per_s)

    @property
    def initial_amplitude_sqrt_J(self) -> complex:
        amp = np.sqrt(max(float(self.initial_energy_J), 0.0))
        return complex(amp * np.exp(1j * float(self.initial_phase_rad)))

    @property
    def coherent_back_reaction_enabled(self) -> bool:
        return bool(self.excitation_enabled and self.back_reaction_enabled)


    def free_space_power_W(
        self,
        B_T: float | np.ndarray,
        gamma: float | np.ndarray,
        mu_J_per_T: float | np.ndarray,
    ) -> np.ndarray:
        """Legacy diagnostic free-space power; not used in coherent cavity IQ."""
        B = np.asarray(B_T, dtype=float)
        g = np.maximum(np.asarray(gamma, dtype=float), 1.0)
        mu = np.maximum(np.asarray(mu_J_per_T, dtype=float), 0.0)
        beta_perp2 = np.clip(2.0 * mu * B / (g * const.M_E * const.C0 * const.C0), 0.0, None)
        prefactor = const.E_CHARGE**4 / (6.0 * np.pi * const.EPS0 * const.M_E**2 * const.C0)
        return np.maximum(prefactor * B**2 * g**2 * beta_perp2, 0.0)

    def purcell_factor(self, resonance_response: float | np.ndarray) -> np.ndarray:
        """Legacy diagnostic Purcell-like factor; not used in coherent cavity IQ."""
        response = np.asarray(resonance_response, dtype=float)
        volume = self.mode_volume_m3
        if volume is None or volume <= 0.0:
            volume = max(0.5 * np.pi * self.cavity.radius_m**2 * self.cavity.length_m, 1.0e-30)
        wavelength_m = const.C0 / max(float(self.cavity.f0_hz), 1.0e-30)
        geometric = 3.0 * wavelength_m**3 / (4.0 * np.pi**2 * float(volume))
        return np.maximum(geometric * max(float(self.cavity.Q), 0.0) * response**2, 0.0)

    def source_power_W(
        self,
        B_T: float | np.ndarray,
        gamma: float | np.ndarray,
        mu_J_per_T: float | np.ndarray,
        coupling: float | np.ndarray,
        resonance_response: float | np.ndarray,
    ) -> np.ndarray:
        """Legacy scalar diagnostic source power.

        This method is retained for old unit tests and diagnostics only.  The production
        signal path and vector-map back-reaction use ``source_power_from_drive_W``.
        """
        coupling_arr = np.asarray(coupling, dtype=float)
        return np.maximum(
            float(self.source_power_scale)
            * self.free_space_power_W(B_T, gamma, mu_J_per_T)
            * coupling_arr**2
            * self.purcell_factor(resonance_response),
            0.0,
        )

    def source_power_from_drive_W(
        self,
        drive_sqrt_J_per_s: float | complex | np.ndarray,
        resonance_response: float | np.ndarray,
    ) -> np.ndarray:
        """Vector-drive source power used for compact energy-loss updates.

        ``drive_sqrt_J_per_s`` is the gyro-averaged q v·E_mode* coefficient.  Its
        magnitude is calibrated by ``source_power_scale`` and the field-map energy
        normalization.  The complex phase is preserved for signal synthesis in the
        response module; the compact back-reaction update uses the associated local
        source power and optional stimulated term.
        """
        if not self.excitation_enabled:
            return np.zeros_like(np.asarray(resonance_response, dtype=float), dtype=float)
        drive = np.asarray(drive_sqrt_J_per_s, dtype=np.complex128)
        response = np.asarray(resonance_response, dtype=float)
        return np.maximum(
            float(self.source_power_scale) * np.abs(drive) ** 2 * self.tau_energy_s * response ** 2,
            0.0,
        )

    def field_work_power_W(
        self,
        source_power_W: float | np.ndarray,
        stored_energy_J: float | np.ndarray,
    ) -> np.ndarray:
        """Compact-grid spontaneous/source work used by the trajectory solver.

        The old phase-maximal scalar stimulated term has intentionally been removed.
        Coherent stimulated emission/absorption is phase-sensitive and belongs to the
        complex cavity response through ``2 Re(a* d)``.  Without carrying the complex
        mode amplitude on the compact dynamics grid, adding a positive scalar
        ``sqrt(P U)`` term is not physically controlled.
        """
        del stored_energy_J
        return np.maximum(np.asarray(source_power_W, dtype=float), 0.0)

    def back_reaction_power_W(
        self,
        source_power_W: float | np.ndarray,
        stored_energy_J: float | np.ndarray,
    ) -> np.ndarray:
        if not self.back_reaction_enabled:
            return np.zeros_like(np.asarray(source_power_W, dtype=float), dtype=float)
        return np.maximum(
            float(self.back_reaction_scale) * self.field_work_power_W(source_power_W, stored_energy_J),
            0.0,
        )

    def advance_stored_energy_J(
        self,
        stored_energy_J: float | np.ndarray,
        field_work_power_W: float | np.ndarray,
        dt_s: float | np.ndarray,
    ) -> np.ndarray:
        stored_energy = np.maximum(np.asarray(stored_energy_J, dtype=float), 0.0)
        field_work_power = np.maximum(np.asarray(field_work_power_W, dtype=float), 0.0)
        dt = np.maximum(np.asarray(dt_s, dtype=float), 0.0)
        tau = self.tau_energy_s
        if not self.ringup_enabled:
            return np.maximum(field_work_power * tau, 0.0)
        decay = np.exp(-dt / tau)
        return np.maximum(stored_energy * decay + field_work_power * tau * (1.0 - decay), 0.0)

    def output_power_W(self, stored_energy_J: float | np.ndarray, output_coupling_fraction: float = 1.0) -> np.ndarray:
        frac = min(max(float(output_coupling_fraction), 0.0), 1.0)
        return frac * np.maximum(np.asarray(stored_energy_J, dtype=float), 0.0) / self.tau_energy_s

    # Legacy free-space helpers intentionally omitted from production source calculation.
    # They remain unnecessary because analytic TE011 fallback is a coherent validation path,
    # not a stimulated-backreaction production mode.
