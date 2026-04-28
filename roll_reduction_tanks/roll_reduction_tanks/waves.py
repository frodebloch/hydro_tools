"""Wave excitation utilities.

Time-domain wave-exciting roll moment is reconstructed from a complex motion
RAO via the *inverse* linear roll equation of motion:

.. math::

    M_e(\\omega) / \\zeta_a
    = \\bigl[-(I_{44}+a_{44})\\,\\omega_e^2
            + i\\,b_{44}\\,\\omega_e
            + c_{44,\\text{pdstrip}}\\bigr]
      \\cdot \\Phi_\\text{roll}(\\omega, \\beta, U)

where ``\\Phi_\\text{roll}`` is the *absolute* (rad/m) roll RAO. The
hydrostatic stiffness ``c_{44,\\text{pdstrip}}`` is the value used in the
pdstrip run (i.e. ``\\rho g \\nabla \\cdot GM_\\text{pdstrip}``); the
simulator's own stiffness is unrelated and can be set arbitrarily — see the
README section on GM decoupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .pdstrip_io import PdstripRAO


@dataclass
class RegularWave:
    """A monochromatic regular wave.

    Parameters
    ----------
    omega
        Wave circular frequency in the earth-fixed frame, rad/s.
    heading_deg
        Wave heading in pdstrip's convention. -90 deg / +90 deg are beam
        seas; 180 deg is head seas.
    amplitude
        Wave amplitude :math:`\\zeta_a`, m.
    speed
        Vessel forward speed, m/s. Default 0.
    """

    omega: float
    heading_deg: float
    amplitude: float
    speed: float = 0.0


def encounter_frequency(omega: float, heading_deg: float, speed: float, g: float = 9.81) -> float:
    """Encounter circular frequency.

    .. math::
        \\omega_e = \\omega - k\\,U\\,\\cos(\\beta)

    with deep-water wavenumber ``k = omega^2 / g``. Heading convention:
    ``\\beta = 0`` corresponds to following seas (wave travelling in the
    same direction as the ship), ``\\beta = 180`` is head seas. We use
    pdstrip's convention where the `heading_deg` value is the angle of
    incidence relative to the ship.
    """
    k = omega**2 / g
    beta = np.deg2rad(heading_deg)
    return omega - k * speed * np.cos(beta)


def roll_moment_from_pdstrip(
    wave: RegularWave,
    pdstrip_data: PdstripRAO,
) -> Callable[[float], float]:
    """Build a callable ``t -> M_wave(t)`` from a complex pdstrip RAO.

    The reconstruction uses the pdstrip-run hydrostatic context
    (``c44_pdstrip``, ``a44_assumed``, ``b44_assumed``) embedded in
    ``pdstrip_data``. The simulator's own vessel stiffness/inertia are
    irrelevant here.

    Returns
    -------
    Callable
        ``f(t) = Re{ M_e * exp(i * omega_e * t) } * zeta_a`` where
        ``M_e/zeta_a`` is the complex amplitude computed from the inverse
        linear roll EOM.
    """
    Phi = pdstrip_data.get_roll_rao(wave.omega, wave.heading_deg, wave.speed)
    omega_e = encounter_frequency(
        wave.omega, wave.heading_deg, wave.speed, g=pdstrip_data.g
    )
    I44 = pdstrip_data.I44
    a44 = pdstrip_data.a44_assumed
    b44 = pdstrip_data.b44_assumed
    c44 = pdstrip_data.c44_pdstrip

    H = -(I44 + a44) * omega_e**2 + 1j * b44 * omega_e + c44
    Me_per_amp = H * Phi  # complex amplitude of M_wave per metre wave amp.
    Me = Me_per_amp * wave.amplitude

    def M_wave(t: float) -> float:
        # Re{ M_e * exp(i omega_e t) } = |Me|*cos(omega_e t + arg(Me))
        return float((Me * np.exp(1j * omega_e * t)).real)

    return M_wave


def roll_moment_complex_amplitude(
    wave: RegularWave,
    pdstrip_data: PdstripRAO,
) -> tuple[complex, float]:
    """Return ``(M_e * zeta_a, omega_e)`` — the complex amplitude of the
    wave-exciting roll moment, plus the encounter frequency. Useful when an
    analytic closed-form RAO calculation is needed elsewhere.
    """
    Phi = pdstrip_data.get_roll_rao(wave.omega, wave.heading_deg, wave.speed)
    omega_e = encounter_frequency(
        wave.omega, wave.heading_deg, wave.speed, g=pdstrip_data.g
    )
    I44 = pdstrip_data.I44
    a44 = pdstrip_data.a44_assumed
    b44 = pdstrip_data.b44_assumed
    c44 = pdstrip_data.c44_pdstrip
    H = -(I44 + a44) * omega_e**2 + 1j * b44 * omega_e + c44
    return H * Phi * wave.amplitude, omega_e
