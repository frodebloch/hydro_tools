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


@dataclass
class IrregularWave:
    """A long-crested irregular sea defined by a JONSWAP spectrum.

    The realisation is a finite sum of harmonic components with random
    phases (deterministic given a seed). Each component is treated as a
    monochromatic wave when reconstructing the wave-exciting roll moment
    via the inverse linear roll EOM (see :func:`roll_moment_from_irregular`).

    Parameters
    ----------
    Hs
        Significant wave height, m.
    Tp
        Spectral peak period, s.
    gamma
        JONSWAP peak-enhancement factor (3.3 = standard).
    heading_deg
        Wave heading (pdstrip convention; 90 deg is beam seas).
    speed
        Vessel forward speed, m/s.
    omega_min, omega_max
        Frequency band over which to discretise the spectrum, rad/s.
    n_components
        Number of harmonic components. 256-1024 is plenty for a 10-minute
        realisation; more components -> longer realisation period before
        the synthetic time series repeats.
    seed
        RNG seed for the random phases. ``None`` -> nondeterministic.

    Notes
    -----
    The sum-of-cosines synthesis used by :func:`roll_moment_from_irregular`
    is periodic with period ``2*pi/d_omega`` where ``d_omega`` is the
    (uniform) component spacing. With the defaults (band [0.05, 1.5] rad/s,
    1024 components) the period is ~4400 s, well above any sensible run
    length.
    """

    Hs: float
    Tp: float
    gamma: float = 3.3
    heading_deg: float = 90.0
    speed: float = 0.0
    omega_min: float = 0.05
    omega_max: float = 1.5
    n_components: int = 1024
    seed: int | None = None

    def discretise(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(omegas, amplitudes, phases)`` for the realisation.

        ``amplitudes[k] = sqrt(2 * S(omega_k) * d_omega)``: the deterministic
        amplitudes of the harmonic components such that
        ``Var(zeta) = sum(0.5 * a_k^2) = m0 = (Hs/4)^2``.
        """
        omegas = np.linspace(self.omega_min, self.omega_max,
                             self.n_components)
        d_omega = float(omegas[1] - omegas[0])
        S = jonswap_spectrum_omega(omegas, self.Hs, self.Tp, self.gamma)
        amps = np.sqrt(2.0 * S * d_omega)
        rng = np.random.default_rng(self.seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return omegas, amps, phases


def jonswap_spectrum_omega(
    omega: np.ndarray, Hs: float, Tp: float, gamma: float = 3.3,
) -> np.ndarray:
    """JONSWAP spectrum S(omega) [m^2 s/rad].

    Numerically renormalised so that ``trapz(S, omega) = (Hs/4)^2`` to
    machine precision over the chosen band — a band-limited realisation
    therefore has *exactly* the requested significant wave height even if
    the band cuts the spectral tails.
    """
    omega = np.asarray(omega, dtype=float)
    omega_p = 2.0 * np.pi / Tp
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    # Pierson-Moskowitz base.
    pm = (5.0 / 16.0) * Hs**2 * omega_p**4 / omega**5 * np.exp(
        -(5.0 / 4.0) * (omega_p / omega) ** 4
    )
    peak = gamma ** np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    S = pm * peak
    # Renormalise to target m0 = (Hs/4)^2 over the supplied band.
    m0 = float(np.trapezoid(S, omega))
    if m0 > 0:
        S *= (Hs / 4.0) ** 2 / m0
    return S


def roll_moment_from_irregular(
    wave: IrregularWave,
    pdstrip_data: PdstripRAO,
) -> Callable[[float], float]:
    """Build a callable ``t -> M_wave(t)`` from a JONSWAP realisation.

    Each spectral component is mapped through the same inverse linear roll
    EOM as :func:`roll_moment_from_pdstrip`, then the resulting harmonic
    moment components are summed:

    .. math::
        M(t) = \\sum_k |M_e(\\omega_k)|\\,\\cos(\\omega_{e,k} t
                                              + \\arg M_e(\\omega_k)
                                              + \\varphi_k)

    where ``M_e(omega_k) = H(omega_e) * Phi(omega_k) * a_k`` and ``a_k =
    sqrt(2 S(omega_k) d_omega)``. Phases ``varphi_k`` are taken from
    :meth:`IrregularWave.discretise`, so passing the same seed yields a
    reproducible realisation.
    """
    omegas, amps, phases = wave.discretise()
    I44 = pdstrip_data.I44
    a44 = pdstrip_data.a44_assumed
    b44 = pdstrip_data.b44_assumed
    c44 = pdstrip_data.c44_pdstrip

    omegas_e = np.empty_like(omegas)
    Me_complex = np.empty_like(omegas, dtype=complex)
    for k, w in enumerate(omegas):
        Phi = pdstrip_data.get_roll_rao(float(w), wave.heading_deg, wave.speed)
        w_e = encounter_frequency(float(w), wave.heading_deg, wave.speed,
                                  g=pdstrip_data.g)
        H = -(I44 + a44) * w_e**2 + 1j * b44 * w_e + c44
        omegas_e[k] = w_e
        Me_complex[k] = H * Phi * amps[k]

    Me_abs = np.abs(Me_complex)
    Me_arg = np.angle(Me_complex)
    total_phase = Me_arg + phases  # combine deterministic + random parts

    def M_wave(t: float) -> float:
        return float(np.sum(Me_abs * np.cos(omegas_e * t + total_phase)))

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
