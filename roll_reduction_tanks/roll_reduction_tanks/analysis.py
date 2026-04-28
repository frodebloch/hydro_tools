"""Frequency-domain analysis utilities built on top of the time-domain
simulator.

Currently provides :func:`compute_rao` which sweeps a callable
"system builder" over a list of wave frequencies, runs each sim to
steady state and extracts the steady-state roll amplitude per unit wave
amplitude — i.e. the roll RAO. This is more general than a closed-form
linear RAO calculation because the system builder can include any
combination of nonlinear tanks, controllers and quadratic damping; the
trade-off is computational cost (one full time-domain run per frequency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .coupling import CoupledSystem
from .simulation import run_simulation
from .waves import RegularWave, roll_moment_from_pdstrip


@dataclass
class RAOPoint:
    omega: float
    amplitude_deg: float
    rao_deg_per_m: float


def compute_rao(
    system_builder: Callable[[Callable[[float], float]], CoupledSystem],
    pdstrip_data,
    omegas: Sequence[float],
    heading_deg: float = 90.0,
    speed: float = 0.0,
    wave_amplitude: float = 1.0,
    n_periods: int = 30,
    n_periods_steady: int = 8,
    dt_per_period: int = 200,
) -> list[RAOPoint]:
    """Sweep ``omegas`` and return the steady-state roll RAO at each.

    Parameters
    ----------
    system_builder
        Callable ``M_wave_func -> CoupledSystem``. Must build a *fresh*
        system on each call (so initial conditions and tank state are not
        carried over between sweeps).
    pdstrip_data
        :class:`PdstripRAO` used to back out the wave-exciting moment.
    omegas
        Wave (absolute, not encounter) frequencies, rad/s.
    heading_deg, speed
        Wave heading and ship speed (passed straight to
        :class:`RegularWave`).
    wave_amplitude
        Wave amplitude in metres used for the back-out (cancels in the
        RAO; kept tunable for numerical conditioning).
    n_periods
        Total number of wave periods to integrate per frequency.
    n_periods_steady
        Number of trailing periods over which the steady-state amplitude
        is measured (peak absolute roll angle).
    dt_per_period
        Time-steps per wave period (encounter period). 200 is a generous
        default.

    Returns
    -------
    list[RAOPoint]
    """
    points: list[RAOPoint] = []
    for omega in omegas:
        wave = RegularWave(
            omega=omega,
            amplitude=wave_amplitude,
            heading_deg=heading_deg,
            speed=speed,
        )
        M_wave = roll_moment_from_pdstrip(wave, pdstrip_data)
        system = system_builder(M_wave)

        # Use encounter frequency to pick a sensible step.
        from .waves import encounter_frequency
        omega_e = encounter_frequency(omega, heading_deg, speed,
                                      g=pdstrip_data.g)
        T_e = 2 * np.pi / abs(omega_e)
        dt = T_e / dt_per_period
        t_end = n_periods * T_e

        results = run_simulation(system, dt=dt, t_end=t_end)
        n_tail = int(n_periods_steady * dt_per_period)
        amp_rad = float(np.max(np.abs(results.phi[-n_tail:])))
        amp_deg = np.rad2deg(amp_rad)
        points.append(RAOPoint(
            omega=omega,
            amplitude_deg=amp_deg,
            rao_deg_per_m=amp_deg / wave_amplitude,
        ))
    return points
