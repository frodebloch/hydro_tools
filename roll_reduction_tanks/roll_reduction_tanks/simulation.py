"""Run a coupled vessel-tank simulation and collect time-series results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .coupling import CoupledSystem


@dataclass
class SimulationResults:
    """Time-series results of a coupled simulation.

    Attributes
    ----------
    t
        Time vector, s.
    phi
        Vessel roll angle, rad.
    phi_dot
        Vessel roll rate, rad/s.
    M_wave
        Wave-exciting roll moment, N*m.
    M_tank
        Total tank-on-vessel roll moment, N*m (sum over all tanks).
    tank_states
        List with one entry per tank; each entry is a 2-D array of shape
        ``(n_steps, n_state)`` with that tank's state history.
    """

    t: np.ndarray
    phi: np.ndarray
    phi_dot: np.ndarray
    M_wave: np.ndarray
    M_tank: np.ndarray
    tank_states: list = field(default_factory=list)

    @property
    def phi_deg(self) -> np.ndarray:
        return np.rad2deg(self.phi)


def run_simulation(
    system: CoupledSystem,
    dt: float,
    t_end: float,
    t_start: float = 0.0,
) -> SimulationResults:
    """Integrate ``system`` from ``t_start`` to ``t_end`` with step ``dt``.

    The vessel and all tanks are stepped in-place; pass copies in if you
    need to preserve initial conditions for later runs.
    """
    n = int(round((t_end - t_start) / dt)) + 1
    t = t_start + dt * np.arange(n)

    phi = np.empty(n)
    phi_dot = np.empty(n)
    M_wave = np.empty(n)
    M_tank = np.empty(n)
    tank_states = [np.empty((n, len(tank.state))) for tank in system.tanks]

    # Sample initial state.
    phi[0] = system.vessel.phi
    phi_dot[0] = system.vessel.phi_dot
    M_wave[0] = system.M_wave_func(t[0])
    M_tank[0] = system._tank_roll_moment(system.vessel.kinematics())
    for j, tank in enumerate(system.tanks):
        tank_states[j][0] = tank.state

    for i in range(1, n):
        system.step(t[i - 1], dt)
        phi[i] = system.vessel.phi
        phi_dot[i] = system.vessel.phi_dot
        M_wave[i] = system.M_wave_func(t[i])
        M_tank[i] = system._tank_roll_moment(system.vessel.kinematics())
        for j, tank in enumerate(system.tanks):
            tank_states[j][i] = tank.state

    return SimulationResults(
        t=t,
        phi=phi,
        phi_dot=phi_dot,
        M_wave=M_wave,
        M_tank=M_tank,
        tank_states=tank_states,
    )
