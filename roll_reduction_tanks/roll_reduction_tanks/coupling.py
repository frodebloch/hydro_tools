"""Vessel + tank explicit-Jacobi coupling.

The tank and the vessel are advanced *in parallel* over each macro time
step ``dt``: each integrator sees the other subsystem's state frozen at
the start of the step. This is the simplest stable scheme that preserves
the loose-coupling architecture (no mutual knowledge of internal state
layouts) and is what the eventual C++ implementation in ``brucon::simulator``
will mirror — the vessel will receive the tank moment as just another
external load.

For the tank-on-vessel feedback we evaluate the tank's :meth:`forces`
once at the start of the step using the *current* vessel kinematics,
then hold that force constant during the vessel's RK4 sub-stages. This
is a first-order operator-splitting; it is consistent and stable
provided ``dt`` is small relative to the fastest natural period of the
coupled system. For the CSOV / Winden-tank parameter range a step of
~25 ms (~ T_roll / 600) is more than adequate.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from .tanks.base import AbstractTank
from .vessel import RollVessel


class CoupledSystem:
    """Explicit-Jacobi coupler for a roll vessel and one or more tanks.

    Parameters
    ----------
    vessel
        :class:`RollVessel` instance (stateful).
    tanks
        Iterable of :class:`AbstractTank` instances. May be empty (then
        the coupler reduces to a plain wave-only vessel integrator).
    M_wave_func
        Callable ``t -> M_wave(t)`` returning the wave-exciting roll
        moment in N*m. Typically built by
        :func:`roll_reduction_tanks.waves.roll_moment_from_pdstrip`.
    """

    def __init__(
        self,
        vessel: RollVessel,
        tanks: Iterable[AbstractTank],
        M_wave_func: Callable[[float], float],
    ):
        self.vessel = vessel
        self.tanks = list(tanks)
        self.M_wave_func = M_wave_func

    # ------------------------------------------------------------------ helpers

    def _tank_roll_moment(self, vessel_kin: dict) -> float:
        """Sum of roll moments applied by all tanks on the hull."""
        total = 0.0
        for tank in self.tanks:
            total += tank.forces(vessel_kin)["roll"]
        return total

    # ------------------------------------------------------------------ stepping

    def step(self, t: float, dt: float) -> None:
        """Advance the coupled system by one macro step ``dt``.

        Order of operations (explicit Jacobi):

          1. Snapshot vessel kinematics ``v_kin0`` at start of step.
          2. Evaluate tank-on-vessel roll moment from ``v_kin0`` and the
             tanks' current state.
          3. Integrate the vessel over ``[t, t+dt]`` with external moment
             ``M_wave(t) + M_tank_on_vessel`` (frozen).
          4. Integrate each tank over ``[t, t+dt]`` with vessel
             kinematics frozen at ``v_kin0``.
        """
        v_kin0 = self.vessel.kinematics()
        M_tank_on_vessel = self._tank_roll_moment(v_kin0)

        def M_ext(tt: float) -> float:
            return self.M_wave_func(tt) + M_tank_on_vessel

        self.vessel.step_rk4(M_ext, t, dt)

        def vessel_kin_const(tt: float) -> dict:
            return v_kin0

        for tank in self.tanks:
            tank.step_rk4(vessel_kin_const, t, dt)
