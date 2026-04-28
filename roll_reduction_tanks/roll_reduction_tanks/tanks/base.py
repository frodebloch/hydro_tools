"""Abstract tank interface.

A roll-reduction tank is any subsystem that:

  * carries its own internal state (e.g. fluid angle, fluid velocity, gas
    chamber pressures);
  * advances that state given the vessel kinematics it is mounted on;
  * exposes the forces and moments it applies back to the vessel.

The vessel is unaware of the concrete tank class; it consumes only the
``forces()`` dictionary as an *external load*. This is the loose coupling
pattern the eventual C++ implementation will mirror.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class AbstractTank(ABC):
    """Abstract base for all roll-reduction tank models.

    Subclasses must:

      * initialise ``self.state`` (a 1-D :class:`numpy.ndarray`);
      * implement :meth:`derivatives` returning ``d state / d t``;
      * implement :meth:`forces` returning a dictionary with keys
        ``'roll', 'sway', 'yaw', 'surge'`` (all in N or N*m, body-fixed).

    The default :meth:`step_rk4` integrates the state with classical RK4.
    Tanks whose force on the vessel depends on the *current* state should
    use :meth:`forces` after :meth:`step_rk4`; for the simple explicit
    Jacobi coupling implemented in :mod:`roll_reduction_tanks.coupling`
    the tank reads vessel kinematics from the start of the time step,
    which is sufficient for stability provided ``dt`` is small relative
    to the fastest natural period.
    """

    state: np.ndarray

    @abstractmethod
    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        """Return ``d state / d t`` for the given state and frozen vessel
        kinematics."""

    @abstractmethod
    def forces(self, vessel_kin: dict) -> dict:
        """Return a dictionary of forces / moments the tank applies to the
        hull, body-fixed. Keys: ``'surge', 'sway', 'roll', 'yaw'``.
        """

    def step_rk4(
        self,
        vessel_kin_func: Callable[[float], dict],
        t: float,
        dt: float,
    ) -> None:
        """Advance the internal state by one RK4 step.

        ``vessel_kin_func`` is a callable returning the vessel kinematics
        snapshot at the requested time (typically a closure capturing the
        kinematics at the start of the step — i.e. the explicit-Jacobi
        coupling pattern).
        """
        s = self.state
        k1 = self.derivatives(s, vessel_kin_func(t), t)
        k2 = self.derivatives(s + 0.5 * dt * k1, vessel_kin_func(t + 0.5 * dt), t + 0.5 * dt)
        k3 = self.derivatives(s + 0.5 * dt * k2, vessel_kin_func(t + 0.5 * dt), t + 0.5 * dt)
        k4 = self.derivatives(s + dt * k3, vessel_kin_func(t + dt), t + dt)
        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # --- helpers --------------------------------------------------------

    def zero_forces(self) -> dict:
        return {"surge": 0.0, "sway": 0.0, "roll": 0.0, "yaw": 0.0}
