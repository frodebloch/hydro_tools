"""Abstract valve controller interface.

A controller observes vessel kinematics (and optionally clock time) and
produces a single scalar in ``[0, 1]`` representing the fractional valve
opening. The controller is *not* given access to tank state — this matches
the user's requirement that the loose-coupled architecture should not
require the active strategy to know anything about the tank's internal
fluid/gas state. This restriction also makes the controller trivially
portable to a real installation where the only available signals are
roll angle, roll rate and (estimated) roll acceleration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractValveController(ABC):
    """Abstract base class for an air-valve controller.

    Subclasses implement :meth:`opening` returning a value in ``[0, 1]``.
    Time ``t`` is passed for stateless time-dependent strategies (e.g.
    duty-cycle, scripted profiles).
    """

    @abstractmethod
    def opening(self, vessel_kin: dict, t: float) -> float:
        """Return fractional valve opening in ``[0, 1]``.

        Parameters
        ----------
        vessel_kin
            Vessel kinematics dict with at least ``phi``, ``phi_dot``,
            ``phi_ddot``.
        t
            Current time, s.
        """
