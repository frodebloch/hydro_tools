"""1-DOF roll vessel model.

Equation of motion (linearized about upright):

    (I44 + a44) phi_ddot + b44_lin phi_dot
        + b44_quad phi_dot |phi_dot| + c44 phi  =  M_ext(t)

with hydrostatic restoring stiffness

    c44 = rho * g * displacement * GM .

The vessel is unaware of any tank; it only consumes a scalar external moment.
This is the loose-coupling pattern that the eventual C++ implementation will
mirror in `brucon::simulator`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class RollVesselConfig:
    """Configuration for a 1-DOF roll vessel.

    Parameters
    ----------
    I44
        Rigid-body roll inertia about the COG, kg*m^2.
    a44
        Added inertia in roll (assumed constant for v1), kg*m^2.
    b44_lin
        Linear roll damping, N*m*s/rad.
    b44_quad
        Quadratic roll damping, N*m*s^2/rad^2 (force = b44_quad * phi_dot * |phi_dot|).
        Default 0 — pure linear model.
    GM
        Metacentric height, m. The simulated vessel's hydrostatic restoring
        stiffness is `c44 = rho * g * displacement * GM`. This is independent
        of any GM that may have been used in the pdstrip run; see waves.py.
    displacement
        Displaced volume, m^3.
    rho
        Sea-water density, kg/m^3.
    g
        Gravity, m/s^2.
    """

    I44: float
    a44: float
    b44_lin: float
    GM: float
    displacement: float
    b44_quad: float = 0.0
    rho: float = 1025.0
    g: float = 9.81

    @property
    def c44(self) -> float:
        """Hydrostatic restoring stiffness in roll, N*m/rad."""
        return self.rho * self.g * self.displacement * self.GM

    @property
    def total_inertia(self) -> float:
        """`I44 + a44`, kg*m^2."""
        return self.I44 + self.a44

    @property
    def natural_frequency(self) -> float:
        """Undamped natural roll frequency `sqrt(c44 / (I44 + a44))`, rad/s."""
        return float(np.sqrt(self.c44 / self.total_inertia))

    @property
    def natural_period(self) -> float:
        """Undamped natural roll period, s."""
        return 2 * np.pi / self.natural_frequency

    @property
    def damping_ratio(self) -> float:
        """Linear-damping ratio (ignores quadratic term)."""
        return self.b44_lin / (2 * np.sqrt(self.c44 * self.total_inertia))


class RollVessel:
    """Stateful integrator for the 1-DOF roll vessel.

    State vector: ``[phi, phi_dot]`` in radians and rad/s.
    """

    def __init__(self, config: RollVesselConfig, phi0: float = 0.0, phi_dot0: float = 0.0):
        self.config = config
        self.state = np.array([phi0, phi_dot0], dtype=float)
        # Cached most-recently-computed angular acceleration (used as the
        # estimate of phi_ddot exposed via `kinematics()` — sufficient for
        # explicit-Jacobi coupling with a tank).
        self._last_phi_ddot: float = 0.0

    # ------------------------------------------------------------------ helpers

    @property
    def phi(self) -> float:
        return float(self.state[0])

    @property
    def phi_dot(self) -> float:
        return float(self.state[1])

    @property
    def phi_ddot(self) -> float:
        """Most recent angular acceleration estimate."""
        return self._last_phi_ddot

    def kinematics(self) -> dict:
        """Snapshot of vessel kinematics for consumption by tanks/controllers."""
        return {
            "phi": self.phi,
            "phi_dot": self.phi_dot,
            "phi_ddot": self._last_phi_ddot,
        }

    # ------------------------------------------------------------------ dynamics

    def derivatives(self, state: np.ndarray, M_ext: float) -> np.ndarray:
        """Return ``d/dt [phi, phi_dot]`` for given state and external moment."""
        c = self.config
        phi, phi_dot = state[0], state[1]
        damping = c.b44_lin * phi_dot + c.b44_quad * phi_dot * abs(phi_dot)
        restoring = c.c44 * phi
        phi_ddot = (M_ext - damping - restoring) / c.total_inertia
        return np.array([phi_dot, phi_ddot])

    def step_rk4(self, M_ext_func: Callable[[float], float], t: float, dt: float) -> None:
        """Advance the state by one RK4 step.

        Parameters
        ----------
        M_ext_func
            Callable ``t -> M_ext`` (N*m). Allowed to depend on time during the
            sub-stages; tank coupling typically supplies it as a closure that
            captures vessel kinematics at the start of the step.
        t
            Current time, s.
        dt
            Step size, s.
        """
        s = self.state
        k1 = self.derivatives(s, M_ext_func(t))
        k2 = self.derivatives(s + 0.5 * dt * k1, M_ext_func(t + 0.5 * dt))
        k3 = self.derivatives(s + 0.5 * dt * k2, M_ext_func(t + 0.5 * dt))
        k4 = self.derivatives(s + dt * k3, M_ext_func(t + dt))
        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Store the acceleration evaluated at the new state with the moment
        # at the end of the step. This is what other modules see as
        # `phi_ddot` for the next coupled step.
        self._last_phi_ddot = float(
            self.derivatives(self.state, M_ext_func(t + dt))[1]
        )

    # ------------------------------------------------------------------ utilities

    def integrate(
        self,
        M_ext_func: Callable[[float], float],
        dt: float,
        t_end: float,
        t_start: float = 0.0,
    ) -> dict:
        """Convenience: integrate the vessel alone from ``t_start`` to ``t_end``.

        Returns time series as a dict of numpy arrays.
        """
        n = int(round((t_end - t_start) / dt)) + 1
        time = t_start + dt * np.arange(n)
        phi = np.empty(n)
        phi_dot = np.empty(n)
        M_ext = np.empty(n)
        phi[0] = self.phi
        phi_dot[0] = self.phi_dot
        M_ext[0] = M_ext_func(time[0])
        for i in range(1, n):
            self.step_rk4(M_ext_func, time[i - 1], dt)
            phi[i] = self.phi
            phi_dot[i] = self.phi_dot
            M_ext[i] = M_ext_func(time[i])
        return {"t": time, "phi": phi, "phi_dot": phi_dot, "M_ext": M_ext}
