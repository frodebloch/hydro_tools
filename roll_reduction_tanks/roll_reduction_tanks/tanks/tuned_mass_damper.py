"""Tuned-mass-damper (TMD) anti-roll device.

The TMD is the canonical first-order representation of any passive
resonant absorber. A point mass ``m_t`` is constrained to slide
laterally in a guide rail mounted at vertical lever ``h_arm`` above
the ship COG, restrained by a linear spring ``k_t`` and a linear
dashpot ``b_t``. There is no gravitational coupling: the mass moves
horizontally in the body frame so gravity is perpendicular to its
motion direction at small roll angle (this is the key distinction
from a gravity pendulum).

State vector
------------
``[x, x_dot]`` -- lateral displacement of the TMD mass in the
ship-fixed frame, m, and its rate.

Equation of motion
------------------
A point at lever ``h_arm`` above the COG has lateral pseudo-acceleration
``h_arm * phi_ddot`` in the rotating body frame (small-angle). With no
gravity coupling the linearised EOM is therefore
::

    m_t * x_ddot + b_t * x_dot + k_t * x  =  -m_t * h_arm * phi_ddot ,

formally identical in structure to :class:`FreeSurfaceTank` (which is
just a TMD with ``m_t = m_eq``, ``k_t = m_eq omega_n^2``,
``h_arm = z_tank - z_cog``).

Roll moment on the hull
-----------------------
The mass exerts a reaction force on the hull at the mounting point.
The lateral force on the mass in the inertial frame is
``m_t * (h_arm * phi_ddot + x_ddot)``, so the reaction on the hull is
the negative of that times the lever arm ``h_arm``:
::

    M_roll_on_hull  =  -m_t * h_arm * (h_arm * phi_ddot + x_ddot)
                    =  -m_t * h_arm^2 * phi_ddot           # added inertia
                       - m_t * h_arm * x_ddot              # dynamic reaction

The first term is just rigid-body added inertia (the mass is rigidly
attached to the vessel through the spring / dashpot in the limit of
locked relative motion). The second term is the useful dynamic
cancellation, peaking when ``x`` is in quadrature with ``phi``.

Why this is "equivalent"
------------------------
Every passive resonant anti-roll device reduces, at first order, to a
TMD with a specific ``(m_t, omega_n, zeta, h_arm)`` set determined by
the device's geometry and physics:

* :class:`FreeSurfaceTank` -- lowest sloshing mode of a rectangular
  tank. ``m_t = m_eq = (8 / pi^3) tanh(pi h / L) m_fluid``.
* :class:`OpenUtubeTank` -- liquid column in the U-tube reduced to a
  single coordinate ``tau``; the moment-on-hull from
  ``+a_phi * tau_ddot`` matches ``-m_t * h_arm * x_ddot`` after a
  sign-and-units mapping. The U-tube also has an additional
  gravity-driven cross-coupling ``+c_phi * tau`` (Holden's
  formulation) that the bare TMD lacks; this can be added as an
  optional ``gravity_coupling`` parameter, but is omitted here so the
  TMD remains a clean Den-Hartog SDOF baseline.
* :class:`AirValveUtubeTank` -- same as the U-tube with an
  additional gas-spring stiffness on the fluid coordinate.

Useful applications of this class:

1.  Sanity-check baseline -- the simplest tank-side EOM in the
    codebase, useful for verifying coupling pipework.
2.  Den Hartog optimal tuning. For a rotational primary system with
    moment of inertia ``I44`` (including added mass) and natural
    frequency ``omega_p``, the equivalent rotational mass ratio is
    ``mu = m_t * h_arm^2 / I44``, and the optimal absorber tuning is
    ::

        omega_t / omega_p   =  1 / (1 + mu)
        zeta_opt            =  sqrt( 3 mu / (8 (1 + mu)^3) ) .

    For ``mu = 0.05`` (5 % effective mass ratio) this gives
    ``omega_t / omega_p = 0.952`` and ``zeta_opt = 0.127``, with a
    theoretical resonance reduction of about 87 % independent of
    primary damping. See :func:`den_hartog_optimal`.
3.  Upper-bound benchmark -- a real tank with mass ratio ``mu`` cannot
    outperform an optimally-tuned TMD with the same ``mu``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base import AbstractTank


@dataclass
class TunedMassDamperConfig:
    """Geometry / mass parameters for a tuned-mass-damper anti-roll device.

    Parameters
    ----------
    mass
        TMD point mass, kg.
    natural_frequency
        Undamped natural frequency of the TMD on its spring, rad/s.
        ``k_t = mass * natural_frequency^2``.
    z_mount
        Vertical position of the guide rail / spring mounting in the
        ship-fixed frame, m, **z-up convention** (positive upward from
        keel). ``h_arm = z_mount - z_cog``.
    z_cog
        Vertical position of the ship COG (same convention as
        ``z_mount``).
    damping_ratio
        Linear modal damping ratio of the TMD, dimensionless. Typical
        0.05 for low-friction guides; for Den Hartog optimal value see
        :func:`den_hartog_optimal`.
    """

    mass: float
    natural_frequency: float
    z_mount: float
    z_cog: float
    damping_ratio: float = 0.05


def den_hartog_optimal(
    mass: float,
    h_arm: float,
    I44_total: float,
    omega_p: float,
) -> Tuple[float, float]:
    """Den Hartog optimal absorber tuning for a rotational primary.

    Returns ``(omega_t_opt, zeta_t_opt)`` for a TMD of point mass
    ``mass`` mounted at lever ``h_arm`` from the rotation axis,
    attached to a primary rotational system of total inertia
    ``I44_total`` (including any added mass / fluid contributions) and
    natural frequency ``omega_p``.

    Den Hartog (1934, 1956) derived these for a translational primary
    with mass ratio ``mu = m_absorber / m_primary``. The rotational
    analogue uses the equivalent rotational mass ratio
    ``mu = mass * h_arm^2 / I44_total``.

    The tuning is independent of primary damping and of the
    excitation; it minimises the peak amplitude of the primary
    response (the "fixed-point" theorem).
    """
    if mass <= 0 or h_arm == 0 or I44_total <= 0 or omega_p <= 0:
        raise ValueError("mass, h_arm, I44_total, omega_p must be > 0.")
    mu = mass * h_arm * h_arm / I44_total
    omega_t = omega_p / (1.0 + mu)
    zeta_t = np.sqrt(3.0 * mu / (8.0 * (1.0 + mu) ** 3))
    return float(omega_t), float(zeta_t)


class TunedMassDamperTank(AbstractTank):
    """Linear tuned-mass-damper anti-roll device.

    State vector: ``[x, x_dot]`` -- lateral displacement (m) and rate
    of the TMD point mass in the ship-fixed frame.
    """

    def __init__(self, config: TunedMassDamperConfig,
                 x0: float = 0.0, x_dot0: float = 0.0):
        self.config = config
        self._compute_coefficients()
        self.state = np.array([x0, x_dot0], dtype=float)
        self._last_x_ddot: float = 0.0

    def _compute_coefficients(self) -> None:
        c = self.config
        if c.mass <= 0:
            raise ValueError("mass must be strictly positive.")
        if c.natural_frequency <= 0:
            raise ValueError("natural_frequency must be strictly positive.")
        self.m_t = c.mass
        self.omega_n = float(c.natural_frequency)
        self.k_t = self.m_t * self.omega_n ** 2
        self.b_t = 2.0 * c.damping_ratio * self.m_t * self.omega_n
        self.h_arm = c.z_mount - c.z_cog

    # ------------------------------------------------------------------ analytic

    @property
    def natural_frequency(self) -> float:
        return self.omega_n

    @property
    def natural_period(self) -> float:
        return 2 * np.pi / self.omega_n

    @property
    def damping_ratio(self) -> float:
        return self.config.damping_ratio

    # ------------------------------------------------------------------ EOM

    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        x, x_dot = state[0], state[1]
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        # m_t x_ddot + b_t x_dot + k_t x = -m_t h_arm phi_ddot
        rhs = -self.m_t * self.h_arm * phi_ddot - (self.b_t * x_dot + self.k_t * x)
        x_ddot = rhs / self.m_t
        return np.array([x_dot, x_ddot])

    def step_rk4(self, vessel_kin_func, t, dt):
        super().step_rk4(vessel_kin_func, t, dt)
        deriv = self.derivatives(self.state, vessel_kin_func(t + dt), t + dt)
        self._last_x_ddot = float(deriv[1])

    # ------------------------------------------------------------------ vessel-side

    def forces(self, vessel_kin: dict) -> dict:
        # Recompute x_ddot from current state and supplied vessel kinematics
        # rather than using the cached value (avoids one-step lag in the
        # explicit-Jacobi feedback path).
        x_ddot = float(self.derivatives(self.state, vessel_kin, 0.0)[1])
        # M = -m_t h_arm (h_arm phi_ddot + x_ddot). The +phi_ddot term is
        # rigid-body added inertia and is normally absorbed into the vessel's
        # I44 by the user when sizing the device; here we expose only the
        # dynamic-reaction part -m_t h_arm x_ddot, consistent with how the
        # other tanks return only the moment due to internal-state motion.
        M_roll = -self.m_t * self.h_arm * x_ddot
        return {"surge": 0.0, "sway": 0.0, "roll": M_roll, "yaw": 0.0}
