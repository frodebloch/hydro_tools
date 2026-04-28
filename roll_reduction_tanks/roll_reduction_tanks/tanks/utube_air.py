"""Air-valve U-tube anti-roll tank.

Uses the same Lagrangian-derived cross-coupling coefficients as
:mod:`tanks.utube_open` (see that module's docstring for the derivation
following Holden, Perez & Fossen 2011 and the comparison with
``brucon::simulator::TankRollModel``).

Extends :class:`OpenUtubeTank` with two sealed air chambers above each
reservoir leg, connected by a single controllable orifice (the "air
valve"). Operating modes:

  * **Fully open valve (A_v -> infinity)**: chamber pressures equalise to
    ``p_atm`` instantaneously and the tank reduces to the passive
    open-top case. (Verified by ``tests/test_utube_air_limits.py``.)
  * **Fully closed valve (A_v = 0)**: no mass exchange between chambers;
    air on each side acts as a stiff spring (isothermal compression),
    raising the tank's natural frequency. The closed-valve linearised
    stiffness is ``c_tau' = c_tau + p_atm * A_res^2 * W^2 / (2 V0)``.
  * **Active control**: an :class:`AbstractValveController` chooses the
    fractional opening at each step using only vessel kinematics.

State vector
------------
::

    [ tau, tau_dot, n1, n2 ]

where ``tau, tau_dot`` are the U-tube fluid angle (rad) and rate (rad/s),
and ``n_i = p_i * V_i = m_i R T`` is the (isothermal) "gas content" of
chamber *i*. Using ``n`` instead of ``p`` removes a divide-by-volume in
the derivative and makes the closed-valve invariance manifest
(``n_i`` constant when valve is shut).

Equations
---------
Geometry adds the chamber volumes
::

    V1(tau) = V0 - A_res * (W/2) * tau
    V2(tau) = V0 + A_res * (W/2) * tau

with ``A_res = resevoir_duct_width * tank_thickness`` and
``W = utube_duct_width + resevoir_duct_width`` (the same ``tank_width``
as in the open-tube model).

Pressures from the isothermal ideal-gas law: ``p_i = n_i / V_i``.

Orifice mass flow (subsonic, simple form):
::

    m_dot = sign(p1 - p2) * Cd * A_v * sqrt(2 * rho_air_avg * |p1 - p2|)

with ``rho_air_avg = (p1 + p2) / (2 R T)`` and ``A_v = A_v_max * u(t)``,
``u in [0, 1]``. Translated to ``n``-coordinates:
::

    n1_dot = -m_dot * R * T
    n2_dot = +m_dot * R * T

Fluid EOM (open-tube terms unchanged, plus pressure-difference forcing):
::

    a_tau * tau_ddot + b_tau * tau_dot + c_tau * tau
       + (p1 - p2) * A_res * (W/2)
       =  +a_phi * phi_ddot + c_phi * phi
          - a_y * v_dot     - a_psi * r_dot

The roll moment on the hull is the same as the open-tube case:
::

    M_roll  =  +a_phi * tau_ddot  +  c_phi * tau

(Pressure forces inside the sealed chambers cancel against the hull
pressures by Newton's third law and contribute zero net moment to the
hull at first order, since the chamber walls are rigid.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..controllers.base import AbstractValveController
from ..controllers.constant import FullyOpenValve
from .base import AbstractTank
from .utube_open import OpenUtubeConfig


@dataclass
class AirValveUtubeConfig(OpenUtubeConfig):
    """Configuration for an air-valve U-tube tank.

    Inherits the open-U-tube geometry; adds the air-side parameters.

    Parameters
    ----------
    chamber_volume_each
        Baseline (zero-tilt) air volume of each sealed chamber, m^3.
    valve_area_max
        Cross-sectional area of the orifice when fully open, m^2.
    valve_discharge_coef
        Orifice discharge coefficient ``Cd``, dimensionless. Typical 0.6.
    p_atm
        Atmospheric (initial) chamber pressure, Pa.
    air_temperature
        Air temperature, K. Held constant (isothermal).
    R_air
        Specific gas constant for air, J/(kg*K). Default 287.05.
    """

    chamber_volume_each: float = 50.0
    valve_area_max: float = 0.5
    valve_discharge_coef: float = 0.6
    p_atm: float = 101_325.0
    air_temperature: float = 293.15
    R_air: float = 287.05


class AirValveUtubeTank(AbstractTank):
    """Air-valve U-tube tank.

    State vector: ``[tau, tau_dot, n1, n2]``.
    """

    def __init__(
        self,
        config: AirValveUtubeConfig,
        controller: Optional[AbstractValveController] = None,
        tau0: float = 0.0,
        tau_dot0: float = 0.0,
    ):
        self.config = config
        self.controller = controller if controller is not None else FullyOpenValve()
        self._compute_geometry_coefficients()

        # Initial gas content: chambers at p_atm.
        n0 = config.p_atm * config.chamber_volume_each
        self.state = np.array([tau0, tau_dot0, n0, n0], dtype=float)
        self._last_tau_ddot: float = 0.0

    # ------------------------------------------------------------------ derived geometry / coefficients

    def _compute_geometry_coefficients(self) -> None:
        c = self.config
        rho, g = c.rho, c.g
        if c.resevoir_duct_width <= 0 or c.utube_duct_height <= 0:
            raise ValueError(
                "resevoir_duct_width and utube_duct_height must be strictly positive."
            )
        if c.chamber_volume_each <= 0:
            raise ValueError("chamber_volume_each must be strictly positive.")

        tank_width = c.utube_duct_width + c.resevoir_duct_width
        self.tank_width = tank_width
        self.A_res = c.resevoir_duct_width * c.tank_thickness  # one reservoir cross section

        # Open-tube coefficients (Holden, Perez & Fossen 2011 eq. 14).
        Q = rho * c.resevoir_duct_width * tank_width**2 * c.tank_thickness / 2.0
        self.Q = Q
        self.a_y = -Q
        self.a_phi = Q * (c.duct_below_waterline + c.undisturbed_fluid_height)
        self.c_phi = Q * g
        self.a_psi = -Q * c.tank_to_xcog
        self.a_tau = Q * c.resevoir_duct_width * (
            tank_width / (2.0 * c.utube_duct_height)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        self.b_tau = Q * c.tank_wall_friction_coef * c.resevoir_duct_width * (
            tank_width / (2.0 * c.utube_duct_height**2)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        self.c_tau = Q * g
        # Quadratic-damping coefficient (Holden eq. 22, mapped to tau-space:
        # b_quad = (W/2) * d_{2,n}). Same physical mechanism as in the open
        # tube -- form-drag at the duct/reservoir junction. Air-side
        # processes do not contribute here because they are modelled
        # separately via the gas pressure term.
        self.b_quad = float(c.quad_damping_coef)

        self.tau_max = (
            2.0 * (c.tank_height - c.undisturbed_fluid_height) / tank_width
        )

        # Cached for force-on-fluid term.
        self._pressure_arm = self.A_res * tank_width / 2.0  # (p1-p2)*this = restoring moment

    # ------------------------------------------------------------------ analytic properties

    @property
    def open_valve_natural_frequency(self) -> float:
        """Natural frequency in the fully-open-valve limit (== open tube)."""
        return float(np.sqrt(self.c_tau / self.a_tau))

    @property
    def closed_valve_natural_frequency(self) -> float:
        """Linearised natural frequency in the fully-closed-valve limit.

        Derivation: with the valve sealed and ``n_i = p_atm V0`` constant,
        ``p_i = p_atm V0 / V_i``. Linearising around ``tau = 0`` gives
        ``p1 - p2 ~= p_atm * A_res * W / V0 * tau``, so the extra
        restoring term is ``+ p_atm * A_res^2 * W^2 / (2 V0) * tau``.
        Hence
        ``omega_closed = sqrt( (c_tau + dk) / a_tau )`` with
        ``dk = p_atm * A_res^2 * W^2 / (2 V0)``.
        """
        c = self.config
        dk = c.p_atm * self.A_res**2 * self.tank_width**2 / (2.0 * c.chamber_volume_each)
        return float(np.sqrt((self.c_tau + dk) / self.a_tau))

    # ------------------------------------------------------------------ EOM

    def _chamber_pressures(self, tau: float, n1: float, n2: float) -> tuple:
        c = self.config
        V0 = c.chamber_volume_each
        dV = self.A_res * self.tank_width / 2.0 * tau
        # Floor volumes to avoid singularity if tau hits the wall.
        V1 = max(V0 - dV, 1e-3 * V0)
        V2 = max(V0 + dV, 1e-3 * V0)
        return n1 / V1, n2 / V2

    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        c = self.config
        tau, tau_dot, n1, n2 = state[0], state[1], state[2], state[3]

        phi = vessel_kin["phi"]
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        v_dot = vessel_kin.get("v_dot", 0.0)
        r_dot = vessel_kin.get("r_dot", 0.0)

        # Chamber pressures and pressure-difference forcing on fluid.
        p1, p2 = self._chamber_pressures(tau, n1, n2)
        dp = p1 - p2
        moment_from_gas = dp * self._pressure_arm  # acts as +stiffness on tau

        damping = (self.b_tau + self.b_quad * abs(tau_dot)) * tau_dot

        rhs = (
            +self.a_phi * phi_ddot
            + self.c_phi * phi
            - self.a_y * v_dot
            - self.a_psi * r_dot
        ) - (damping + self.c_tau * tau + moment_from_gas)

        tau_ddot = rhs / self.a_tau

        # Orifice mass flow.
        u = float(np.clip(self.controller.opening(vessel_kin, t), 0.0, 1.0))
        A_v = c.valve_area_max * u
        if A_v <= 0.0:
            n1_dot = 0.0
            n2_dot = 0.0
        else:
            rho_air_avg = (p1 + p2) / (2.0 * c.R_air * c.air_temperature)
            m_dot_mag = c.valve_discharge_coef * A_v * np.sqrt(
                2.0 * rho_air_avg * abs(dp)
            )
            m_dot = np.sign(dp) * m_dot_mag  # positive = chamber 1 -> chamber 2
            RT = c.R_air * c.air_temperature
            n1_dot = -m_dot * RT
            n2_dot = +m_dot * RT

        return np.array([tau_dot, tau_ddot, n1_dot, n2_dot])

    def step_rk4(self, vessel_kin_func, t, dt):
        super().step_rk4(vessel_kin_func, t, dt)
        # Clamp tau to the physical wall.
        if abs(self.state[0]) > self.tau_max:
            self.state[0] = np.clip(self.state[0], -self.tau_max, self.tau_max)
            if (self.state[0] > 0 and self.state[1] > 0) or (
                self.state[0] < 0 and self.state[1] < 0
            ):
                self.state[1] = 0.0
        # Floor n_i above zero to keep the gas physical.
        self.state[2] = max(self.state[2], 1e-6)
        self.state[3] = max(self.state[3], 1e-6)
        deriv = self.derivatives(self.state, vessel_kin_func(t + dt), t + dt)
        self._last_tau_ddot = float(deriv[1])

    # ------------------------------------------------------------------ vessel-side forces

    def forces(self, vessel_kin: dict) -> dict:
        tau = self.state[0]
        tau_dot = self.state[1]
        tau_ddot = self._last_tau_ddot
        return {
            "surge": 0.0,
            "sway": -self.a_y * tau_dot,
            "roll":  self.a_phi * tau_ddot + self.c_phi * tau,
            "yaw": -self.a_psi * tau_ddot,
        }

    # ------------------------------------------------------------------ diagnostics

    def chamber_pressures(self) -> tuple:
        """Return current ``(p1, p2)`` in Pa."""
        return self._chamber_pressures(self.state[0], self.state[2], self.state[3])
