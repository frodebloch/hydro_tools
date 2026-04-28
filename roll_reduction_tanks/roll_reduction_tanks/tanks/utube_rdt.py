"""Active U-tube anti-roll tank with rim-driven thruster (RDT) in the duct.

Architecture
------------
Mechanically identical to the open-top passive U-tube
(:class:`OpenUtubeTank`), but the duct contains a rim-driven thruster
that imposes a *commanded* force ``F_RDT(t)`` on the fluid moving
through the duct. The thruster is bidirectional (rim drive can spin
either way and produce thrust in either duct direction) and is treated
here as an ideal force actuator with amplitude saturation only:

    F_RDT,actual(t) = clip( F_command(t), -F_max, +F_max )

Slew-rate limits, motor torque ramp time constants and battery / power-
budget constraints are deliberately *not* modelled here -- the prototype
focuses on the analytical performance ceiling of an actively driven
U-tube. A more realistic actuator model would wrap the command through
a first-order lag plus an instantaneous-power constraint.

Force mapping into tau-space
----------------------------
The U-tube fluid coordinate ``tau`` (rad-like, equal to the small-angle
tilt of the imaginary line joining the two reservoir free surfaces) has
generalised-force conjugate found from work-conjugacy:

    P_RDT = F_RDT * v_duct
    v_duct = (A_res / A_duct) * (W/2) * tau_dot
    => generalised force on tau:  F_tau = F_RDT * (A_res / A_duct) * (W/2)

This term is added to the RHS of the open-U-tube tau-EOM:

    a_tau tau_ddot + (b_tau + b_quad |tau_dot|) tau_dot + c_tau tau
       = +a_phi phi_ddot + c_phi phi - a_y v_dot - a_psi r_dot
         + F_RDT * (A_res / A_duct) * (W/2)

The hull-side moment is *unchanged* in form (still ``+a_phi tau_ddot
+ c_phi tau``) -- the RDT acts internally to the tank/fluid system and
does not appear directly in the vessel forces. The active force shows
up on the vessel only via the resulting fluid motion ``tau`` and
``tau_ddot``. (This is the same as Newton's-third-law accounting for
the air-valve scheme: closed-tank internal pressure differentials never
exert net force on the vessel beyond the fluid mass redistribution they
produce.)

Controllers
-----------
The thruster command ``F_RDT(t)`` is supplied by an
:class:`AbstractRDTController`. Two reference controllers are provided:

  * :class:`InverseDynamicsRDTController` -- assumes perfect knowledge
    of the wave-exciting moment ``M_wave(t)`` and inverts the coupled
    vessel + tank dynamics to find the force command that exactly
    cancels the wave moment at the vessel. This is the *ideal active*
    baseline: best-case time-domain performance, with no phase lag.
    Implementation note: the controller solves for ``F_RDT`` such
    that the resulting ``M_tank = a_phi * tau_ddot + c_phi * tau``
    cancels ``M_wave(t)``. Because ``tau_ddot`` depends linearly on
    ``F_RDT`` (via the tau-EOM), this is a one-line algebraic
    inversion at each call.

  * :class:`StateFeedbackRDTController` -- PD controller on roll angle
    and rate, with no wave knowledge. Honest but limited: cannot
    preempt the wave so the achievable reduction is bounded by the
    propagation lag through the tank dynamics. Closer in spirit to the
    existing :class:`FrequencyTrackingController` for the air-valve
    tank.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utube_open import OpenUtubeConfig, OpenUtubeTank


# =====================================================================
# Controller interface and reference implementations
# =====================================================================


class AbstractRDTController(ABC):
    """Abstract base for an RDT thrust controller.

    Subclasses implement :meth:`thrust` returning a force command in
    Newtons (positive value -> push fluid in the +tau direction, i.e.
    from port to starboard reservoir). The controller may consult
    vessel kinematics and time but is *not* given access to the tank
    state -- this preserves the loose-coupling property of the
    architecture (the controller only needs signals that would actually
    be measurable on a real ship: phi, phi_dot, phi_ddot, plus
    optionally an externally-supplied wave-moment estimate).
    """

    @abstractmethod
    def thrust(self, vessel_kin: dict, t: float) -> float:
        """Return commanded thrust in Newtons.

        Sign convention: positive value pushes fluid from port to
        starboard, i.e. drives ``tau`` more positive.
        """


@dataclass
class InverseDynamicsRDTController(AbstractRDTController):
    """Ideal active controller assuming perfect wave-moment knowledge.

    Solves at each call for the ``F_RDT`` that, given the current
    vessel kinematics, produces a tank moment ``M_tank`` exactly
    cancelling the externally-supplied wave moment ``M_wave(t)``.

    The mapping from F_RDT to M_tank (= a_phi*tau_ddot + c_phi*tau)
    factors through the tau-EOM:

        a_tau * tau_ddot = (RHS_of_passive_terms) + F_RDT * gain

    where ``gain = (A_res / A_duct) * (W/2)``. So tau_ddot is linear
    in F_RDT, and the relevant a_phi*tau_ddot contribution to M_tank
    is also linear in F_RDT. The c_phi*tau term is independent of
    F_RDT (tau is a state, not directly forced).

    Therefore::

        M_tank(F_RDT)  =  M_tank,passive  +  (a_phi/a_tau) * gain * F_RDT

    Setting M_tank = -M_wave gives::

        F_RDT  =  -(M_wave + M_tank,passive) / [(a_phi/a_tau) * gain]

    Parameters
    ----------
    M_wave_func
        Callable ``t -> M_wave(t)`` returning the (presumed known)
        wave-exciting roll moment in N*m.

    Notes
    -----
    The "passive" tank moment ``M_tank,passive`` here is the moment
    the tank would exert on the vessel *if F_RDT were zero* -- it
    depends on the current tau, tau_dot, phi, phi_dot, phi_ddot. The
    controller needs the full set of these signals (provided via
    vessel_kin and a small extra hook in :class:`RDTUtubeTank` that
    publishes the current tau / tau_dot to the controller's namespace
    via a callback set on construction).

    Because the controller needs the tank's current ``a_phi``, ``a_tau``
    etc., and access to ``tau``, ``tau_dot`` (which strictly speaking
    violates the loose-coupling restriction for this *baseline*
    controller), the tank passes these in via a setter at registration
    time. This is a deliberate exception for the ideal-baseline
    controller; the realistic :class:`StateFeedbackRDTController`
    keeps the strict loose-coupling.
    """

    M_wave_func: Callable[[float], float]

    # --- set by RDTUtubeTank.attach_controller() at registration ------
    _tank: Optional["RDTUtubeTank"] = None

    def attach_tank(self, tank: "RDTUtubeTank") -> None:
        self._tank = tank

    def thrust(self, vessel_kin: dict, t: float) -> float:
        if self._tank is None:
            raise RuntimeError(
                "InverseDynamicsRDTController.thrust() called before "
                "attach_tank(); RDTUtubeTank should call attach_tank() "
                "in its constructor."
            )
        tank = self._tank
        # Current state
        tau, tau_dot = float(tank.state[0]), float(tank.state[1])
        phi = float(vessel_kin["phi"])
        phi_ddot = float(vessel_kin.get("phi_ddot", 0.0))
        v_dot = float(vessel_kin.get("v_dot", 0.0))
        r_dot = float(vessel_kin.get("r_dot", 0.0))

        # Passive RHS of the tau-EOM (without RDT term)
        damping = (tank.b_tau + tank.b_quad * abs(tau_dot)) * tau_dot
        rhs_passive = (
            +tank.a_phi * phi_ddot
            + tank.c_phi * phi
            - tank.a_y * v_dot
            - tank.a_psi * r_dot
        ) - damping - tank.c_tau * tau

        tau_ddot_passive = rhs_passive / tank.a_tau
        M_tank_passive = tank.a_phi * tau_ddot_passive + tank.c_phi * tau

        # Linear sensitivity of M_tank to F_RDT
        gain = tank._rdt_gain                     # (A_res/A_duct) * (W/2)
        dMtank_dFrdt = (tank.a_phi / tank.a_tau) * gain

        if dMtank_dFrdt == 0.0:
            return 0.0

        M_wave = float(self.M_wave_func(t))
        F_cmd = -(M_wave + M_tank_passive) / dMtank_dFrdt
        return F_cmd


@dataclass
class StateFeedbackRDTController(AbstractRDTController):
    """PD controller on roll angle and rate; no wave knowledge.

    Honest baseline for what is achievable from on-board signals only:

        F_RDT = -K_phi * phi - K_phidot * phi_dot

    Sign: with the convention that positive F_RDT drives tau more
    positive (port -> starboard fluid flow), and noting that a
    starboard-down hull tilt (phi > 0) wants a tank moment in the
    -phi direction (i.e. M_tank < 0), and that M_tank = a_phi*tau_ddot
    + c_phi*tau follows tau, we want positive phi to drive tau
    *negative* (starboard -> port). Hence the negative gains above.

    Tuning: starting point is K_phidot ~ b_crit_vessel (critical
    damping of bare vessel) and K_phi ~ 0; pure rate damping is the
    most robust unforced-vessel-stabiliser policy. Mixing in a
    proportional term can sharpen response but risks stability if the
    tank-vessel coupling adds additional phase.
    """

    K_phi: float = 0.0           # N / rad
    K_phidot: float = 0.0        # N / (rad/s)

    def thrust(self, vessel_kin: dict, t: float) -> float:
        phi = float(vessel_kin["phi"])
        phi_dot = float(vessel_kin["phi_dot"])
        return -self.K_phi * phi - self.K_phidot * phi_dot


# =====================================================================
# Tank
# =====================================================================


@dataclass
class RDTUtubeConfig(OpenUtubeConfig):
    """Geometry / fluid / actuator parameters for an RDT-active U-tube.

    Inherits all fields of :class:`OpenUtubeConfig` and adds the
    actuator amplitude limit. The duct cross-section used to map
    actuator force to flow velocity is computed from
    ``utube_duct_height * tank_thickness`` (the same A_duct that
    appears in the Bertram self-coefficients).
    """

    F_max: float = 200_000.0     # actuator amplitude limit, N. Default
                                 # ~210 kN ~ bow-tunnel-thruster class.


class RDTUtubeTank(OpenUtubeTank):
    """U-tube tank with a rim-driven thruster commanding duct flow.

    Subclasses :class:`OpenUtubeTank` -- inherits coefficient
    computation, hull-side force formulae, saturation handling. Adds
    an actuator-force term to the tau-EOM RHS supplied by an
    :class:`AbstractRDTController`.

    The hull-side ``forces()`` dictionary is unchanged in *form* from
    the passive open-top tank: ``M_roll = a_phi * tau_ddot
    + c_phi * tau``. The active drive shows up only through its effect
    on the fluid state.

    Parameters
    ----------
    config
        :class:`RDTUtubeConfig`.
    controller
        Implementation of :class:`AbstractRDTController`. Use
        :class:`InverseDynamicsRDTController` for the ideal-active
        baseline (assumes perfect wave-moment knowledge);
        :class:`StateFeedbackRDTController` for an honest signal-only
        controller.

    Diagnostics
    -----------
    The most recent commanded and applied (post-saturation) thrust
    are stored on the instance as ``self.last_F_cmd`` and
    ``self.last_F_applied`` after each :meth:`step_rk4` call -- handy
    for plotting and energy bookkeeping in examples / tests.
    """

    def __init__(
        self,
        config: RDTUtubeConfig,
        controller: AbstractRDTController,
        tau0: float = 0.0,
        tau_dot0: float = 0.0,
    ):
        super().__init__(config, tau0=tau0, tau_dot0=tau_dot0)
        self.config: RDTUtubeConfig = config        # narrow type for IDEs
        self.controller = controller
        # Pre-compute the actuator gain into the tau-EOM RHS:
        #   F_tau = F_RDT * (A_res / A_duct) * (W/2)
        c = config
        A_res = c.resevoir_duct_width * c.tank_thickness
        A_duct = c.utube_duct_height * c.tank_thickness
        if A_duct <= 0.0:
            raise ValueError("A_duct must be positive (utube_duct_height * tank_thickness).")
        self._rdt_gain = (A_res / A_duct) * (self.tank_width / 2.0)

        # If controller supports it, give it a back-reference (only the
        # inverse-dynamics baseline does -- it needs the tank coeffs and
        # state to invert the dynamics).
        if hasattr(self.controller, "attach_tank"):
            self.controller.attach_tank(self)

        self.last_F_cmd: float = 0.0
        self.last_F_applied: float = 0.0

    # ------------------------------------------------------------------ EOM

    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        """Tau-EOM with the controller's commanded thrust applied.

        Identical to :meth:`OpenUtubeTank.derivatives` except for the
        added ``+F_tau`` term on the RHS, where
        ``F_tau = clip(F_command, -F_max, +F_max) * gain`` and
        ``gain = (A_res / A_duct) * (W/2)``.
        """
        tau, tau_dot = state[0], state[1]
        phi = vessel_kin["phi"]
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        v_dot = vessel_kin.get("v_dot", 0.0)
        r_dot = vessel_kin.get("r_dot", 0.0)

        damping = (self.b_tau + self.b_quad * abs(tau_dot)) * tau_dot

        # Controller produces a force command using the *current* (most
        # recent) tank state, not the RK4-substep state. This is a
        # deliberate ZOH within the macro time-step: the actuator force
        # is held constant across the substeps. Otherwise the inverse-
        # dynamics controller would feed back on intermediate substep
        # tau values and produce a closed-form algebraic loop with the
        # passive coefficients.
        # (Honest realism: real RDT drives have ~kHz update rates,
        # comfortably faster than wave-frequency dynamics, so ZOH at
        # the macro dt ~ 0.05 s is conservative.)
        F_cmd = self._cached_F_cmd
        F_applied = float(np.clip(F_cmd, -self.config.F_max, +self.config.F_max))

        rhs = (
            +self.a_phi * phi_ddot
            + self.c_phi * phi
            - self.a_y * v_dot
            - self.a_psi * r_dot
            + F_applied * self._rdt_gain
        ) - damping - self.c_tau * tau

        tau_ddot = rhs / self.a_tau
        return np.array([tau_dot, tau_ddot])

    def step_rk4(self, vessel_kin_func, t, dt):
        # Compute the controller command once at the start of the macro
        # step, using the start-of-step vessel kinematics. Hold ZOH
        # across all four RK4 substeps (see derivatives() docstring).
        kin0 = vessel_kin_func(t)
        self._cached_F_cmd = float(self.controller.thrust(kin0, t))
        try:
            super().step_rk4(vessel_kin_func, t, dt)
        finally:
            self.last_F_cmd = self._cached_F_cmd
            self.last_F_applied = float(np.clip(
                self._cached_F_cmd,
                -self.config.F_max, +self.config.F_max,
            ))

    # ZOH cache (cleared / overwritten each step_rk4)
    _cached_F_cmd: float = 0.0
