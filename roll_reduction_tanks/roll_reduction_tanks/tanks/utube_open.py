"""Open-top passive U-tube anti-roll tank.

Lagrangian-derived model for a U-tube tank coupled to a 1-DOF roll vessel.
The cross-coupling and self coefficients follow Holden, Perez & Fossen
(2011), *A Lagrangian approach to nonlinear modeling of anti-roll tanks*,
Ocean Engineering 38, 341-359 — a derivation that is experimentally
validated against 44 model-scale tests. See ``docs/utube_derivation.md``
Part I for the mapping from Holden's variables to ours.

Coordinate convention
---------------------
Body frame, z-down (matches brucon and pdstrip / SNAME): x fore, y
starboard, z down. The roll axis is taken at the centre of floatation
``Cf`` (≈ waterline). All vertical levers are referenced to **the
waterline**, positive **downwards**:

* ``duct_below_waterline > 0``  ⇒ duct is *below* the waterline
  (e.g. inside the hull, near the keel).
* ``duct_below_waterline < 0``  ⇒ duct is *above* the waterline
  (e.g. on the deck or in the superstructure).

This is a deliberate departure from brucon's variable name
``utube_datum_to_cog`` (whose datum reference was ambiguous and led to a
hidden ``KG``-sized bias in the cross-coupling coefficient when the
underlying derivation references ``Cf``, not ``COG``). The user is
expected to supply the geometric distance from the waterline directly.

Geometry
--------
Two vertical reservoirs of height ``tank_height`` and horizontal width
``resevoir_duct_width``, joined at the bottom by a duct of height
``utube_duct_height`` and length ``utube_duct_width``. The fluid has
undisturbed free surface ``undisturbed_fluid_height`` above the duct
datum. ``tank_thickness`` is the lateral (along-ship) extent.
``tank_to_xcog`` is the lateral offset of the tank centroid from the
ship centerline.

Sign convention for the fluid coordinate ``tau``: ``tau > 0`` means the
*starboard* (positive y) reservoir sits *higher*. Equivalently, fluid
has flowed from port to starboard. In a beam sea that rolls the vessel
starboard-down (``phi > 0``), gravity in the body frame pushes fluid
toward starboard so ``tau`` quasi-statically follows ``phi`` with the
**same sign** — the classic destabilising free-surface effect at low
frequency, which becomes stabilising near tank resonance through the
dynamic phase shift.

Equations of motion
-------------------
With ``W = w_d + b_r`` the centreline distance between reservoirs and
``Q = rho * b_r * W^2 * t / 2``, the linearised tank EOM is

    a_tau * tau_ddot + b_tau * tau_dot + c_tau * tau
       = +a_phi * phi_ddot + c_phi * phi
         - a_y * v_dot      - a_psi * r_dot

(``v_dot``, ``r_dot`` are sway/yaw acceleration; both zero in 1-DOF
roll). The moment the tank exerts on the hull is

    M_tank_on_phi  =  +a_phi * tau_ddot  +  c_phi * tau

with cross-coupling coefficients

    a_phi  =  Q * (z_d + h_0)
    c_phi  =  Q * g

where ``z_d = duct_below_waterline``. The self coefficients are the
classical Bertram (4.123) values:

    a_tau  =  Q * b_r * (W/(2 h_d) + h_0 / b_r)
    b_tau  =  Q * mu * b_r * (W/(2 h_d^2) + h_0 / b_r)
    c_tau  =  Q * g

Tank natural frequency: ``omega_tau = sqrt(c_tau / a_tau) =
sqrt(g / (h_0 + W b_r / (2 h_d)))`` (Bertram 4.123 / Holden eq. 41).

Comparison to ``brucon::simulator::TankRollModel``
--------------------------------------------------
Brucon's coefficient *formula* ``a_phi = Q*(z_d + h_0)`` is correct in
form (matches Holden eq. 14), but brucon's ``z_d`` variable
(``utube_datum_to_cog``) is *named* COG-referenced while the underlying
derivation references the roll axis Cf (≈ waterline). When ``KG != 0``
this mismatch produces a hidden bias of magnitude ``Q*KG`` in
``a_phi``. We resolve this by renaming and re-referencing the field to
``duct_below_waterline``.

Brucon also has two unambiguous bugs we do **not** reproduce here:

1. Brucon writes ``-c_phi*phi`` on the tank-EOM RHS. The Lagrangian
   gives ``+c_phi*phi``. (With brucon's sign, the static fluid response
   to a hull tilt would be in the wrong direction — fluid running to
   the windward side under gravity.)
2. Brucon's ``RollMoment`` (vessel side) uses ``a_tau`` (the
   self-inertia of the tank fluid) instead of ``a_phi`` (the
   cross-inertia coefficient required by Lagrangian reciprocity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AbstractTank


@dataclass
class OpenUtubeConfig:
    """Geometry / fluid parameters for an open-top passive U-tube.

    Vertical reference is the **waterline**; positive levers point
    downward (z-down body frame).
    """

    duct_below_waterline: float     # z_d, m. Positive = duct below WL.
    undisturbed_fluid_height: float # h_0, m, undisturbed free-surface height above duct datum.
    utube_duct_height: float        # h_d, duct vertical extent, m.
    resevoir_duct_width: float      # b_r, reservoir horizontal width, m  (kept brucon's spelling)
    utube_duct_width: float         # w_d, duct horizontal length, m.
    tank_thickness: float           # t, lateral (along-ship) extent, m.
    tank_to_xcog: float             # x_T, lateral offset of tank from ship COG, m
    tank_wall_friction_coef: float  # mu, dimensionless wall-friction coefficient (linear damping).
    tank_height: float              # H_t, total reservoir height, m.

    # --- nonlinear damping (Holden, Perez & Fossen 2011 eq. 22) -----------
    # Holden's tank-EOM damping function in his q_2 (port-reservoir level,
    # m) coordinate is
    #     d_xi(q̇_2)  =  d_{2,l}  +  d_{2,n} * |q̇_2|       (eq. 22)
    # i.e. linear-plus-quadratic-in-velocity. Damping force = d_xi · q̇_2.
    # In our tau coordinate (q_2 = -(W/2) tau), the equivalent damping term
    # added to the LHS of the tau-EOM is
    #     b_lin_tau  * tau_dot  +  b_quad_tau * |tau_dot| * tau_dot
    # with
    #     b_lin_tau  =  d_{2,l}              (= our existing b_tau)
    #     b_quad_tau =  (W/2) * d_{2,n}      (NEW)
    #
    # The nonlinear term is what makes the difference between Holden's
    # linear (Ll) and nonlinear (L, eL) models: per Holden tab. 1/2, the
    # linear-only fit overestimates d_{2,l} by ~3x to compensate for the
    # missing quadratic term. Ignoring it (b_quad = 0) is fine for small
    # tau (small fluid velocities) but underestimates dissipation as
    # amplitude grows.
    #
    # `quad_damping_coef` here is b_quad_tau directly (units kg/(rad/s)
    # in our tau-space; physically, the slope of damping force per unit
    # |tau_dot|^2). Defaults to 0 (linear-only, matches our previous
    # behaviour); set to a positive value to turn on the nonlinear
    # damping. Use `OpenUtubeTank.estimate_quad_damping_from_loss(K)`
    # to get a physical first guess from a discharge-loss coefficient.
    quad_damping_coef: float = 0.0

    rho: float = 1025.0
    g: float = 9.81


class OpenUtubeTank(AbstractTank):
    """Open-top passive U-tube anti-roll tank.

    State vector: ``[tau, tau_dot]``  (rad, rad/s).
    """

    def __init__(self, config: OpenUtubeConfig, tau0: float = 0.0, tau_dot0: float = 0.0):
        self.config = config
        self._compute_coefficients()
        self.state = np.array([tau0, tau_dot0], dtype=float)
        # Cached most-recent acceleration for forces() - used by the hull moment.
        self._last_tau_ddot: float = 0.0

    # ------------------------------------------------------------------ derived geometry / coefficients

    def _compute_coefficients(self) -> None:
        c = self.config
        rho, g = c.rho, c.g

        if c.resevoir_duct_width <= 0 or c.utube_duct_height <= 0:
            raise ValueError(
                "resevoir_duct_width and utube_duct_height must be strictly positive."
            )
        # Centreline distance between the two reservoirs = duct length + leg width.
        # Brucon names this `tank_width` and uses it in Q.
        W = c.utube_duct_width + c.resevoir_duct_width
        self.tank_width = W

        # FlowRateTank Q = rho * b_r * W^2 * t / 2
        Q = rho * c.resevoir_duct_width * W * W * c.tank_thickness / 2.0
        self.Q = Q

        # Cross-coupling coefficients (Holden, Perez & Fossen 2011 eq. 14).
        self.a_y    = -Q
        self.a_phi  =  Q * (c.duct_below_waterline + c.undisturbed_fluid_height)
        self.c_phi  =  Q * g
        self.a_psi  = -Q * c.tank_to_xcog

        # Self coefficients (Bertram 4.123).
        self.a_tau = Q * c.resevoir_duct_width * (
            W / (2.0 * c.utube_duct_height)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        self.b_tau = Q * c.tank_wall_friction_coef * c.resevoir_duct_width * (
            W / (2.0 * c.utube_duct_height ** 2)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        self.c_tau = Q * g

        # Quadratic-damping coefficient in tau-space (Holden eq. 22, mapped
        # via q_2 = -(W/2) tau): b_quad = (W/2) * d_{2,n}.
        self.b_quad = float(c.quad_damping_coef)

        # Maximum mechanical tilt before fluid hits the top of the reservoir.
        self.tau_max = (
            2.0 * (c.tank_height - c.undisturbed_fluid_height) / W
        )

    # ------------------------------------------------------------------ helpers

    def estimate_quad_damping_from_loss(self, K_loss: float) -> float:
        """Estimate ``b_quad`` from a discharge / form-loss coefficient.

        For fluid in the duct moving at velocity ``v_d``, the head loss
        is ``dh = K_loss v_d^2 / (2 g)`` and the corresponding pressure
        force on the duct cross-section ``A_d = h_d * t`` is
        ``F = K_loss rho A_d v_d^2 / 2`` (Idelchik 1986).

        Continuity in the duct gives ``v_d = (b_r / h_d) * (W/2) |tau_dot|``.
        Equating the dissipation rate ``F * v_d`` to ``b_quad |tau_dot|^3``
        yields::

            b_quad  =  K_loss * rho * t * (W * b_r)^3 / (16 * h_d^2)

        ``K_loss`` is dimensionless. Typical values: ~0.5 for a
        well-rounded transition, ~1.0 for a sharp 90 deg
        duct-reservoir junction, ~1.5 for very sharp / contracted
        junctions. Returns the value in kg m^2 (rad-space) but does
        *not* mutate ``self.b_quad`` -- feed it back through
        ``OpenUtubeConfig.quad_damping_coef`` if you want it active.
        """
        c = self.config
        if c.utube_duct_height <= 0:
            raise ValueError("utube_duct_height must be positive.")
        W = self.tank_width
        return float(
            K_loss * c.rho * c.tank_thickness * (W * c.resevoir_duct_width) ** 3
            / (16.0 * c.utube_duct_height ** 2)
        )

    # ------------------------------------------------------------------ analytic properties

    @property
    def natural_frequency(self) -> float:
        """Undamped natural frequency of the U-tube fluid, rad/s."""
        return float(np.sqrt(self.c_tau / self.a_tau))

    @property
    def natural_period(self) -> float:
        return 2 * np.pi / self.natural_frequency

    @property
    def damping_ratio(self) -> float:
        return self.b_tau / (2.0 * np.sqrt(self.c_tau * self.a_tau))

    # ------------------------------------------------------------------ EOM

    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        """`d/dt [tau, tau_dot]`.

        Tank EOM (Holden, Perez & Fossen 2011 eq. 41, with eq. 22
        nonlinear damping mapped to tau)::

            a_tau tau_ddot
              + (b_tau + b_quad |tau_dot|) tau_dot
              + c_tau tau
                = +a_phi phi_ddot + c_phi phi
                  - a_y v_dot     - a_psi r_dot

        The ``b_quad |tau_dot| tau_dot`` term is the form-drag /
        vortex-shedding loss at the duct-reservoir junction (Holden
        eq. 22). Set ``quad_damping_coef = 0`` to recover the pure
        Lloyd / Holden-Ll linear model.
        """
        tau, tau_dot = state[0], state[1]
        phi = vessel_kin["phi"]
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        v_dot = vessel_kin.get("v_dot", 0.0)
        r_dot = vessel_kin.get("r_dot", 0.0)

        damping = (self.b_tau + self.b_quad * abs(tau_dot)) * tau_dot

        rhs = (
            +self.a_phi * phi_ddot
            + self.c_phi * phi
            - self.a_y * v_dot
            - self.a_psi * r_dot
        ) - damping - self.c_tau * tau

        tau_ddot = rhs / self.a_tau
        return np.array([tau_dot, tau_ddot])

    def step_rk4(self, vessel_kin_func, t, dt):
        super().step_rk4(vessel_kin_func, t, dt)
        # Clamp angular displacement to physical limit; reset velocity if hit.
        if abs(self.state[0]) > self.tau_max:
            self.state[0] = np.clip(self.state[0], -self.tau_max, self.tau_max)
            # Don't kill velocity entirely -- just prevent overshoot. Set
            # velocity to zero only if it is pushing further into the wall.
            if (self.state[0] > 0 and self.state[1] > 0) or (
                self.state[0] < 0 and self.state[1] < 0
            ):
                self.state[1] = 0.0
        # Cache most recent acceleration for forces().
        deriv = self.derivatives(self.state, vessel_kin_func(t + dt), t + dt)
        self._last_tau_ddot = float(deriv[1])

    # ------------------------------------------------------------------ vessel-side forces

    def forces(self, vessel_kin: dict) -> dict:
        """Forces / moments the tank applies to the vessel.

        Roll moment (Holden eq. 36/37, mapped via tau = -2 xi / W)::

            M_roll  =  +a_phi * tau_ddot  +  c_phi * tau

        - The ``+c_phi * tau`` static term is the destabilising
          free-surface effect: when fluid sits on the starboard side
          (``tau > 0``) it pulls the hull further to starboard.
        - The ``+a_phi * tau_ddot`` dynamic term is the resonant
          absorber action: at the tank natural frequency, ``tau_ddot``
          is in phase opposition with hull roll, so this term provides
          the anti-roll moment.

        Note (vs brucon): brucon uses ``a_tau`` (self-inertia) on the
        vessel side, breaking Lagrangian reciprocity. We use ``a_phi``
        (cross-inertia) here as required by Holden's derivation and by
        brucon's own (correct) ``TankAngleSemiCoupled`` method.
        """
        tau = self.state[0]
        tau_dot = self.state[1]
        tau_ddot = self._last_tau_ddot

        return {
            "surge": 0.0,
            "sway": -self.a_y * tau_dot,                      # = +Q * tau_dot
            "roll":  self.a_phi * tau_ddot + self.c_phi * tau,
            "yaw":  -self.a_psi * tau_ddot,
        }
