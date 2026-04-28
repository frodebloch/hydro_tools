"""Free-surface (Lloyd-style) anti-roll tank.

A simple rectangular partially-filled tank acting as an SDOF sloshing
oscillator. Following Lloyd (1989) and Bhattacharyya (1978), the lowest
sloshing mode of a rectangular tank with along-beam length ``L`` and
fluid depth ``h`` has natural frequency::

    omega_n^2 = (pi * g / L) * tanh(pi * h / L)

We treat the lowest sloshing mode as an equivalent tuned-mass damper:
the modal coordinate ``q`` (m) is the lateral displacement of the
sloshing equivalent mass ``m_eq``, mounted at lever ``h_arm`` above the
ship COG.

Linearised EOM (Lagrangian derivation, including gravity cross-coupling)
-----------------------------------------------------------------------
With the sloshing mass at body-frame lever ``h_arm = z_tank - z_cog``
above the COG (z-up convention), the linearised tank EOM is
::

    m_eq * q_ddot + b_eq * q_dot + k_eq * q
        =  -m_eq * h_arm * phi_ddot  +  m_eq * g * phi

* The ``-m_eq h_arm phi_ddot`` term is the inertial pseudo-force from
  the rotating frame (the mass tries to lag behind the rolling hull).
* The ``+m_eq g phi`` term is the gravity cross-coupling: when the
  hull is rolled by a steady ``phi``, gravity in the body frame has a
  lateral component ``g sin(phi) ~ g phi`` that drives the sloshing
  mode just like a horizontal pseudo-acceleration.

The roll moment on the hull then has three contributions::

    M_roll  =  -m_eq * h_arm * q_ddot                # dynamic TMD reaction
              +  m_eq * g * q                        # gravity moment of displaced mass
              +  Delta_c44_extra * phi               # extra static surface-tilt loss

* ``-m_eq h_arm q_ddot`` is the dynamic-reaction TMD term: peaks when
  ``q_ddot`` is in quadrature with ``phi``, providing useful
  cancellation around resonance.
* ``+m_eq g q`` is the static gravity moment from the displaced
  sloshing mass. At zero frequency the equilibrium ``q_ss = g phi /
  omega_n^2`` makes this contribution destabilising (positive feedback
  on phi), accounting for the **lowest-mode share** of the
  free-surface effect on GM.
* ``-Delta_c44_extra * phi`` is a *static* destabilising stiffness that
  recovers the remaining (~93 % for shallow tanks) of the classical
  hydrostatic free-surface GM loss; see below.

Static free-surface effect on GM
--------------------------------
The classical (multi-mode, hydrostatic-tilt) destabilising stiffness from
a free fluid surface is
::

    Delta_c44_classical = rho_t * g * W * L^3 / 12

equivalent to GM loss ``Delta_GM_classical = rho_t W L^3 / (12 rho_s
volume)``.

The lowest sloshing mode in our model contributes only
::

    Delta_c44_dyn = m_eq * g^2 / omega_n^2
                  = (8 / pi^4) * rho_t * g * W * L^2 * h
                  = (96 h / (pi^4 L)) * Delta_c44_classical

i.e. only ~7 % of the classical value for our shallow geometry
(``h/L = 0.07``). The remaining ~93 % is **not** in higher sloshing
modes -- those are negligible at vessel-roll frequencies because their
modal mass falls as ``1/n^3`` and their natural periods (~4 s, ~2.6 s,
...) are far from the wave/vessel band. Instead, the missing 93 % is
the **rigid-surface tilt** contribution: even at zero frequency the
free surface remains horizontal in the inertial frame, displacing the
fluid centre of gravity laterally and producing a destabilising hull
moment with no associated sloshing dynamics.

We add this as a static stiffness::

    Delta_c44_extra = Delta_c44_classical - Delta_c44_dyn

so that, at the static (zero-frequency) limit, the total free-surface
effect on GM matches Bhattacharyya / classical naval architecture.

Run-dry / large-amplitude limit
-------------------------------
The model is linear and assumes the free surface stays well-defined,
i.e. ``|eta_wall| << h``. The lowest-mode wall elevation is related to
the modal coordinate ``q`` by energy matching::

    beta_1 = sqrt(k_eq / K_1) * q,   K_1 = rho_t g W L / 2

When ``|beta_1|`` approaches ``h`` the tank wall runs dry on one side
and the linear model breaks down: real shallow tanks develop hydraulic
jumps, traveling bores and large viscous losses (Faltinsen & Timokha
2009 §4-6). We expose ``wall_elevation()``, the related ``fill_ratio``,
and emit a one-shot ``UserWarning`` at the first time the fill ratio
exceeds ``warn_fill_ratio`` (default 0.5). The user is then expected to
either accept the model's reduced fidelity or reduce wave amplitude /
add more damping.

Faltinsen (1990) §5.4 sizing rule
---------------------------------
Faltinsen recommends sizing the tank so that the *total* free-surface
GM loss is in the range::

    Delta_GM_classical / GM   in [0.15, 0.30]

For the CSOV (rho_s nabla = 1.11e7 kg, GM = 3.0 m), with the along-beam
length fixed at ``L = 21.7 m`` (constrained by the 22.4 m beam and by
the tuning requirement ``omega_n = vessel_omega_p``), this rule gives
a along-ship width range::

    W_min = 12 * 0.15 * GM * rho_s * volume / (rho_t * L^3)  ~= 5.7 m
    W_max = 12 * 0.30 * GM * rho_s * volume / (rho_t * L^3)  ~= 11.5 m

References
----------
* A.R.J.M. Lloyd, *Seakeeping* (1989), chapter on anti-roll tanks.
* O.M. Faltinsen, *Sea Loads on Ships and Offshore Structures* (1990),
  Section 5.4.
* O.M. Faltinsen and A.N. Timokha, *Sloshing* (2009), §4-6 (nonlinear
  shallow-water sloshing).
* R. Bhattacharyya, *Dynamics of Marine Vehicles* (1978), Section 8.7.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .base import AbstractTank


@dataclass
class FreeSurfaceConfig:
    """Geometry / fluid parameters for a rectangular free-surface tank.

    Parameters
    ----------
    length
        Along-beam length of the tank, m. Sets the sloshing wavelength.
    width
        Along-ship dimension, m. (Used for fluid mass only.)
    fluid_depth
        Undisturbed fluid depth, m.
    z_tank
        Vertical position of the undisturbed free surface in the
        ship-fixed frame, m. Positive *up*; ``z_tank - z_cog`` is the
        moment arm.
    z_cog
        Vertical position of the ship COG (same convention as
        ``z_tank``).
    damping_ratio
        Linear modal damping ratio, dimensionless. Typical 0.05--0.15
        for sloshing (much higher than U-tube fluid friction because
        of strong wave breaking and viscous boundary-layer dissipation
        at the side walls).
    rho
        Fluid density, kg/m^3.
    g
        Gravity, m/s^2.
    warn_fill_ratio
        Wall-elevation fill ratio ``|beta_1| / h`` at which a one-shot
        ``UserWarning`` is emitted, signalling that the linear model is
        being pushed past its valid range. Set to ``inf`` to suppress.
    """

    length: float
    width: float
    fluid_depth: float
    z_tank: float
    z_cog: float
    damping_ratio: float = 0.10
    rho: float = 1025.0
    g: float = 9.81
    warn_fill_ratio: float = 0.5


class FreeSurfaceTank(AbstractTank):
    """Lloyd-style SDOF free-surface anti-roll tank.

    State vector: ``[q, q_dot]`` (m, m/s) -- lateral displacement of the
    equivalent sloshing mass.
    """

    def __init__(self, config: FreeSurfaceConfig, q0: float = 0.0, q_dot0: float = 0.0):
        self.config = config
        self._compute_coefficients()
        self.state = np.array([q0, q_dot0], dtype=float)
        self._last_q_ddot: float = 0.0
        self._fill_warning_issued: bool = False

    # ------------------------------------------------------------------ derived

    def _compute_coefficients(self) -> None:
        c = self.config
        if c.length <= 0 or c.fluid_depth <= 0:
            raise ValueError("length and fluid_depth must be strictly positive.")

        # Total fluid mass.
        m_fluid = c.rho * c.length * c.width * c.fluid_depth
        self.m_fluid = m_fluid

        # Linear-mode natural frequency.
        kL = np.pi / c.length
        self.omega_n = float(np.sqrt(c.g * kL * np.tanh(kL * c.fluid_depth)))

        # Equivalent sloshing mass (Faltinsen 1990, eq 5.46).
        self.m_eq = (8.0 / np.pi**3) * np.tanh(kL * c.fluid_depth) * m_fluid
        self.k_eq = self.m_eq * self.omega_n**2
        self.b_eq = 2.0 * c.damping_ratio * np.sqrt(self.k_eq * self.m_eq)

        # Lever arm (free surface above COG).
        self.h_arm = c.z_tank - c.z_cog

        # Wall-elevation mapping: beta_1 = beta_over_q * q, derived by
        # matching modal PE  (1/2) k_eq q^2  to gravity PE of the
        # cosine surface elevation  (1/2)(rho g W L / 2) beta_1^2.
        K_1 = c.rho * c.g * c.width * c.length / 2.0
        self.beta_over_q = float(np.sqrt(self.k_eq / K_1))

        # Static surface-tilt destabilising stiffness, the part NOT
        # captured by the lowest dynamic mode (i.e. classical
        # hydrostatic free-surface effect minus low-frequency limit
        # of the dynamic mode).
        dc44_classical = c.rho * c.g * c.width * c.length**3 / 12.0
        dc44_dynamic = self.m_eq * c.g**2 / self.omega_n**2
        self.dc44_extra = float(dc44_classical - dc44_dynamic)
        self.dc44_classical = float(dc44_classical)
        self.dc44_dynamic = float(dc44_dynamic)

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

    def wall_elevation(self) -> float:
        """Lowest-mode wall free-surface elevation ``beta_1`` (m,
        relative to the undisturbed level)."""
        return self.beta_over_q * float(self.state[0])

    def fill_ratio(self) -> float:
        """Absolute fill ratio ``|beta_1| / h``. Reaches 1 at run-dry."""
        return abs(self.wall_elevation()) / self.config.fluid_depth

    # ------------------------------------------------------------------ EOM

    def derivatives(self, state: np.ndarray, vessel_kin: dict, t: float) -> np.ndarray:
        q, q_dot = state[0], state[1]
        phi = vessel_kin.get("phi", 0.0)
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        # Linearised lowest-mode sloshing EOM with gravity cross-coupling:
        #   m_eq q_ddot + b_eq q_dot + k_eq q
        #       = -m_eq h_arm phi_ddot + m_eq g phi
        rhs = (
            -self.m_eq * self.h_arm * phi_ddot
            + self.m_eq * self.config.g * phi
            - (self.b_eq * q_dot + self.k_eq * q)
        )
        q_ddot = rhs / self.m_eq
        return np.array([q_dot, q_ddot])

    def step_rk4(self, vessel_kin_func, t, dt):
        super().step_rk4(vessel_kin_func, t, dt)
        deriv = self.derivatives(self.state, vessel_kin_func(t + dt), t + dt)
        self._last_q_ddot = float(deriv[1])
        self._maybe_warn_fill()

    def _maybe_warn_fill(self) -> None:
        if self._fill_warning_issued:
            return
        if self.fill_ratio() > self.config.warn_fill_ratio:
            warnings.warn(
                f"FreeSurfaceTank: wall fill ratio |beta_1|/h = "
                f"{self.fill_ratio():.2f} exceeded the warn threshold "
                f"{self.config.warn_fill_ratio:.2f}. The linear model is "
                f"being pushed past its valid range; near run-dry "
                f"(ratio -> 1) shallow-water nonlinearities, hydraulic "
                f"jumps and traveling bores invalidate this SDOF model.",
                UserWarning,
                stacklevel=2,
            )
            self._fill_warning_issued = True

    # ------------------------------------------------------------------ vessel-side

    def forces(self, vessel_kin: dict) -> dict:
        # Recompute q_ddot from current state and supplied vessel kinematics
        # rather than using the cached value from the previous step (avoids
        # a one-step lag in the explicit-Jacobi feedback path).
        q_ddot = float(self.derivatives(self.state, vessel_kin, 0.0)[1])
        q = float(self.state[0])
        phi = vessel_kin.get("phi", 0.0)
        # Three contributions on the hull:
        #   -m_eq h_arm q_ddot : dynamic TMD-style reaction moment.
        #   +m_eq g q          : static gravity moment from the
        #                        displaced sloshing mass (lowest-mode
        #                        share of the free-surface GM loss; at
        #                        zero frequency q -> g phi / omega_n^2).
        #   +dc44_extra * phi  : extra static destabilising stiffness
        #                        from the rigid-surface-tilt component
        #                        not captured by the lowest dynamic
        #                        mode (recovers classical multi-mode
        #                        hydrostatic free-surface GM loss in
        #                        the static limit). Positive, i.e.
        #                        adds to phi -- destabilising, like
        #                        the +m_eq g q term.
        M_roll = (
            -self.m_eq * self.h_arm * q_ddot
            + self.m_eq * self.config.g * q
            + self.dc44_extra * phi
        )
        return {"surge": 0.0, "sway": 0.0, "roll": M_roll, "yaw": 0.0}
