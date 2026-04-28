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


# ---------------------------------------------------------------------------- helpers


def _depth_for_omega(omega_target: float, L: float, g: float = 9.81) -> float:
    """Invert ``omega^2 = (pi g / L) tanh(pi h / L)`` for ``h`` (m).

    Returns NaN if ``omega_target`` exceeds the deep-water limit
    ``sqrt(pi g / L)`` (no real positive depth solves the relation).
    """
    arg = omega_target**2 * L / (np.pi * g)
    if arg <= 0.0 or arg >= 1.0:
        return float("nan")
    return float(L / np.pi * np.arctanh(arg))


def tune_self_consistent(
    *,
    length: float,
    width: float,
    z_tank: float,
    z_cog: float,
    damping_ratio: float,
    vessel_c44: float,
    vessel_inertia_total: float,
    rho: float = 1025.0,
    g: float = 9.81,
    warn_fill_ratio: float = 0.5,
    rel_tol: float = 1e-7,
    max_iter: int = 100,
) -> tuple[FreeSurfaceConfig, dict]:
    """Self-consistently tune fluid depth ``h`` for a fixed-length tank.

    Physics. A free-surface tank's static surface-tilt stiffness
    ``dc44_extra`` enters the hull moment as ``+dc44_extra * phi`` and
    therefore reduces the *effective* roll restoring stiffness from
    ``c44`` to ``c44_eff = c44 - dc44_extra``. The vessel's effective
    natural frequency is then ``omega_eff = sqrt(c44_eff / I_tot)``,
    which is lower than the bare ``omega_n``. Tuning the tank to the
    bare vessel period would leave it mistuned in operation.

    Why fix L (not W). The along-beam tank length ``L`` is the
    *geometric* constraint: it cannot exceed the ship beam. The
    sloshing dispersion ``omega^2 = (pi g / L) tanh(pi h / L)`` lets
    us tune to any frequency below the deep-water limit
    ``sqrt(pi g / L)`` by adjusting the *depth* ``h``. With ``L``
    fixed at (or near) the beam, this helper iterates ``h`` such that
    ``omega_n_tank == omega_eff(W, L, h)``. ``W`` is held fixed and
    chosen by the user from the Faltinsen 1990 §5.4 GM-loss budget.

    Iteration. ``h_0`` from the dispersion at ``omega_bare``, then::

        repeat:  dc44_extra(L, W, h) -> c44_eff -> omega_eff -> h_new

    Converges in ~10-30 iterations for sane inputs.

    Parameters
    ----------
    length
        Along-beam tank length ``L`` (m). Held fixed; should be
        ``<= beam``.
    width
        Along-ship tank width ``W`` (m). Held fixed.
    z_tank, z_cog
        Vertical positions (z-up convention).
    damping_ratio
        Tank modal damping ratio.
    vessel_c44
        Bare-vessel hydrostatic restoring stiffness (N*m/rad).
    vessel_inertia_total
        ``I44 + a44`` (kg*m^2).
    rho, g
        Fluid density and gravity (defaults: sea water, standard gravity).
    warn_fill_ratio
        Forwarded to :class:`FreeSurfaceConfig`.
    rel_tol, max_iter
        Convergence tolerance (relative change in h) and iteration cap.

    Returns
    -------
    cfg : FreeSurfaceConfig
        Self-consistently tuned configuration.
    info : dict
        Diagnostics: ``fluid_depth, omega_eff, omega_bare, dc44_extra,
        gm_loss_ratio, iterations, converged``.

    Raises
    ------
    RuntimeError
        If ``dc44_extra`` would exceed ``vessel_c44`` (tank too wide,
        free-surface effect would capsize the bare hull) or if no real
        depth satisfies the dispersion at the required frequency.
    """
    if vessel_c44 <= 0 or vessel_inertia_total <= 0:
        raise ValueError("vessel_c44 and vessel_inertia_total must be positive.")
    if length <= 0 or width <= 0:
        raise ValueError("length and width must be strictly positive.")
    omega_bare = float(np.sqrt(vessel_c44 / vessel_inertia_total))
    omega_eff = omega_bare
    h = _depth_for_omega(omega_bare, length, g)
    if not np.isfinite(h):
        raise RuntimeError(
            f"No real depth solves the dispersion at omega_bare={omega_bare:.3f} "
            f"rad/s with L={length:.2f} m (deep-water limit "
            f"omega_max={np.sqrt(np.pi * g / length):.3f} rad/s). Increase L."
        )
    converged = False
    dc44_extra = 0.0
    for it in range(1, max_iter + 1):
        # dc44_dyn = m_eq * g^2 / omega_n^2 simplifies to (8/pi^4) rho g W L^2 h.
        dc44_dyn = (8.0 / np.pi**4) * rho * g * width * length**2 * h
        dc44_classical = rho * g * width * length**3 / 12.0
        dc44_extra = dc44_classical - dc44_dyn
        c44_eff = vessel_c44 - dc44_extra
        if c44_eff <= 0:
            raise RuntimeError(
                f"Iteration {it}: dc44_extra={dc44_extra:.3e} exceeds vessel "
                f"c44={vessel_c44:.3e}; tank too wide (free-surface effect "
                f"would capsize the bare hull). Reduce W."
            )
        omega_eff_new = float(np.sqrt(c44_eff / vessel_inertia_total))
        h_new = _depth_for_omega(omega_eff_new, length, g)
        if not np.isfinite(h_new) or h_new <= 0:
            raise RuntimeError(
                f"Iteration {it}: required omega_eff={omega_eff_new:.3f} rad/s "
                f"exceeds deep-water limit at L={length:.2f} m. Increase L."
            )
        rel = abs(h_new - h) / h
        h, omega_eff = h_new, omega_eff_new
        if rel < rel_tol:
            converged = True
            break
    cfg = FreeSurfaceConfig(
        length=length, width=width, fluid_depth=h,
        z_tank=z_tank, z_cog=z_cog,
        damping_ratio=damping_ratio, rho=rho, g=g,
        warn_fill_ratio=warn_fill_ratio,
    )
    info = {
        "fluid_depth": h,
        "omega_eff": omega_eff,
        "omega_bare": omega_bare,
        "dc44_extra": float(dc44_extra),
        "gm_loss_ratio": float(dc44_extra / vessel_c44),
        "iterations": it,
        "converged": converged,
    }
    return cfg, info
