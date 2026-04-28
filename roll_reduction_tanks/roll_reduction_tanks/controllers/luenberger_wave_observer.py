"""Luenberger observer for wave-induced roll moment estimation.

Implements the Sælid-style augmented-plant observer described in
README sec. 4.z: the bare-vessel roll EOM is augmented with a 2nd-order
linear oscillator (the "wave resonator") at the encounter frequency
``omega_e`` plus a slow bias state, and a constant-gain Luenberger
correction is designed by pole placement on the augmented LTI plant.

State vector
------------
    x = [ phi, phi_dot, eta_1, eta_2, M_bias ]^T   (5 states)

with continuous-time dynamics::

    phi_dot     = phi_dot
    phi_ddot    = (1/I) * ( -b*phi_dot - c*phi
                            + eta_1 + M_bias + M_tank_known )
    eta_1_dot   = eta_2
    eta_2_dot   = -2*zeta_w*omega_e*eta_2 - omega_e**2 * eta_1
    M_bias_dot  = 0

The wave-moment estimate is read off the resonator output plus bias::

    M_wave_hat  =  eta_1  +  M_bias

The known tank moment ``M_tank_known(t)`` (whatever the active tank is
currently producing on the hull) is fed in as an exogenous input so the
observer does NOT mistakenly attribute it to the wave. This is the
single most important reason the architecture is causal and stable: the
controller commands ``F_RDT``, the tank produces a known ``M_tank``,
the observer sees it, and only the *unexplained* roll content is
folded into ``M_wave_hat``.

Measurement
-----------
By default ``y = phi`` (single-channel rate-gyro integration / MRU).
Optionally ``y = [phi, phi_dot]^T`` if both signals are available -- the
extra channel materially improves the observer's ability to track
``eta_2`` (and hence wave-moment phase).

Pole placement
--------------
``observer_poles`` must contain 5 desired closed-loop eigenvalues for
the matrix ``(A - L*C)``, supplied as complex-conjugate pairs (no
isolated complex poles). The recommended recipe (see README sec. 4.z):

  * 1 pair near ``-3 * omega_n_roll``  (vessel-state error decay)
  * 1 pair near ``-zeta_obs * omega_e ± j * omega_e * sqrt(1-zeta_obs^2)``
    with ``zeta_obs ~ 0.15-0.20`` (resonator correction)
  * 1 real pole near ``-0.05`` rad/s   (slow bias drift)

If ``observer_poles`` is None, the constructor builds them automatically
from ``omega_n_roll`` (computed from ``c, I``) and ``omega_e``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import place_poles


def _default_observer_poles(omega_n_roll: float, omega_e: float,
                            zeta_obs: float = 0.5) -> list[complex]:
    """Recommended 5 observer poles. See module docstring."""
    # Vessel-state error decay: critically-damped pair at 3*omega_n_roll
    p_vessel = -3.0 * omega_n_roll
    # Resonator-correction pair: 2-4x faster decay than open-loop
    # resonator (which has zeta_w ~ 0.05); same imaginary part.
    sigma = zeta_obs * omega_e
    omega_d = omega_e * np.sqrt(max(1.0 - zeta_obs ** 2, 1e-9))
    # Slow bias pole
    p_bias = -0.05
    return [
        complex(p_vessel, +0.5 * omega_n_roll),
        complex(p_vessel, -0.5 * omega_n_roll),
        complex(-sigma,   +omega_d),
        complex(-sigma,   -omega_d),
        complex(p_bias,    0.0),
    ]


@dataclass
class LuenbergerWaveObserverConfig:
    """Tuning parameters for the wave-moment observer."""

    I_total: float                     # total roll inertia incl. a44, kg*m^2
    b44: float                         # linear roll damping, N*m / (rad/s)
    c44: float                         # roll restoring stiffness, N*m / rad
    omega_e: float                     # encounter frequency, rad/s
    zeta_w: float = 0.05               # resonator damping (model side)
    zeta_obs: float = 0.5              # resonator damping (observer side, > zeta_w)
    measure_phi_dot: bool = False      # if True, y = [phi, phi_dot]^T
    observer_poles: Optional[Sequence[complex]] = None  # override


class LuenbergerWaveObserver:
    """Constant-gain Luenberger observer for the augmented plant."""

    def __init__(self, cfg: LuenbergerWaveObserverConfig):
        self.cfg = cfg
        I = cfg.I_total
        b = cfg.b44
        c = cfg.c44
        omega_e = cfg.omega_e

        # Continuous-time augmented plant:
        #   x_dot = A x + B_u * M_tank + B_w * 0    (no process noise channel)
        #   y     = C x
        A = np.zeros((5, 5))
        # Row 0: phi_dot
        A[0, 1] = 1.0
        # Row 1: phi_ddot (excluding M_tank input -> goes to B_u)
        A[1, 0] = -c / I
        A[1, 1] = -b / I
        A[1, 2] = +1.0 / I            # eta_1 contribution to phi_ddot
        A[1, 4] = +1.0 / I            # M_bias contribution to phi_ddot
        # Row 2: eta_1_dot = eta_2
        A[2, 3] = 1.0
        # Row 3: eta_2_dot = -2*zeta_w*omega_e*eta_2 - omega_e^2 * eta_1
        A[3, 2] = -omega_e ** 2
        A[3, 3] = -2.0 * cfg.zeta_w * omega_e
        # Row 4: M_bias_dot = 0  (already zero)

        B_u = np.zeros((5, 1))        # input: M_tank_known
        B_u[1, 0] = 1.0 / I

        if cfg.measure_phi_dot:
            C = np.zeros((2, 5))
            C[0, 0] = 1.0   # phi
            C[1, 1] = 1.0   # phi_dot
        else:
            C = np.zeros((1, 5))
            C[0, 0] = 1.0

        self.A = A
        self.B_u = B_u
        self.C = C

        # Pole placement: design L such that (A - L*C) has the desired poles.
        # scipy place_poles solves (A - B*K) so we use the dual:
        # design K' on (A^T, C^T) then L = K'^T.
        omega_n_roll = float(np.sqrt(c / I))
        if cfg.observer_poles is None:
            poles = _default_observer_poles(omega_n_roll, omega_e, cfg.zeta_obs)
        else:
            poles = list(cfg.observer_poles)

        if len(poles) != 5:
            raise ValueError(f"observer_poles must have 5 entries, got {len(poles)}")

        # place_poles is happiest with a numpy array of poles
        place_result = place_poles(A.T, C.T, np.asarray(poles))
        self.L = place_result.gain_matrix.T   # shape (5, n_meas)
        self.placed_poles = place_result.computed_poles

        # State estimate (initialised to zero -- ~5-10 wave periods to
        # settle from cold start; tests warm up first).
        self._x = np.zeros(5)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _xdot(self, x: np.ndarray, y: np.ndarray, M_tank: float) -> np.ndarray:
        """Continuous-time observer ODE: x_dot = A x + B_u * M_tank + L (y - C x)."""
        innovation = y - self.C @ x          # shape (n_meas,)
        return self.A @ x + (self.B_u * M_tank).ravel() + self.L @ innovation

    def update(self, y, M_tank_known: float, dt: float) -> None:
        """Advance the observer by one macro time-step using RK4.

        Parameters
        ----------
        y : float or array-like, shape (n_meas,)
            Latest measurement: ``phi`` if ``measure_phi_dot=False``,
            else ``[phi, phi_dot]``.
        M_tank_known : float
            The roll moment the tank just exerted on the hull (use the
            *applied* / saturated value, not the commanded one). N*m.
        dt : float
            Macro time-step, s.
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if y_arr.shape[0] != self.C.shape[0]:
            raise ValueError(
                f"y has {y_arr.shape[0]} components but observer expects "
                f"{self.C.shape[0]}"
            )
        x = self._x
        k1 = self._xdot(x,                  y_arr, M_tank_known)
        k2 = self._xdot(x + 0.5 * dt * k1,  y_arr, M_tank_known)
        k3 = self._xdot(x + 0.5 * dt * k2,  y_arr, M_tank_known)
        k4 = self._xdot(x +       dt * k3,  y_arr, M_tank_known)
        self._x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def M_wave_hat(self) -> float:
        """Best estimate of the wave-induced roll moment, N*m."""
        return float(self._x[2] + self._x[4])

    @property
    def phi_hat(self) -> float:
        return float(self._x[0])

    @property
    def phi_dot_hat(self) -> float:
        return float(self._x[1])

    @property
    def M_bias_hat(self) -> float:
        return float(self._x[4])

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        self._x = np.zeros(5) if x0 is None else np.asarray(x0, dtype=float).copy()
