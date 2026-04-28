"""Frequency-tracking air-valve controller.

Strategy
--------
The air-valve U-tube has two natural frequencies:

* ``omega_open``   — valve fully open: the gas chambers vent freely so
  the tank reduces to the open-top passive U-tube.
* ``omega_closed`` — valve fully sealed: trapped gas adds a stiffness
  ``Δk = p_atm A_res^2 W^2 / (2 V0)``, so
  ``omega_closed > omega_open``.

This controller assumes the recommended **active design**: the U-tube
geometry is detuned so ``T_open > T_n`` (the tank is "soft" relative to
the vessel) and the chamber volume is sized so that
``T_closed == T_n``. The controller then picks an opening ``u in [0,1]``
to interpolate the effective tank period between ``T_open`` (``u=1``)
and ``T_closed`` (``u=0``), continuously retuning the tank to the
current wave period.

Algorithm
---------
1.  Estimate the dominant roll period ``T_meas`` from successive
    sign-changes of ``phi_dot`` (which equals the wave encounter
    period at steady state for a linear system in regular seas).
2.  Map ``T_meas`` to a target opening:

    .. code-block:: text

        T_meas <= T_design - Δ  ->  u = 0     (close: tank at T_closed)
        T_meas >= T_design + Δ  ->  u = 1     (open:  tank at T_open)
        in between              ->  linear interpolation

    where ``T_design = T_n`` (the vessel resonance, where peak
    rejection is critical) and ``Δ = T_window`` provides hysteresis to
    avoid chatter on the boundary.
3.  Low-pass-filter the opening with time constant ``smoothing_tau``
    so the valve does not slam.

The controller relies *only* on ``phi_dot`` (no tank state is
required), preserving the loose-coupling property of the architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import AbstractValveController


@dataclass
class FrequencyTrackingController(AbstractValveController):
    """Adaptive valve controller that tracks the roll period.

    Maps the measured wave/encounter period ``T_meas`` linearly onto the
    valve opening so the tank's effective natural period tracks
    ``T_meas`` between the two extremes ``T_closed`` (``u=0``) and
    ``T_open`` (``u=1``).

    Parameters
    ----------
    T_closed
        Tank natural period at fully-closed valve (``u=0``), s.
        Should equal the vessel roll natural period for the recommended
        active design.
    T_open
        Tank natural period at fully-open valve (``u=1``), s. Should
        be greater than ``T_closed``; sets the upper end of the
        controllable band.
    smoothing_tau
        First-order low-pass time constant on the commanded opening, s.
        Set to a few measurement periods to smooth out jitter and avoid
        valve chatter.
    initial_opening
        Initial commanded opening before any zero-crossings have been
        observed. Default 1.0 (open) so the system behaves like the
        passive tank during start-up.
    min_period, max_period
        Saturation bounds on the period estimate, s. Estimates outside
        this band are discarded as spurious (start-up transient,
        noise).
    """

    T_closed: float = 11.4
    T_open: float = 14.0
    smoothing_tau: float = 5.0
    initial_opening: float = 1.0
    min_period: float = 4.0
    max_period: float = 25.0

    # internal state (not user-facing)
    _last_sign: Optional[int] = field(default=None, init=False, repr=False)
    _last_cross_t: Optional[float] = field(default=None, init=False, repr=False)
    _T_meas: Optional[float] = field(default=None, init=False, repr=False)
    _u_smooth: Optional[float] = field(default=None, init=False, repr=False)
    _last_t: Optional[float] = field(default=None, init=False, repr=False)

    def _target_opening(self, T: float) -> float:
        # Want tank period = T_meas, where tank period varies linearly
        # (approximately) with u between T_closed (u=0) and T_open (u=1).
        # Saturate at bounds.
        if T <= self.T_closed:
            return 0.0
        if T >= self.T_open:
            return 1.0
        return (T - self.T_closed) / (self.T_open - self.T_closed)

    def opening(self, vessel_kin: dict, t: float) -> float:
        phi_dot = vessel_kin.get("phi_dot", 0.0)
        sign = 1 if phi_dot > 0 else (-1 if phi_dot < 0 else 0)

        # Period estimation: time between successive same-direction sign
        # changes of phi_dot. We measure full periods (positive-going
        # zero-crossing to next positive-going), so divide by the number
        # of half-periods elapsed.
        if sign != 0 and self._last_sign is not None and sign != self._last_sign:
            if self._last_cross_t is not None:
                # Half-period from last zero-crossing of either sign.
                half_T = t - self._last_cross_t
                T_est = 2.0 * half_T
                if self.min_period <= T_est <= self.max_period:
                    self._T_meas = T_est
            self._last_cross_t = t
        if sign != 0:
            self._last_sign = sign

        # Determine target opening from current period estimate.
        if self._T_meas is None:
            u_target = self.initial_opening
        else:
            u_target = self._target_opening(self._T_meas)

        # First-order low-pass on the opening.
        if self._u_smooth is None or self._last_t is None:
            self._u_smooth = u_target
        else:
            dt = max(t - self._last_t, 0.0)
            alpha = dt / (self.smoothing_tau + dt) if self.smoothing_tau > 0 else 1.0
            self._u_smooth += alpha * (u_target - self._u_smooth)
        self._last_t = t

        return float(max(0.0, min(1.0, self._u_smooth)))

    @property
    def estimated_period(self) -> Optional[float]:
        """Latest measured roll period (s), or None if not yet observed."""
        return self._T_meas
