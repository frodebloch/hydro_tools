"""Bang-bang air-valve controller driven by vessel roll kinematics.

.. warning::

   **This controller is deprecated.** Empirical testing on the CSOV
   showed it performs *worse* than a fully-open passive valve at
   resonance (45 % reduction vs 68 %). It is retained here only so that
   existing imports do not break. Use
   :class:`roll_reduction_tanks.controllers.frequency_tracking.FrequencyTrackingController`
   instead.

Why it fails
------------
The original rule was

    open whenever  phi_dot * phi_ddot < 0  (i.e. roll is decelerating).

For an approximately-sinusoidal vessel response ``phi = A sin(omega t)``,

    phi_dot * phi_ddot = -(A^2 omega^3 / 2) * sin(2 omega t),

so the rule chatters at twice the roll frequency with a fixed 50 % duty
cycle and **no useful phase selection**. The valve simply oscillates
between open and closed every quarter-period regardless of the wave
period or detuning, smearing the tank's effective dynamics rather than
retuning it.

A properly designed adaptive controller needs a *period* estimate (or
phase reference), not just an instantaneous sign of a product. See
``frequency_tracking.py`` for the working approach.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from .base import AbstractValveController


@dataclass
class BangBangController(AbstractValveController):
    """Deprecated bang-bang controller. See module docstring.

    Parameters
    ----------
    open_value
        Opening commanded when the rule decides to open. Default 1.
    closed_value
        Opening commanded when the rule decides to close. Default 0.
    phi_dot_threshold
        Magnitude of ``phi_dot`` below which the valve is forced closed
        (intended to avoid chatter near zero crossings; in practice does
        not help — see module docstring).
    """

    open_value: float = 1.0
    closed_value: float = 0.0
    phi_dot_threshold: float = 1e-4

    def __post_init__(self):
        warnings.warn(
            "BangBangController is deprecated: it chatters at 2*omega "
            "with no useful phase selection and is empirically worse "
            "than a fully-open passive valve. Use "
            "FrequencyTrackingController instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def opening(self, vessel_kin: dict, t: float) -> float:
        phi_dot = vessel_kin.get("phi_dot", 0.0)
        phi_ddot = vessel_kin.get("phi_ddot", 0.0)
        if abs(phi_dot) < self.phi_dot_threshold:
            return self.closed_value
        return self.open_value if (phi_dot * phi_ddot) < 0.0 else self.closed_value
