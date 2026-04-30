"""Approximation of the brucon DP controller for the cqa study.

PD on position/heading + integral action + bias feed-forward. The integral
action accumulates an estimate of the slow environmental force (the
'bias') and feeds it forward to the thrust allocator so that, in steady
state, the controller is producing low-frequency feedback only against
*deviations* from the bias estimate.

For the linearised closed-loop covariance analysis we treat the bias path
as perfect cancellation of the mean force, leaving a PD law on position
deviation against the residual zero-mean disturbance:

    tau_ctrl = -Kp * eta - Kd * nu

with Kp, Kd 3x3 diagonal gains. Tuning is parametrised by closed-loop
natural frequency `omega_n` (rad/s) and damping ratio `zeta` per DOF, so
that:

    Kp_i = M_ii * omega_n_i^2
    Kd_i = 2 * zeta_i * M_ii * omega_n_i + D_ii   (cancels open-loop damping
                                                    bias and adds desired
                                                    closed-loop damping)

The integral action contributes only at frequencies below the dominant
disturbance band and is omitted from the Lyapunov analysis (it is a slow
process tracking the bias, modelled instead as 'perfect bias rejection').
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LinearDpController:
    Kp: np.ndarray  # 3x3
    Kd: np.ndarray  # 3x3

    @classmethod
    def from_bandwidth(
        cls,
        M: np.ndarray,
        D: np.ndarray,
        omega_n: tuple[float, float, float] = (0.06, 0.06, 0.05),
        zeta: tuple[float, float, float] = (0.9, 0.9, 0.9),
    ) -> "LinearDpController":
        """Tune diagonal PD gains from desired closed-loop bandwidth/damping.

        omega_n in rad/s. Defaults give surge/sway natural period ~105 s and
        yaw natural period ~125 s, typical for DP station-keeping.
        """
        omega = np.asarray(omega_n)
        zet = np.asarray(zeta)
        Mdiag = np.diag(M)
        Ddiag = np.diag(D)
        kp = Mdiag * omega ** 2
        kd = 2.0 * zet * Mdiag * omega - Ddiag
        # Ensure non-negative effective damping.
        kd = np.maximum(kd, 0.5 * Ddiag)
        return cls(Kp=np.diag(kp), Kd=np.diag(kd))

    def feedback(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (Kp, Kd) gains for state x = [eta, nu].

        Control law: tau_ctrl = -[Kp Kd] x.
        """
        return self.Kp, self.Kd
