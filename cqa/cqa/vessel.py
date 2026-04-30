"""Linearised 3-DOF vessel model and environmental force models.

State convention (vessel body frame, zero forward speed about the DP setpoint):
    x = [eta_n, eta_e, psi, u, v, r]
where
    eta_n, eta_e : north/east position deviation from setpoint [m]
    psi          : heading deviation from setpoint [rad]
    u, v         : surge/sway velocity in body frame [m/s]
    r            : yaw rate [rad/s]

Linearised vessel dynamics about the DP setpoint:
    M nu_dot + D nu = tau_thr + tau_env
    eta_dot = R(psi0) nu  ~  R0 nu  (psi small)

with M = diag(m11, m22, m66) (rigid + added mass/inertia) and
     D = diag(d11, d22, d66) linearised damping.

For the closed-loop covariance analysis we work in earth-fixed position with
R0 = R(psi0) = identity at the chosen relative-direction frame, since we only
need standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import CqaConfig, VesselParticulars, WindParticulars, CurrentParticulars


@dataclass
class LinearVesselModel:
    """3-DOF linearised vessel: M nu_dot + D nu = tau, eta_dot = nu (small psi)."""

    M: np.ndarray  # 3x3 mass matrix [kg, kg, kg m^2]
    D: np.ndarray  # 3x3 linear damping matrix
    name: str = ""

    @classmethod
    def from_config(cls, vp: VesselParticulars) -> "LinearVesselModel":
        m11 = vp.displacement_mass * (1.0 + vp.surge_added_mass_frac)
        m22 = vp.displacement_mass * (1.0 + vp.sway_added_mass_frac)
        m66 = vp.yaw_inertia * (1.0 + vp.yaw_added_inertia_frac)
        M = np.diag([m11, m22, m66])
        D = np.diag(
            [
                vp.linear_damping_surge,
                vp.linear_damping_sway,
                vp.linear_damping_yaw,
            ]
        )
        return cls(M=M, D=D, name=vp.name)

    @property
    def n_dof(self) -> int:
        return 3

    def state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Return open-loop state-space (A, B) for state x = [eta(3), nu(3)].

        A in R^{6x6}, B in R^{6x3}. Force input enters through nu_dot:
            eta_dot = nu
            nu_dot  = -M^{-1} D nu + M^{-1} tau
        """
        Minv = np.linalg.inv(self.M)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 3:6] = -Minv @ self.D
        B = np.zeros((6, 3))
        B[3:6, :] = Minv
        return A, B


# ---------------------------------------------------------------------------
# Environmental force models (mean / linearisable parts).
# ---------------------------------------------------------------------------


def _signed_cosine(angle: float) -> float:
    return float(np.cos(angle))


@dataclass
class WindForceModel:
    """Simple cosine-shaped wind drag model.

    Forces given relative wind speed `Vw` and relative wind direction `theta_rw`
    (radians, 0 = head wind, pi/2 = wind on port beam from convention here:
    we use 'wind from' direction relative to bow, positive towards port).
    """

    wp: WindParticulars
    loa: float

    def force(self, Vw: float, theta_rw: float) -> np.ndarray:
        """Mean wind force/moment in body frame [Fx, Fy, Mz]."""
        q = 0.5 * self.wp.rho_air * Vw ** 2
        Cx = -self.wp.cx_amp * np.cos(theta_rw)
        Cy = self.wp.cy_amp * np.sin(theta_rw)
        Cn = self.wp.cn_amp * np.sin(2 * theta_rw)
        Fx = q * self.wp.frontal_area * Cx
        Fy = q * self.wp.lateral_area * Cy
        Mz = q * self.wp.lateral_area * self.loa * Cn
        return np.array([Fx, Fy, Mz])

    def linearise_about(self, Vw0: float, theta_rw0: float) -> np.ndarray:
        """Sensitivity of mean wind force to wind speed perturbation, dF/dVw.

        Used to convert wind-speed gust spectra into force PSDs:
            S_F(omega) = (dF/dVw)^2 * S_Vw(omega)
        evaluated at the operating point (Vw0, theta_rw0).
        """
        q_lin = self.wp.rho_air * Vw0  # d(0.5 rho V^2)/dV = rho V
        Cx = -self.wp.cx_amp * np.cos(theta_rw0)
        Cy = self.wp.cy_amp * np.sin(theta_rw0)
        Cn = self.wp.cn_amp * np.sin(2 * theta_rw0)
        dFx = q_lin * self.wp.frontal_area * Cx
        dFy = q_lin * self.wp.lateral_area * Cy
        dMz = q_lin * self.wp.lateral_area * self.loa * Cn
        return np.array([dFx, dFy, dMz])


@dataclass
class CurrentForceModel:
    """DNV ST-0111 style mean current force, used for the operating-point bias only.

    Current variability (slow drift in current speed) is handled separately as a
    PSD term in psd.py.
    """

    cp: CurrentParticulars
    lateral_area_underwater: float
    frontal_area_underwater: float
    loa: float

    def force(self, Vc: float, theta_rc: float) -> np.ndarray:
        q = 0.5 * self.cp.rho_water * Vc ** 2
        Fx = -q * self.frontal_area_underwater * self.cp.cx_amp * np.cos(theta_rc)
        Fy = q * self.lateral_area_underwater * self.cp.cy_amp * np.sin(theta_rc)
        Mz = q * self.lateral_area_underwater * self.loa * self.cp.cn_amp * np.sin(2 * theta_rc)
        return np.array([Fx, Fy, Mz])
