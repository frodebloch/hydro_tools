"""Excursion polar over relative weather direction.

For a given environmental condition (wind speed, wave Hs/Tp, current Vc) we
sweep the relative weather direction theta_rw in [0, 2 pi) and compute the
position+heading excursion covariance via Lyapunov for each direction.

Output (per direction):
    sigma_n, sigma_e, sigma_psi,
    95-percentile position ellipse semi-axes and orientation,
    full 6x6 covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import CqaConfig
from .vessel import LinearVesselModel, WindForceModel, CurrentForceModel
from .controller import LinearDpController
from .closed_loop import ClosedLoop, state_covariance_freqdomain
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)


@dataclass
class ExcursionResult:
    theta_rel_rad: np.ndarray  # (N,) relative weather direction
    sigma_n: np.ndarray  # (N,) std north position [m]
    sigma_e: np.ndarray  # (N,) std east position [m]
    sigma_psi: np.ndarray  # (N,) std heading [rad]
    sigma_u: np.ndarray  # (N,) std surge velocity [m/s]
    sigma_v: np.ndarray  # (N,) std sway velocity [m/s]
    sigma_r: np.ndarray  # (N,) std yaw rate [rad/s]
    ellipse_semi_major: np.ndarray  # (N,) 95% ellipse semi-major [m]
    ellipse_semi_minor: np.ndarray  # (N,) 95% ellipse semi-minor [m]
    ellipse_angle_rad: np.ndarray  # (N,) ellipse orientation angle in body frame
    P: np.ndarray  # (N, 6, 6) full covariance

    def radial_excursion_95(self) -> np.ndarray:
        """Approximate 95% radial excursion = semi-major axis."""
        return self.ellipse_semi_major


def _ellipse_from_cov(C2: np.ndarray, prob: float = 0.95) -> tuple[float, float, float]:
    """Return (semi-major, semi-minor, angle_rad) of a probability ellipse.

    For a 2D Gaussian, the prob-content ellipse satisfies
        x^T C^{-1} x = chi2_inv(prob, 2) = -2 ln(1 - prob).
    Semi-axes = sqrt(eigenvalue * scale).
    """
    scale = -2.0 * np.log(1.0 - prob)
    eigvals, eigvecs = np.linalg.eigh(C2)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    a = float(np.sqrt(max(eigvals[0], 0.0) * scale))
    b = float(np.sqrt(max(eigvals[1], 0.0) * scale))
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return a, b, angle


def excursion_polar(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    n_directions: int = 36,
    omega_band: tuple[float, float] = (1e-4, 1.5),
    omega_n: tuple[float, float, float] = (0.06, 0.06, 0.05),
    zeta: tuple[float, float, float] = (0.9, 0.9, 0.9),
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
) -> ExcursionResult:
    """Compute excursion polar for a single environmental condition.

    Parameters
    ----------
    cfg : CqaConfig
    Vw_mean : float
        Mean wind speed at 10 m [m/s].
    Hs, Tp : float
        Significant wave height [m] and peak period [s] for a single sea
        state (combined wind-sea + swell or just one component for the study).
    Vc : float
        Mean current speed [m/s].
    n_directions : int
        Number of relative weather directions sampled in the polar.
    omega_band : (float, float)
        Frequency-domain integration band [rad/s]. Default 1e-4 .. 1.5 covers
        all the slow-drift / wind-gust energy and a bit of the wave-frequency
        tail (which is heavily filtered by the closed loop).
    omega_n, zeta : tuples of 3 floats
        Closed-loop bandwidth and damping per DOF for the controller.
    sigma_Vc, tau_Vc : floats
        Std and correlation time of current speed variability.

    Notes
    -----
    Assumes wind, wave, and current all come from the same direction
    `theta_rel`. This is conservative (worst-case alignment) and matches the
    DNV ST-0111 prevailing-condition assumption used in brucon's
    `CapabilityAnalysis::Prevailing` mode.
    """
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift

    vessel = LinearVesselModel.from_config(vp)
    A, B = vessel.state_space()
    M_diag = np.diag(vessel.M)
    D_diag = np.diag(vessel.D)

    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=omega_n, zeta=zeta
    )
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    # Underwater areas approximated from draft x lpp / loa.
    lateral_uw = vp.lpp * vp.draft
    frontal_uw = vp.beam * vp.draft
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=lateral_uw,
        frontal_area_underwater=frontal_uw,
        loa=vp.loa,
    )

    thetas = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    sig_n = np.zeros(n_directions)
    sig_e = np.zeros(n_directions)
    sig_psi = np.zeros(n_directions)
    sig_u = np.zeros(n_directions)
    sig_v = np.zeros(n_directions)
    sig_r = np.zeros(n_directions)
    ell_a = np.zeros(n_directions)
    ell_b = np.zeros(n_directions)
    ell_ang = np.zeros(n_directions)
    P_all = np.zeros((n_directions, 6, 6))

    for i, theta in enumerate(thetas):
        # --- Wind gust force PSD ---
        S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta)

        # --- Slow-drift wave force PSD ---
        S_drift = slow_drift_force_psd_newman(
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp),
            Hs,
            Tp,
            theta,
        )

        # --- Current variability PSD ---
        # Linearise current force about Vc: dF/dVc = rho * Vc * area * C * shape.
        if Vc > 1e-9:
            F0 = current_model.force(Vc, theta)
            dFdVc = 2.0 * F0 / Vc  # since F ~ Vc^2
        else:
            dFdVc = np.zeros(3)
        S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)

        P = state_covariance_freqdomain(
            cl,
            [S_wind, S_drift, S_curr],
            omega_lo=omega_band[0],
            omega_hi=omega_band[1],
            n_points=512,
        )

        sig_n[i] = np.sqrt(max(P[0, 0], 0.0))
        sig_e[i] = np.sqrt(max(P[1, 1], 0.0))
        sig_psi[i] = np.sqrt(max(P[2, 2], 0.0))
        sig_u[i] = np.sqrt(max(P[3, 3], 0.0))
        sig_v[i] = np.sqrt(max(P[4, 4], 0.0))
        sig_r[i] = np.sqrt(max(P[5, 5], 0.0))
        a, b, ang = _ellipse_from_cov(P[0:2, 0:2], prob=0.95)
        ell_a[i] = a
        ell_b[i] = b
        ell_ang[i] = ang
        P_all[i] = P

    return ExcursionResult(
        theta_rel_rad=thetas,
        sigma_n=sig_n,
        sigma_e=sig_e,
        sigma_psi=sig_psi,
        sigma_u=sig_u,
        sigma_v=sig_v,
        sigma_r=sig_r,
        ellipse_semi_major=ell_a,
        ellipse_semi_minor=ell_b,
        ellipse_angle_rad=ell_ang,
        P=P_all,
    )
