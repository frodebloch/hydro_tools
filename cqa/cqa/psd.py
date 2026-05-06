"""Environmental force power spectral densities (PSDs) for excursion analysis.

The closed-loop position-keeping bandwidth of a DP-controlled vessel is on the
order of 0.05 rad/s (period ~ 100 s). Disturbance energy *near and below*
that bandwidth determines the position excursion variance. The wave-frequency
band (0.4 .. 1.5 rad/s) is filtered out by the wave filter in the observer
and is irrelevant for the position-keeping cov; it is treated separately at
the gangway tip in the operability map (P3).

For P1 we provide three force-PSD contributors:

1. **Wind gust force PSD** from NPD/Davenport wind-speed spectrum, mapped
   through the linearised wind-force sensitivity dF/dVw.
2. **Slow-drift wave force PSD** via Newman's approximation: the wave drift
   force quadratic transfer function evaluated on the diagonal generates a
   slow-drift spectrum at frequency differences. We use a closed-form
   approximation valid for narrow-band wave spectra.
3. **Current variability PSD**: the current speed varies slowly (tide, eddy
   activity) with a characteristic time scale of minutes to hours; modelled
   as a first-order Gauss-Markov process and projected onto force.

Each contributor returns a function returning the 3x3 force PSD matrix
S_F(omega) [N^2/(rad/s) for surge/sway diagonal; (Nm)^2/(rad/s) for yaw].
For the Lyapunov solve we condense to a variance-equivalent white-noise
intensity W = integral S_F(omega) d omega / (2 pi) over the relevant band,
applied as B_w W B_w^T at the slow-frequency input.
"""

from __future__ import annotations

from typing import Callable, Literal
import numpy as np

from .vessel import WindForceModel


# Wave-elevation spectrum kind used by all downstream consumers
# (1st-order RAO variance, Newman slow-drift PSD, mean drift force,
# time-domain realisation). 'bretschneider' is the IMCA / DNV-ST-0111
# operability standard (a 2-parameter PM-family spectrum with no
# peak-enhancement); it is implemented exactly as ``jonswap_psd`` with
# ``gamma = 1.0``. 'jonswap' uses ``gamma = 3.3`` by default
# (DNV-RP-C205 fetch-limited mean), peakier than Bretschneider --
# more conservative for QTF integrals when ``T_p`` is at or above the
# QTF peak (typical wind-sea range), less conservative for very short
# T_p; see analysis.md section 12.16 for the per-T_p comparison.
WaveSpectrumKind = Literal["bretschneider", "jonswap"]


# ---------------------------------------------------------------------------
# 1. Wind gust force PSD
# ---------------------------------------------------------------------------


def npd_wind_speed_psd(omega: np.ndarray, Vw_mean: float, z: float = 10.0) -> np.ndarray:
    """NPD wind speed spectrum (single-sided, rad/s).

    NPD spectrum (Norwegian Petroleum Directorate / DNV-RP-C205) is a
    standard for offshore wind gust modelling:

        S_V(f) = 320 * (Vw_mean/10)^2 * (z/10)^0.45 / (1 + f_hat^n)^(5/(3n))
        f_hat  = 172 * f * (z/10)^(2/3) * (Vw_mean/10)^(-0.75)
        n      = 0.468

    Returned as one-sided PSD in [m^2/s^2 / (rad/s)] vs. omega [rad/s].
    """
    omega = np.asarray(omega, dtype=float)
    f = np.maximum(omega, 1e-9) / (2 * np.pi)
    n = 0.468
    f_hat = 172.0 * f * (z / 10.0) ** (2.0 / 3.0) * (Vw_mean / 10.0) ** (-0.75)
    Sf = 320.0 * (Vw_mean / 10.0) ** 2 * (z / 10.0) ** 0.45 / (1.0 + f_hat ** n) ** (5.0 / (3.0 * n))
    # Convert from S(f) [per Hz] to S(omega) [per rad/s]: divide by 2*pi.
    return Sf / (2.0 * np.pi)


def npd_wind_gust_force_psd(
    wind_model: WindForceModel,
    Vw_mean: float,
    theta_rw: float,
    z: float = 10.0,
    direction_spread_std_deg: float = 10.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function S_F(omega) -> 3x3 wind force PSD matrix.

    Linearises wind force about (Vw_mean, theta_rw): dF/dVw = g, then
    S_F(omega) = g g^T * S_V(omega), expanded over a wind-direction spread
    so the force PSD matrix is full-rank.
    """
    sigma_th = np.radians(direction_spread_std_deg)
    n_dir = 21
    dth = np.linspace(-3.0 * sigma_th, 3.0 * sigma_th, n_dir)
    w = np.exp(-(dth ** 2) / (2.0 * sigma_th ** 2))
    w /= w.sum()
    G = np.zeros((3, 3))
    for wi, dthi in zip(w, dth):
        g = wind_model.linearise_about(Vw_mean, theta_rw + dthi)
        G += wi * np.outer(g, g)

    def S_F(omega: np.ndarray) -> np.ndarray:
        S_V = npd_wind_speed_psd(omega, Vw_mean, z)
        # Broadcast: result shape (..., 3, 3)
        return G[None, :, :] * S_V[..., None, None] if np.ndim(omega) else G * S_V

    return S_F


# ---------------------------------------------------------------------------
# 2. Slow-drift wave force PSD (Newman approximation)
# ---------------------------------------------------------------------------


def jonswap_psd(omega: np.ndarray, Hs: float, Tp: float, gamma: float = 3.3) -> np.ndarray:
    """Single-sided JONSWAP wave-elevation spectrum [m^2 / (rad/s)] vs omega [rad/s].

    Form (DNV-RP-C205):
        S(omega) = A_gamma * (5/16) * Hs^2 * omega_p^4 / omega^5
                   * exp(-1.25 (omega_p/omega)^4) * gamma^r
        r        = exp(-(omega - omega_p)^2 / (2 sigma^2 omega_p^2))
        A_gamma  = 1 - 0.287 ln(gamma)    (normalization so that m_0 ~ Hs^2/16)
    """
    omega = np.asarray(omega, dtype=float)
    out = np.zeros_like(omega)
    mask = omega > 1e-9
    om = omega[mask]
    omega_p = 2.0 * np.pi / Tp
    sigma = np.where(om <= omega_p, 0.07, 0.09)
    r = np.exp(-((om - omega_p) ** 2) / (2.0 * sigma ** 2 * omega_p ** 2))
    A_gamma = 1.0 - 0.287 * np.log(gamma)
    S = (
        A_gamma
        * (5.0 / 16.0)
        * Hs ** 2
        * omega_p ** 4
        / om ** 5
        * np.exp(-1.25 * (omega_p / om) ** 4)
        * gamma ** r
    )
    out[mask] = S
    return out


def bretschneider_psd(omega: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
    """Single-sided Bretschneider wave-elevation spectrum [m^2 / (rad/s)].

    Bretschneider (1959) is the IMCA / DNV-ST-0111 reference operability
    spectrum, a 2-parameter PM-family form with no peak enhancement.
    It is mathematically identical to the JONSWAP form with
    ``gamma = 1`` (the peak-enhancement multiplier ``gamma^r`` collapses
    to 1, the normalisation ``A_gamma = 1 - 0.287 ln(1) = 1``):

        S(omega) = (5/16) * Hs^2 * omega_p^4 / omega^5
                   * exp(-1.25 (omega_p / omega)^4)

    Implemented as a thin wrapper around :func:`jonswap_psd` with
    ``gamma = 1.0`` so the two paths share the same code (and the
    equivalence is enforced numerically). Consumers that want
    spectrum-agnostic behaviour should call :func:`wave_elevation_psd`
    with the ``kind`` kwarg.
    """
    return jonswap_psd(omega, Hs, Tp, gamma=1.0)


def wave_elevation_psd(
    omega: np.ndarray,
    Hs: float,
    Tp: float,
    kind: WaveSpectrumKind = "bretschneider",
    gamma: float = 3.3,
) -> np.ndarray:
    """Dispatch to :func:`bretschneider_psd` or :func:`jonswap_psd`.

    ``gamma`` is ignored for ``kind == 'bretschneider'``. Default kind
    is Bretschneider, matching brucon's ``vessel_simulator`` default
    (see ``vessel_simulator_wrapper.cpp`` -- the ``WaveSpectrum`` ctor
    branch when ``wave_spectrum_type`` is unset in
    ``vessel_simulator_settings.prototxt``) and the IMCA/DNV-ST-0111
    operability conventions.
    """
    if kind == "bretschneider":
        return jonswap_psd(omega, Hs, Tp, gamma=1.0)
    if kind == "jonswap":
        return jonswap_psd(omega, Hs, Tp, gamma=gamma)
    raise ValueError(
        f"Unknown wave spectrum kind: {kind!r} (expected 'bretschneider' or 'jonswap')"
    )


def slow_drift_force_psd_newman(
    drift_amp: tuple[float, float, float],
    Hs: float,
    Tp: float,
    theta_rw: float,
    gamma: float = 3.3,
    spreading_std_deg: float = 25.0,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> Callable[[np.ndarray], np.ndarray]:
    """Newman-approximation slow-drift force PSD.

    Newman (1974): for narrow-band sea, the second-order force PSD at low
    difference-frequency mu is
        S_F2(mu) = 8 * integral S_eta(omega) * S_eta(omega + mu) * T_ii(omega)^2 d omega
    where T_ii is the diagonal of the QTF (~ mean drift coefficient at
    frequency omega). We approximate T_ii by a constant equal to the mean
    drift force per Hs^2, giving:
        S_F2(mu) ~ 8 * T^2 * integral S_eta(omega) S_eta(omega + mu) d omega

    For the Lyapunov solve we only need the value of S_F2 in the
    closed-loop band (mu << omega_p), which is well approximated by
        S_F2(mu) ~ 8 * T^2 * integral S_eta(omega)^2 d omega   for small mu.

    Directional spreading of the wave field (default 25 deg std) injects
    energy into off-mean-direction force components, ensuring the force PSD
    matrix is full-rank and the resulting position covariance has a proper
    2-D ellipse rather than a line.
    """
    sigma_th = np.radians(spreading_std_deg)
    # Sample directions on a fine grid weighted by a wrapped-Gaussian spread.
    n_dir = 31
    dth = np.linspace(-3.0 * sigma_th, 3.0 * sigma_th, n_dir)
    w = np.exp(-(dth ** 2) / (2.0 * sigma_th ** 2))
    w /= w.sum()

    # Pre-compute the wave-spectrum self-correlation integral I = int S_eta^2 d omega.
    omega_grid = np.linspace(0.05, 4.0, 4000)
    S_eta = wave_elevation_psd(omega_grid, Hs, Tp, kind=spectrum, gamma=gamma)
    I = np.trapezoid(S_eta ** 2, omega_grid)
    S0 = 8.0 * I

    # Build expected outer-product over directional spread:
    #   G = sum_i w_i * T(theta_rw + dth_i) T(theta_rw + dth_i)^T * S0
    G = np.zeros((3, 3))
    for wi, dthi in zip(w, dth):
        th = theta_rw + dthi
        T_vec = np.array(
            [
                drift_amp[0] * np.cos(th),
                drift_amp[1] * np.sin(th),
                drift_amp[2] * np.sin(2.0 * th),
            ]
        )
        G += wi * np.outer(T_vec, T_vec) * S0

    def S_F(omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        rolloff = 1.0 / (1.0 + (omega / 0.2) ** 4)
        return G[None, :, :] * rolloff[..., None, None] if np.ndim(omega) else G * float(rolloff)

    return S_F


# ---------------------------------------------------------------------------
# 3. Current variability force PSD
# ---------------------------------------------------------------------------


def current_variability_force_psd(
    current_force_at_unit_speed: np.ndarray,  # 3-vector dF/dVc at operating point
    sigma_Vc: float = 0.1,  # std of current speed variation [m/s]
    tau: float = 600.0,  # correlation time [s]
) -> Callable[[np.ndarray], np.ndarray]:
    """First-order Gauss-Markov current speed variability mapped to force PSD.

    Vc(t) = Vc_mean + delta_Vc(t),  delta_Vc ~ OU(sigma_Vc, tau).
    S_dVc(omega) = 2 sigma_Vc^2 tau / (1 + (omega tau)^2).
    """
    g = current_force_at_unit_speed  # 3-vector
    G = np.outer(g, g)

    def S_F(omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        S_dVc = 2.0 * sigma_Vc ** 2 * tau / (1.0 + (omega * tau) ** 2)
        return G[None, :, :] * S_dVc[..., None, None] if np.ndim(omega) else G * float(S_dVc)

    return S_F


# ---------------------------------------------------------------------------
# Helper: integrate a 3x3 PSD over a band to get a white-noise intensity.
# ---------------------------------------------------------------------------


def integrated_intensity(
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega_lo: float = 1e-4,
    omega_hi: float = 0.5,
    n_points: int = 2000,
) -> np.ndarray:
    """Equivalent white-noise intensity matrix W matching variance in [omega_lo, omega_hi].

    The Lyapunov analysis treats the disturbance as white noise with
    autocorrelation
        E[w(t) w(s)^T] = W * delta(t-s).
    For a one-sided PSD S_F(omega) [units of force^2 / (rad/s)] in rad/s, the
    relationship to the white-noise intensity is

        W = pi * S_F(omega_n)        (matching the resonance peak of a 2nd-order system)

    For our disturbances (wind gust, slow drift, current variability) the PSD
    is slowly varying across the closed-loop band, so we use the band-averaged
    one-sided value:

        S_F_avg = (1/(omega_hi - omega_lo)) * integral S_F(omega) d omega
        W       = pi * S_F_avg

    This is a standard quasi-stationary equivalent-white-noise approximation;
    the resulting variance matches direct frequency-domain integration to
    within a few percent for typical DP closed-loop bandwidths.
    """
    omega = np.linspace(omega_lo, omega_hi, n_points)
    S_sum = np.zeros((3, 3))
    for S_F in S_F_funcs:
        S = np.array([S_F(w) for w in omega])  # (n,3,3)
        S_sum += np.trapezoid(S, omega, axis=0)
    S_avg = S_sum / (omega_hi - omega_lo)
    W = np.pi * S_avg
    return W
