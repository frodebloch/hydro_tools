"""1st-order wave-frequency response of the latched gangway telescope.

Once the gangway has latched onto the structure, the slew (alpha) and
luff (beta) joints free-wheel angularly while the telescope absorbs the
linear motion of the rotation centre relative to the world-fixed tip.
With no ship-side compensator on this Seaonics design, the telescope
sees the full vessel wave-frequency motion at the rotation centre.

Linearisation (small motion) gives

    Delta_L_wave(t) = c6 . xi(t)

with

    c6  = telescope_sensitivity_6dof(joint, gw)         (length 6)
    xi  = body 6-DOF wave-frequency motion at the
          vessel reference origin used by the pdstrip RAOs.

If the wave elevation has spectrum ``S_eta(omega)`` and the complex RAO
matrix ``H_6dof(omega, beta)`` gives the per-metre-amplitude motion, the
variance of Delta_L_wave is the standard frequency-domain quadrature

    sigma_L_wave^2 = integral_0^inf | c6 . H_6dof(omega, beta) |^2
                                   * S_eta(omega) d omega.

Production / prototype boundary
-------------------------------
* H_6dof is read once at vessel-config load from pdstrip data
  (csov_pdstrip.dat). In the C++ port it lives next to
  ``brucon::dp::WaveResponse``.
* S_eta(omega) and the wave direction theta_wave_rel are NOT measured
  on board. They come from a wave-spectrum provider — forecast
  (NORA3 / WW3), wave radar, or wind-sea analogy via
  ``hydro_tools/environment/wave_buoy.py``. The C++ port should expose
  this as an injectable interface (e.g. ``WaveSpectrumProvider``).
* The current cqa prototype uses a single parametric JONSWAP spectrum
  with operator-set (Hs, Tp, theta). Multi-modal seas
  (wind-sea + swell) and Torsethaugen are straightforward extensions
  via adding spectra in quadrature -- ``sigma_L_wave_total`` accepts a
  list.

Caveat on roll resonance / GM sensitivity / anti-roll tanks
-----------------------------------------------------------
The pdstrip roll RAO has a resonance at the model's roll natural
frequency, which scales as sqrt(GM). Two effects that the pdstrip
data does NOT model can shift sigma_roll significantly relative to
the real vessel:

1. **GM mismatch.** If the GM used to generate the pdstrip input is
   on the low side relative to the vessel's actual loading
   condition, the roll resonance period T_roll will be too long and
   the resonance peak will sit further from a typical Tp than it
   should -- resulting in an UNDER-prediction of sigma_roll. (Strip
   theory is under-damped on roll if anything, which would
   over-predict at resonance, so the GM effect is the more likely
   issue.)

2. **Anti-roll tanks.** A passive U-tube or flume anti-roll tank
   acts as a tuned liquid damper, tuned near T_roll. Its effect:
     - strong attenuation of the roll RAO at resonance,
     - mild amplification on the resonance shoulders.
   pdstrip does not model the tank. So the bare-hull pdstrip RAO
   has a sharper resonance peak than reality but smaller
   off-resonance response. Whether sigma_L_wave from pdstrip is
   conservative or non-conservative therefore depends on where Tp
   sits relative to T_roll.

Implications for the production system: the RAO data must be
sourced for the correct loading condition AND the correct hull
configuration (anti-roll tank state). For the prototype, the
sigma_L_wave numbers should be cross-checked against time-domain
simulation with the as-built GM and the tank model active before
they are used as an operability gate.

Conventions
-----------
* ``theta_wave_rel`` (rad) follows the cqa-wide relative weather
  convention: 0 = head (waves coming from bow), +pi/2 = waves from
  port beam, +pi = following. Internally converted to the pdstrip
  beta convention (180 - theta_rel_deg, wrapped to [0, 360)).
* Returned ``sigma_L_wave`` is in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .config import CqaConfig
from .gangway import GangwayJointState, telescope_sensitivity_6dof
from .psd import jonswap_psd, wave_elevation_psd, WaveSpectrumKind
from .rao import RaoTable, evaluate_rao
from .sea_spreading import SeaSpreading, spreading_quadrature


# ---------------------------------------------------------------------------
# Angle conversion
# ---------------------------------------------------------------------------


def cqa_theta_rel_to_pdstrip_beta_deg(theta_wave_rel: float) -> float:
    """Convert cqa relative-weather angle [rad] to pdstrip beta [deg].

    cqa convention (matches WindForceModel and excursion_polar):
        theta_rel = 0     => weather from the bow (head)
        theta_rel = +pi/2 => from the port beam
        theta_rel = +pi   => following

    pdstrip convention (matches csov_pdstrip.dat / brucon WaveResponse,
    decoded from brucon/libs/dp/vessel_model/wave_response.cpp:74 and
    cross-checked against vessel_simulator_model_tests.cpp:493):
        beta = 180  => head sea
        beta =  90  => beam from starboard (force pushes vessel to port)
        beta =   0  => following
        beta = 270  => beam from port (force pushes vessel to starboard)

    Mapping: beta_deg = (180 + theta_rel_deg) mod 360.

    Derivation: brucon uses
        pdstrip_angle = (heading_compass + 180 - wave_from_compass) mod 360.
    cqa's theta_rel = wrap_pi(heading_compass - wave_from_compass), so
        pdstrip_angle = (theta_rel_deg + 180) mod 360.
    """
    theta_deg = float(np.degrees(theta_wave_rel))
    return float(np.mod(180.0 + theta_deg, 360.0))


# ---------------------------------------------------------------------------
# Frequency grid for the wave integral
# ---------------------------------------------------------------------------


def _default_omega_grid(table: RaoTable, n: int = 256) -> np.ndarray:
    """Linear grid spanning the RAO frequency range, for the variance integral.

    JONSWAP S_eta peaks near omega_p (~0.6-1.2 rad/s for SOV weather
    windows); the integrand has most of its energy in the lower half
    of the pdstrip range. A linear grid with 256 points across
    [omega_min, omega_max] is plenty for trapezoidal accuracy at the
    1% level on quantities that are integrals of smooth spectra times
    smooth RAOs.

    Note: this is NOT the same grid the brucon C++ time-domain code
    uses for wave realisation. Pdstrip itself stores the RAOs on a
    geometrically-spaced (log) grid, and brucon's WaveResponse
    realises the time-domain wave elevation on that geometric grid
    deliberately to avoid commensurate periods (a uniform grid would
    make the random-phase wave realisation periodic with period
    2*pi/Delta_omega). For our frequency-domain VARIANCE integral
    that concern does not apply -- the only thing the grid affects is
    trapezoidal quadrature accuracy, which a linear grid handles
    slightly better than a log grid because it puts more points in the
    high-omega range where the smooth RAO resonances live. The
    spectrum factor S_eta itself drops off so fast at high omega that
    coverage of the low-omega range is not the bottleneck.
    """
    return np.linspace(table.omega[0], table.omega[-1], n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveLengthResult:
    """Detailed result of a single sigma_L_wave evaluation."""

    sigma_L_wave: float                   # m, 1-sigma telescope-length deviation
    sigma_L_wave_per_dof: np.ndarray      # (6,) m, per-DOF independent contribution
                                          #          sqrt( int |c_k H_k|^2 S_eta domega )
    omega: np.ndarray                     # rad/s, integration grid
    integrand: np.ndarray                 # |c6 . H_6dof|^2 * S_eta on the grid,
                                          #         summed over directional spread
    beta_deg: float                       # pdstrip wave direction at MEAN heading
    beta_deg_samples: np.ndarray          # (n_dir,) pdstrip betas across spread
    spread_weights: np.ndarray            # (n_dir,) directional weights (sum=1)
    Hs: float
    Tp: float


def sigma_L_wave(
    joint: GangwayJointState,
    cfg: CqaConfig,
    rao_table: RaoTable,
    Hs: float,
    Tp: float,
    theta_wave_rel: float,
    gamma: float = 3.3,
    omega_grid: Optional[np.ndarray] = None,
    spreading: Optional[SeaSpreading] = None,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> WaveLengthResult:
    """1st-order wave-frequency telescope-length std dev for one sea state.

    Parameters
    ----------
    joint       : current gangway joint state.
    cfg         : full cqa config (used for cfg.gangway).
    rao_table   : 6-DOF RAO table (typically from load_pdstrip_rao()).
    Hs, Tp      : significant wave height [m] and peak period [s].
    theta_wave_rel : MEAN relative wave direction [rad], cqa convention
                     (0 = head, +pi/2 = from port).
    gamma       : JONSWAP peakedness. Default 3.3 (DNV-RP-C205 mean).
                  Ignored when ``spectrum == 'bretschneider'``.
    omega_grid  : optional custom integration grid [rad/s]; defaults
                  to a 256-point linear grid across the RAO range.
    spreading   : directional-spreading model. Default: cos-2s, s=4
                  (DNV-RP-C205 wind-sea range s in 2-10; matches
                  brucon WaveSpectrum default cos^n n=2 in the
                  narrow-spread Gaussian-width sense).
                  Pass ``SeaSpreading.long_crested()`` for the
                  single-direction long-crested limit.
    spectrum    : wave-elevation PSD shape. Default
                  ``'bretschneider'`` (IMCA / DNV-ST-0111 / brucon
                  vessel_simulator default). Pass ``'jonswap'`` for
                  the peakier DNV-RP-C205 spectrum.

    Returns
    -------
    WaveLengthResult.

    Notes
    -----
    * Short-crested generalisation: the 2-D spectrum factors as
      ``S_eta(omega) D(phi)``, so
        sigma_L^2 = sum_k w_k * integral
                       |c6 . H_6dof(omega, beta_bar + phi_k)|^2
                       * S_eta(omega) d omega.
      Spreading reduces sigma_L when the RAO is sharply peaked
      around the mean direction (typical for beam-sea sway/roll);
      it has weaker effect for broad RAO directional dependence
      (head-sea surge).
    * The per-DOF breakdown ``sigma_L_wave_per_dof[k]`` is what each
      DOF would contribute *if it were the only one excited*, summed
      across the directional spread. The total ``sigma_L_wave`` is
      NOT the quadrature sum of these because the 6 DOFs share the
      same wave elevation -- they are perfectly correlated via
      H_6dof(omega, beta), so the proper total uses
      ``|c6 . H_6dof|`` first, then squares and integrates.
    """
    omega = _default_omega_grid(rao_table) if omega_grid is None else np.asarray(omega_grid, dtype=float)

    if spreading is None:
        spreading = SeaSpreading()  # cos-2s s=4 (brucon n=2 equivalent)

    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)
    beta_deg_mean = cqa_theta_rel_to_pdstrip_beta_deg(theta_wave_rel)
    beta_deg_samples = np.array(
        [cqa_theta_rel_to_pdstrip_beta_deg(a) for a in angles_rel]
    )

    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)  # (6,) real
    S_eta = wave_elevation_psd(omega, Hs, Tp, kind=spectrum, gamma=gamma)  # (n_omega,)

    integrand_total = np.zeros_like(omega)
    var_total = 0.0
    sigma_per_dof_var = np.zeros(6)

    for w_k, beta_k in zip(w_dir, beta_deg_samples):
        H_k = evaluate_rao(rao_table, omega, beta_k)        # (n_omega, 6) complex
        proj_k = H_k @ c6                                    # (n_omega,) complex
        integrand_k = (np.abs(proj_k) ** 2) * S_eta
        integrand_total += w_k * integrand_k
        var_total += w_k * float(np.trapezoid(integrand_k, omega))

        # Per-DOF diagnostic (averaged across spread):
        for j in range(6):
            integrand_j = (np.abs(c6[j] * H_k[:, j]) ** 2) * S_eta
            sigma_per_dof_var[j] += w_k * float(np.trapezoid(integrand_j, omega))

    sigma_total = float(np.sqrt(max(var_total, 0.0)))
    sigma_per_dof = np.sqrt(np.maximum(sigma_per_dof_var, 0.0))

    return WaveLengthResult(
        sigma_L_wave=sigma_total,
        sigma_L_wave_per_dof=sigma_per_dof,
        omega=omega,
        integrand=integrand_total,
        beta_deg=beta_deg_mean,
        beta_deg_samples=beta_deg_samples,
        spread_weights=w_dir,
        Hs=Hs,
        Tp=Tp,
    )


def sigma_L_wave_multimodal(
    joint: GangwayJointState,
    cfg: CqaConfig,
    rao_table: RaoTable,
    sea_states: Iterable[Tuple[float, float, float, float]],
    omega_grid: Optional[np.ndarray] = None,
    spreading: Optional[SeaSpreading] = None,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> float:
    """Sigma_L_wave summed in quadrature over multiple sea states.

    Each sea state in ``sea_states`` is a tuple
    ``(Hs, Tp, theta_wave_rel, gamma)`` with theta in radians. The
    ``gamma`` per-component is **only used when ``spectrum=='jonswap'``**;
    for the default ``'bretschneider'`` it is ignored (the tuple shape
    is preserved for backwards compatibility). Components are assumed
    mutually independent (typical assumption for wind-sea + distinct
    swell), so variances add:

        sigma_L_wave_total^2 = sum_i sigma_L_wave_i^2.

    The same ``spreading`` and ``spectrum`` are applied to every
    component; use multiple calls + manual quadrature sum if components
    need different spreading or spectra.

    Returned value is metres.
    """
    var_total = 0.0
    for Hs, Tp, theta, gamma in sea_states:
        res = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta,
                           gamma=gamma, omega_grid=omega_grid, spreading=spreading,
                           spectrum=spectrum)
        var_total += res.sigma_L_wave ** 2
    return float(np.sqrt(var_total))


@dataclass(frozen=True)
class PositionWaveResult:
    """1st-order wave-frequency horizontal position response at a body point.

    Returned by :func:`sigma_pos_wave_at_body_point`. Mirrors
    :class:`WaveLengthResult` but for the two horizontal components
    (body-frame x = surge-at-point, body-frame y = sway-at-point) of
    the position of a body-fixed point. The dp_base point is the
    intended use, but any body-frame point works.

    Fields
    ------
    sigma_x_wave_m, sigma_y_wave_m : 1-sigma WF position deviation
        at the requested body point, per axis [m].
    omega : (n_omega,) integration grid [rad/s].
    integrand_x, integrand_y : (n_omega,) one-sided position PSDs
        |c6_axis . H_6dof|^2 * S_eta * spreading_weights, summed over
        the directional spread. Variance recovery:
        sigma_axis^2 = trapezoid(integrand_axis, omega).
    nu0_x, nu0_y : zero-up-crossing rate of the WF position spectrum
        per axis [Hz]. Use as the wave-band ``nu_0`` input to a
        multi-band running-max formula.
    q_x, q_y : Vanmarcke spectral bandwidth of the WF position
        spectrum per axis (in (0, 1]). Use as the wave-band ``q``
        input to a multi-band running-max formula.
    beta_deg, beta_deg_samples, spread_weights, Hs, Tp : same
        meaning as in :class:`WaveLengthResult`.
    """

    sigma_x_wave_m: float
    sigma_y_wave_m: float
    omega: np.ndarray
    integrand_x: np.ndarray
    integrand_y: np.ndarray
    nu0_x: float
    nu0_y: float
    q_x: float
    q_y: float
    beta_deg: float
    beta_deg_samples: np.ndarray
    spread_weights: np.ndarray
    Hs: float
    Tp: float


def sigma_pos_wave_at_body_point(
    body_point: tuple[float, float, float],
    rao_table: RaoTable,
    Hs: float,
    Tp: float,
    theta_wave_rel: float,
    gamma: float = 3.3,
    omega_grid: Optional[np.ndarray] = None,
    spreading: Optional[SeaSpreading] = None,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> PositionWaveResult:
    """1st-order wave-frequency body-frame position std dev at a body point.

    For a body-fixed point ``r_b = (x_b, y_b, z_b)`` (relative to the
    pdstrip RAO body origin), the horizontal-plane position of the
    point in the body frame at small motion is

        eta_x_at_point = xi_surge + z_b * xi_pitch - y_b * xi_yaw
        eta_y_at_point = xi_sway  - z_b * xi_roll  + x_b * xi_yaw

    so the per-axis sensitivity 6-vectors are

        c6_x = [1, 0, 0, 0, z_b, -y_b]
        c6_y = [0, 1, 0, -z_b, 0, x_b]

    and the variance is the standard frequency-domain quadrature

        sigma_axis^2 = sum_k w_k * integral
                                 |c6_axis . H_6dof(omega, beta_bar + phi_k)|^2
                                 * S_eta(omega) d omega.

    This is the position-axis analogue of :func:`sigma_L_wave`. Use it
    to populate the wave-frequency band of the vessel-base position
    excursion in :class:`IntactPriorSummary`.

    Parameters
    ----------
    body_point : 3-tuple ``(x_b, y_b, z_b)``, body-frame coordinates of
        the point of interest, relative to the pdstrip RAO body origin
        (the same convention as ``cfg.gangway.base_position_body``). For
        the dp_base point pass ``cfg.gangway.base_position_body``. For
        the vessel CG pass ``(0, 0, 0)`` (the pdstrip origin is at the
        vessel reference point used elsewhere in cqa).
    rao_table : 6-DOF RAO table (typically from load_pdstrip_rao()).
    Hs, Tp : significant wave height [m] and peak period [s].
    theta_wave_rel : MEAN relative wave direction [rad], cqa convention.
    gamma : JONSWAP peakedness (ignored for spectrum='bretschneider').
    omega_grid : optional integration grid [rad/s].
    spreading : directional-spreading model. Default cos-2s, s=4
        (matches brucon WaveSpectrum cos^n n=2).
    spectrum : wave-elevation PSD shape. Default 'bretschneider'
        (matches brucon vessel_simulator default).

    Returns
    -------
    :class:`PositionWaveResult`.

    Notes
    -----
    * Independence assumption between WF and slow-drift bands.
      Combining sigma_*_wave_m with the slow-band sigma from the
      closed-loop covariance assumes the two bands are statistically
      independent. This is good to within a few percent for realistic
      DP setups: the slow-drift forcing PSD lives at omega < 0.05
      rad/s, while the 1st-order RAO sees significant power only at
      omega > 0.3 rad/s; cross-spectral leakage between the two is
      negligible.
    * For dp_base near the centerline (small y_b), the dominant term
      in c6_y is the sway DOF; for dp_base far from the centerline,
      the yaw lever-arm term grows and beam-on motion gets a small
      yaw-RAO contribution.
    """
    x_b, y_b, z_b = (float(c) for c in body_point)

    omega = _default_omega_grid(rao_table) if omega_grid is None else np.asarray(omega_grid, dtype=float)
    if spreading is None:
        spreading = SeaSpreading()

    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)
    beta_deg_mean = cqa_theta_rel_to_pdstrip_beta_deg(theta_wave_rel)
    beta_deg_samples = np.array(
        [cqa_theta_rel_to_pdstrip_beta_deg(a) for a in angles_rel]
    )

    # Body-point position 6-vectors (see derivation in docstring).
    c6_x = np.array([1.0, 0.0, 0.0, 0.0,  z_b, -y_b])
    c6_y = np.array([0.0, 1.0, 0.0, -z_b, 0.0,  x_b])

    S_eta = wave_elevation_psd(omega, Hs, Tp, kind=spectrum, gamma=gamma)

    integrand_x = np.zeros_like(omega)
    integrand_y = np.zeros_like(omega)
    var_x = 0.0
    var_y = 0.0
    for w_k, beta_k in zip(w_dir, beta_deg_samples):
        H_k = evaluate_rao(rao_table, omega, beta_k)         # (n_omega, 6) complex
        proj_x = H_k @ c6_x                                   # (n_omega,) complex
        proj_y = H_k @ c6_y
        i_x = (np.abs(proj_x) ** 2) * S_eta
        i_y = (np.abs(proj_y) ** 2) * S_eta
        integrand_x += w_k * i_x
        integrand_y += w_k * i_y
        var_x += w_k * float(np.trapezoid(i_x, omega))
        var_y += w_k * float(np.trapezoid(i_y, omega))

    sigma_x = float(np.sqrt(max(var_x, 0.0)))
    sigma_y = float(np.sqrt(max(var_y, 0.0)))

    # Spectral-moment-derived nu_0 and Vanmarcke q for each axis.
    # For one-sided PSD S(omega) [unit^2 / (rad/s)],
    #   m_n = integral S(omega) * omega^n d omega,
    #   nu_0 (zero-up-crossing rate, [Hz]) = (1/(2 pi)) * sqrt(m_2 / m_0),
    #   epsilon^2 = 1 - m_1^2 / (m_0 m_2)  (spectral bandwidth),
    #   q (Vanmarcke) = sqrt(1 - m_1^2 / (m_0 m_2))  (in [0, 1]).
    def _moments_nu_q(S: np.ndarray) -> tuple[float, float]:
        m0 = float(np.trapezoid(S, omega))
        if m0 <= 0.0:
            return 0.0, 1.0
        m1 = float(np.trapezoid(S * omega, omega))
        m2 = float(np.trapezoid(S * omega ** 2, omega))
        nu0 = (1.0 / (2.0 * np.pi)) * np.sqrt(max(m2 / m0, 0.0))
        eps2 = max(1.0 - (m1 ** 2) / max(m0 * m2, 1e-300), 0.0)
        q = float(np.sqrt(min(eps2, 1.0)))
        return float(nu0), q

    nu0_x, q_x = _moments_nu_q(integrand_x)
    nu0_y, q_y = _moments_nu_q(integrand_y)

    return PositionWaveResult(
        sigma_x_wave_m=sigma_x,
        sigma_y_wave_m=sigma_y,
        omega=omega,
        integrand_x=integrand_x,
        integrand_y=integrand_y,
        nu0_x=nu0_x,
        nu0_y=nu0_y,
        q_x=q_x,
        q_y=q_y,
        beta_deg=beta_deg_mean,
        beta_deg_samples=beta_deg_samples,
        spread_weights=w_dir,
        Hs=float(Hs),
        Tp=float(Tp),
    )


__all__ = [
    "WaveLengthResult",
    "PositionWaveResult",
    "cqa_theta_rel_to_pdstrip_beta_deg",
    "sigma_L_wave",
    "sigma_L_wave_multimodal",
    "sigma_pos_wave_at_body_point",
]
