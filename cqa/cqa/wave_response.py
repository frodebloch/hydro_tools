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
from .psd import jonswap_psd
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

    pdstrip convention (matches csov_pdstrip.dat / brucon WaveResponse):
        beta = 180  => head sea
        beta =  90  => beam from port (wave going to starboard)
        beta =   0  => following
        beta = 270  => beam from starboard

    Mapping: beta_deg = (180 - theta_rel_deg) mod 360.
    """
    theta_deg = float(np.degrees(theta_wave_rel))
    return float(np.mod(180.0 - theta_deg, 360.0))


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
    omega_grid  : optional custom integration grid [rad/s]; defaults
                  to a 256-point linear grid across the RAO range.
    spreading   : directional-spreading model. Default: cos-2s, s=15
                  (DNV-RP-C205 wind-sea typical, ~21 deg one-sigma).
                  Pass ``SeaSpreading.long_crested()`` for the
                  single-direction long-crested limit.

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
        spreading = SeaSpreading()  # cos-2s, s=15 default

    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)
    beta_deg_mean = cqa_theta_rel_to_pdstrip_beta_deg(theta_wave_rel)
    beta_deg_samples = np.array(
        [cqa_theta_rel_to_pdstrip_beta_deg(a) for a in angles_rel]
    )

    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)  # (6,) real
    S_eta = jonswap_psd(omega, Hs, Tp, gamma)            # (n_omega,)

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
) -> float:
    """Sigma_L_wave summed in quadrature over multiple sea states.

    Each sea state in ``sea_states`` is a tuple
    ``(Hs, Tp, theta_wave_rel, gamma)`` with theta in radians. Components
    are assumed mutually independent (typical assumption for wind-sea
    + distinct swell), so variances add:

        sigma_L_wave_total^2 = sum_i sigma_L_wave_i^2.

    The same ``spreading`` model is applied to every component;
    use multiple calls + manual quadrature sum if components need
    different spreading.

    Returned value is metres.
    """
    var_total = 0.0
    for Hs, Tp, theta, gamma in sea_states:
        res = sigma_L_wave(joint, cfg, rao_table, Hs, Tp, theta, gamma=gamma,
                           omega_grid=omega_grid, spreading=spreading)
        var_total += res.sigma_L_wave ** 2
    return float(np.sqrt(var_total))


__all__ = [
    "WaveLengthResult",
    "cqa_theta_rel_to_pdstrip_beta_deg",
    "sigma_L_wave",
    "sigma_L_wave_multimodal",
]
