"""2nd-order wave-drift forces from pdstrip diagonal QTF.

Two products, both consumed downstream by the closed-loop / WCFDI
pipeline:

  1. **Mean drift force** (deterministic, time-invariant for a given
     sea state) -- adds to the wind/current mean load. Reduces the
     residual "unmodelled environmental disturbance" the bias estimator
     has to absorb, and through that reduces bias-state variance and
     position-error variance after a WCF.

  2. **Slow-drift force PSD** in vessel surge/sway/yaw -- non-zero at
     low difference frequency mu, exciting the DP closed-loop
     pendulum (typical period 60-200 s). This is the "Newman
     approximation" PSD that drives slow horizontal drift oscillation
     in the position state. Drops in to
     ``cqa.closed_loop.state_covariance_freqdomain_general`` next to
     the wind-gust and current-variability PSDs.

Theory
------
Pinkster's diagonal-QTF approximation: only the diagonal D_i(omega) of
the second-order force quadratic transfer function is significant for
slow drift; off-diagonal terms are constructed by Newman's
approximation from D_i evaluated at neighbouring frequencies.

For a long-crested sea with one-sided wave-elevation spectrum
``S_eta(omega)`` and pdstrip drift coefficient ``D_i(omega, beta)``
(physical units N/m^2 or Nm/m^2 per unit wave amplitude squared, sign
in vessel body frame at heading beta):

  Mean drift, Faltinsen [90] eq. 5.41:
      F_i,mean = 2 * integral_0^inf  D_i(omega, beta) * S_eta(omega) d omega

  Slow-drift PSD at difference frequency mu, Newman approximation
  (Faltinsen [90] eq. 5.39, arithmetic-mean variant -- this is the
  variant brucon's wave_response.cpp uses in its time-domain code at
  line 333, ``T(omega_i, omega_j) ~ 0.5*(D(omega_i)+D(omega_j))``):

      S_F,i(mu) = 8 * integral_0^inf
                       [0.5 * (D_i(omega) + D_i(omega+mu))]^2
                       * S_eta(omega) * S_eta(omega+mu) d omega

  We use the same arithmetic-mean variant for parity with brucon's
  time-domain realisation. The geometric-mean variant
  ``T = sign(D) * sqrt(|D(omega_i) * D(omega_j)|)`` is also common
  (Pinkster) and gives slightly different cross-frequency weighting
  but the same low-mu limit when D is smooth.

Directional spreading
---------------------
Long-crested input is sufficient for a first cut. We follow the
parametric ``cqa.psd.slow_drift_force_psd_newman`` convention and
sample a wrapped-Gaussian directional spread ``N(theta_rel, sigma_th)``
of the wave field at one fixed frequency-grid evaluation, then form
the force outer product in the (surge, sway, yaw) frame. This makes
the force PSD matrix full-rank in 2-D plan and keeps the resulting
position covariance from collapsing to a line.

Production / prototype boundary
-------------------------------
* Drift coefficients ``D_i(omega, beta)`` are read once at vessel
  config load from pdstrip (csov_pdstrip.dat). Same source as the
  1st-order RAOs. Already exposed via
  ``cqa.rao.RaoTable.drift`` and interpolated by
  ``cqa.rao.evaluate_drift``.
* Wave spectrum ``S_eta`` and direction ``theta_wave_rel`` are NOT
  measured on board; they come from a wave-spectrum provider
  (forecast / radar / wind-sea analogy via
  ``hydro_tools/environment/wave_buoy.py``).

Limitations
-----------
* 19-column pdstrip files have no roll-drift coefficient
  (``has_roll_drift_data_`` is false in brucon). cqa ignores roll
  drift; for the SOV beam/GM combination that's a small effect
  relative to roll restoring stiffness.
* Newman approximation under-predicts slow-drift force at very low mu
  in narrow swell. For typical SOV operating wind-sea this is fine.
* Long-crested only (1-D spectrum); spreading is a heuristic
  perturbation to avoid singular force PSDs.
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

from .psd import jonswap_psd, wave_elevation_psd, WaveSpectrumKind
from .rao import RaoTable, evaluate_drift
from .sea_spreading import SeaSpreading, spreading_quadrature


# ---------------------------------------------------------------------------
# Default integration grid
# ---------------------------------------------------------------------------


def _default_omega_grid(table: RaoTable, n: int = 256) -> np.ndarray:
    """Linear grid spanning the pdstrip frequency range. Same convention
    as ``cqa.wave_response._default_omega_grid``."""
    return np.linspace(table.omega[0], table.omega[-1], n)


# ---------------------------------------------------------------------------
# 1. Mean drift force (Faltinsen [90] eq. 5.41)
# ---------------------------------------------------------------------------


def mean_drift_force_pdstrip(
    rao_table: RaoTable,
    Hs: float,
    Tp: float,
    theta_wave_rel: float,
    gamma: float = 3.3,
    n_omega: int = 256,
    spreading: SeaSpreading | None = None,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> np.ndarray:
    """Mean wave-drift force/moment in vessel body axes.

    Parameters
    ----------
    rao_table : pdstrip RAO table (must include populated ``drift``).
    Hs, Tp : sea-state significant wave height [m] and peak period [s].
    theta_wave_rel : MEAN wave direction relative to vessel,
        **cqa convention** (rad): 0 = head, +pi/2 = port beam,
        +pi = following.
    gamma : JONSWAP peak-enhancement factor. Ignored when
        ``spectrum == 'bretschneider'``.
    n_omega : number of points on the linear quadrature grid.
    spreading : directional-spreading model. Default: cos-2s, s=15
        (DNV-RP-C205 wind-sea typical, ~21 deg one-sigma equivalent).
        Pass ``SeaSpreading.long_crested()`` for the single-direction
        long-crested limit. Pass ``SeaSpreading.cos_n(2)`` to match
        brucon ``WaveSpectrum`` defaults.
    spectrum : wave-elevation PSD shape. Default
        ``'bretschneider'`` (IMCA / DNV-ST-0111 / brucon
        vessel_simulator default). Pass ``'jonswap'`` for the
        peakier DNV-RP-C205 spectrum (more conservative for QTF
        integrals when T_p is at or above the QTF peak).

    Returns
    -------
    (3,) float64 array (F_surge, F_sway, M_yaw) in vessel body axes.
        F_surge, F_sway in N
        M_yaw in N m

    Notes
    -----
    Long-crested integrand is Faltinsen [90] eq. 5.41,
    ``F_i = 2 * integral D_i(omega, beta) S_eta(omega) d omega``.
    Short-crested generalisation averages over the spreading function
    ``D(phi)`` about the mean direction:
        F_i = 2 * sum_k w_k * integral D_i(omega, beta_bar + phi_k)
                                   * S_eta(omega) d omega.
    Asymmetric drift coefficients across the spread mean a beam-sea
    drift force is reduced (energy spreads onto less effective
    directions); a head-sea symmetry preserved.
    """
    from .wave_response import cqa_theta_rel_to_pdstrip_beta_deg

    if spreading is None:
        spreading = SeaSpreading()  # cos-2s, s=15

    # Build directional quadrature in cqa-theta_rel space, then convert
    # each sample to its pdstrip beta. Spread is applied symmetrically
    # in cqa-theta_rel space; the conversion is a constant offset, so
    # that's equivalent to spreading directly in pdstrip-beta space
    # (the angular metric is the same).
    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)

    omega = _default_omega_grid(rao_table, n_omega)
    S_eta = wave_elevation_psd(omega, Hs, Tp, kind=spectrum, gamma=gamma)

    F = np.zeros(3, dtype=np.float64)
    for theta_k, w_k in zip(angles_rel, w_dir):
        beta_deg_k = cqa_theta_rel_to_pdstrip_beta_deg(theta_k)
        D = evaluate_drift(rao_table, omega, beta_deg_k)            # (n_omega, 3)
        F += 2.0 * w_k * np.trapezoid(D * S_eta[:, None], omega, axis=0)
    return F


# ---------------------------------------------------------------------------
# 2. Slow-drift force PSD (Newman, arithmetic-mean variant)
# ---------------------------------------------------------------------------


def slow_drift_force_psd_newman_pdstrip(
    rao_table: RaoTable,
    Hs: float,
    Tp: float,
    theta_wave_rel: float,
    gamma: float = 3.3,
    spreading: SeaSpreading | None = None,
    n_omega: int = 256,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a callable S_F(mu) -> (3, 3) slow-drift force PSD matrix.

    Parameters
    ----------
    rao_table, Hs, Tp, gamma : as :func:`mean_drift_force_pdstrip`.
    theta_wave_rel : mean wave direction (rad, cqa convention).
    spreading : directional-spreading model. Default: cos-2s, s=15
        (DNV-RP-C205 wind-sea typical, ~21 deg one-sigma equivalent).
        Pass ``SeaSpreading.long_crested()`` for the single-direction
        long-crested limit. Pass ``SeaSpreading.cos_n(2)`` to match
        brucon ``WaveSpectrum`` defaults.
    n_omega : quadrature points on the omega grid for the inner
        Newman integral.
    spectrum : wave-elevation PSD shape. Default ``'bretschneider'``
        (IMCA / DNV-ST-0111 / brucon vessel_simulator default). Pass
        ``'jonswap'`` for the peakier DNV-RP-C205 spectrum.

    Returns
    -------
    Callable mapping ``mu`` (rad/s, scalar or 1-D array) to a (3, 3)
    real-symmetric force PSD matrix in vessel body axes
    (surge, sway, yaw).

    Implementation
    --------------
    Newman approximation (arithmetic-mean variant, matches brucon
    wave_response.cpp line 333):
        T_i(omega1, omega2) ~ 0.5 * (D_i(omega1) + D_i(omega2))
        S_F,i(mu) = 8 * integral T_i(omega, omega+mu)^2
                                * S_eta(omega) S_eta(omega+mu) d omega

    For short-crested seas, the 2-D wave spectrum factors as
    ``S_eta(omega) D(phi)`` with D normalised to integrate to 1. Each
    direction sample contributes its own diagonal-QTF cross-spectrum;
    we accumulate the weighted sum
        S_F,ij(mu) = sum_k w_k * [8 integral T_i(omega+phi_k)
                                          * T_j(omega+phi_k)
                                          * S_a S_b d omega]
    which gives a full-rank 3x3 PSD.
    """
    from .wave_response import cqa_theta_rel_to_pdstrip_beta_deg

    if spreading is None:
        spreading = SeaSpreading()  # cos-2s, s=15

    # Build directional quadrature in cqa-theta_rel space, then convert
    # each sample to its pdstrip beta. Spread is symmetric in either
    # frame (same angular metric).
    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)
    beta_deg_dir = np.array(
        [cqa_theta_rel_to_pdstrip_beta_deg(a) for a in angles_rel]
    )
    n_dir = angles_rel.size

    omega_grid = _default_omega_grid(rao_table, n_omega)
    S_eta_grid = wave_elevation_psd(omega_grid, Hs, Tp, kind=spectrum, gamma=gamma)

    # Pre-evaluate D_i(omega, beta_dir) on the omega grid for each
    # direction sample. Shape (n_dir, n_omega, 3).
    D_dir = np.zeros((n_dir, omega_grid.size, 3), dtype=np.float64)
    for k in range(n_dir):
        D_dir[k] = evaluate_drift(rao_table, omega_grid, beta_deg_dir[k])

    omega_max = float(rao_table.omega[-1])

    def S_F(mu) -> np.ndarray:
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        n_mu = mu_arr.size
        out = np.zeros((n_mu, 3, 3), dtype=np.float64)

        for i_mu, mu_val in enumerate(mu_arr):
            # Build shifted-grid quantities.
            omega_shift = omega_grid + mu_val
            valid = omega_shift <= omega_max
            if not valid.any():
                continue
            o_grid = omega_grid[valid]
            o_shift = omega_shift[valid]
            S_a = S_eta_grid[valid]
            S_b = wave_elevation_psd(o_shift, Hs, Tp, kind=spectrum, gamma=gamma)
            spec_prod = S_a * S_b              # (n_valid,)

            G = np.zeros((3, 3), dtype=np.float64)
            for k, w in enumerate(w_dir):
                D_a = D_dir[k][valid]                                    # (n_valid, 3)
                D_b = evaluate_drift(
                    rao_table, o_shift, beta_deg_dir[k]
                )                                                        # (n_valid, 3)
                # Newman arithmetic-mean diagonal QTF for this direction:
                #   T_i(omega, omega+mu) = 0.5 * (D_i(omega) + D_i(omega+mu))
                T = 0.5 * (D_a + D_b)                                    # (n_valid, 3)

                # Cross-spectrum at mu, per direction:
                #   S_F,ij(mu) = 8 * integral T_i T_j * S_a * S_b  d omega
                # Build the 3x3 integrand and integrate along omega.
                # T has shape (n_valid, 3) so T_outer has shape
                # (n_valid, 3, 3) via einsum.
                T_outer = np.einsum("ki,kj->kij", T, T)                  # (n_valid, 3, 3)
                integrand = 8.0 * T_outer * spec_prod[:, None, None]
                G_dir = np.trapezoid(integrand, o_grid, axis=0)          # (3, 3)
                G += w * G_dir

            # Symmetrise to suppress numerical asymmetry from the
            # einsum / quadrature; the analytical S_F is symmetric.
            out[i_mu] = 0.5 * (G + G.T)

        if np.ndim(mu) == 0:
            return out[0]
        return out

    return S_F


__all__ = [
    "mean_drift_force_pdstrip",
    "slow_drift_force_psd_newman_pdstrip",
]
