"""Directional spreading models for short-crested wave fields.

A short-crested sea spreads its 2-D wave-elevation spectrum across
directions about a mean direction theta_bar:

    S_eta(omega, theta) = S_eta_long(omega) * D(theta - theta_bar)

with the spreading function D normalised so that
``integral_{-pi}^{pi} D(phi) d phi = 1``.

Two standard models are provided:

  * **cos-2s** (DNV-RP-C205, Mitsuyasu 1975):
        D(phi; s) = (Gamma(s+1)^2 / (2 pi Gamma(2s+1))) * 2^(2s-1) * cos(phi/2)^(2s)
    for |phi| <= pi, zero elsewhere. Larger ``s`` => narrower spread.
    Typical wind-sea range: s = 2 - 10 (Mitsuyasu et al. 1975, DNV
    RP-C205 Sec. 3.5.8.4). The cqa default s=4 is chosen to match
    brucon's WaveSpectrum default of cos^n(delta) with n=2 in the
    narrow-spread Gaussian-width sense (s ~= 2 n), which is also
    consistent with DNV-ST-0111 wind-sea practice.

  * **Wrapped Gaussian** (parametric variance):
        D(phi; sigma) ~ N(0, sigma^2), wrapped to [-pi, pi].
    More convenient for analytical estimates of variance reduction.

For variance computation in linear/quadratic wave-response analyses
we discretise D(phi) on a fixed angular grid and accumulate a
weighted sum. The grid is chosen to span +/- pi for cos-2s (which
is naturally bounded) and +/- 3 sigma for the wrapped Gaussian.

References
----------
* DNV-RP-C205 (Environmental conditions and environmental loads),
  Sec. 3.5.8.4 (Cosine-2s).
* Faltinsen (1990), "Sea Loads on Ships and Offshore Structures",
  ch. 2 for spreading conventions used in 2nd-order force theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Literal, Tuple
import numpy as np


@dataclass(frozen=True)
class SeaSpreading:
    """Directional spreading of a wave field.

    Attributes
    ----------
    kind : "cos_2s" | "gaussian" | "cos_n"
    s : cosine-2s exponent (used when kind == "cos_2s"). Default 4
        (DNV-RP-C205 wind-sea range s in 2-10; chosen to match brucon
        WaveSpectrum default cos^n(delta) n=2 via the narrow-spread
        Gaussian-width equivalence s ~= 2 n). Force-level cross-
        validation against brucon at the P7 operating point shows mean
        and std drift force matching to ~6 percent at this spreading.
    sigma_deg : wrapped-Gaussian one-sigma (used when kind ==
        "gaussian").
    n : cos^n exponent over (-pi/2, +pi/2) (used when kind == "cos_n").
        This is the **brucon** wave_spectrum convention (see
        ``vessel_simulator_wrapper.cpp`` -- ``WaveSpectrum`` ctor's
        4th argument ``spreading_factor``). brucon's default for the
        CSOV simulator is n = 2 (i.e. cos^2). The Gaussian-limit
        equivalence with cos-2s is ``s ~= 2 n``: brucon cos^2 has the
        same one-sigma width as cqa cos_2s s=4.
    n_dir : number of quadrature angles. 31 is enough for trapezoidal
        accuracy at the 1 % level on smooth integrands.
    """

    kind: Literal["cos_2s", "gaussian", "cos_n"] = "cos_2s"
    s: float = 4.0
    sigma_deg: float = 25.0
    n: float = 2.0
    n_dir: int = 31

    @classmethod
    def long_crested(cls) -> "SeaSpreading":
        """Single-direction "spreading": one sample, weight 1.

        Use this to recover the long-crested limit deterministically.
        """
        return cls(kind="gaussian", sigma_deg=0.0, n_dir=1)

    @classmethod
    def cos_n(cls, n: float = 2.0, n_dir: int = 31) -> "SeaSpreading":
        """brucon-style ``cos^n(delta)`` spreading over (-pi/2, +pi/2).

        Use this constructor to match brucon ``WaveSpectrum`` behaviour
        bit-for-bit. brucon's CSOV simulator default is n = 2 (cos^2).

        Equivalence with the cqa default ``cos_2s``: in the narrow-spread
        Gaussian limit both functions give the same one-sigma width when
        ``s ~= 2 n``. So cos^2 (brucon default) approximates cos_2s s=4,
        which is also the cqa default. ``SeaSpreading()`` and
        ``SeaSpreading.cos_n(2)`` therefore give force levels matching
        brucon to within sampling noise (verified at the P7 operating
        point: mean and std drift force agree to ~6 percent).
        """
        return cls(kind="cos_n", n=float(n), n_dir=int(n_dir))


def cos_2s_norm_const(s: float) -> float:
    """Normalisation constant ``c(s)`` in
    D(phi; s) = c(s) * cos(phi/2)^(2s).

    From DNV-RP-C205 / Mitsuyasu:
        c(s) = 2^(2s-1) / pi  *  Gamma(s+1)^2 / Gamma(2s+1)

    Implemented via lgamma to stay numerically stable for s up to ~75.
    """
    log_c = (2.0 * s - 1.0) * np.log(2.0) - np.log(np.pi)
    log_c += 2.0 * lgamma(s + 1.0) - lgamma(2.0 * s + 1.0)
    return float(np.exp(log_c))


def spreading_quadrature(
    spreading: SeaSpreading,
    theta_bar_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a (angles_rad, weights) quadrature for the spreading.

    Returns
    -------
    angles_rad : (n_dir,) float, angles in **radians** about
        ``theta_bar_rad`` covering the support of D.
    weights : (n_dir,) float, sum to 1. Suitable for
        ``sum_k w_k f(angles[k])`` quadrature of expectations
        ``E_D[f] = integral D(phi) f(theta_bar + phi) d phi``.

    Notes
    -----
    * For ``cos_2s`` the support is +/- pi about theta_bar; we use a
      uniform angular grid across that interval and trapezoidal
      weights, then renormalise so the discrete sum hits 1 exactly
      (compensates for the sub-percent discretisation error in the
      Gamma normalisation).
    * For ``gaussian`` we sample +/- 3 sigma uniformly and use
      Gaussian weights, renormalised to sum to 1.
    * Long-crested limit (n_dir == 1 or sigma==0): single sample at
      ``theta_bar_rad`` with weight 1.
    """
    # Long-crested degenerate cases.
    if spreading.n_dir <= 1:
        return np.array([theta_bar_rad]), np.array([1.0])
    if spreading.kind == "gaussian" and spreading.sigma_deg <= 0.0:
        return np.array([theta_bar_rad]), np.array([1.0])

    n = int(spreading.n_dir)

    if spreading.kind == "cos_2s":
        s = float(spreading.s)
        phi = np.linspace(-np.pi, np.pi, n)
        # D(phi) = c(s) * cos(phi/2)^(2s). cos(+/- pi/2) = 0 cleanly.
        c = cos_2s_norm_const(s)
        D = c * np.cos(phi / 2.0) ** (2.0 * s)
        # Trapezoidal weight to integrate D(phi) d phi.
        dphi = phi[1] - phi[0]
        trap = np.full(n, dphi)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        w = D * trap
    elif spreading.kind == "gaussian":
        sigma = np.radians(spreading.sigma_deg)
        phi = np.linspace(-3.0 * sigma, 3.0 * sigma, n)
        D = np.exp(-(phi ** 2) / (2.0 * sigma ** 2))
        w = D
    elif spreading.kind == "cos_n":
        # brucon convention: D(phi) ~ cos^n(phi) for |phi| <= pi/2,
        # zero outside. The normalisation constant cancels under the
        # final renormalise-to-1 step below; we only need the shape.
        phi = np.linspace(-np.pi / 2.0, np.pi / 2.0, n)
        # cos at +/- pi/2 is exactly 0 -- safe to raise to a positive power.
        D = np.cos(phi) ** float(spreading.n)
        dphi = phi[1] - phi[0]
        trap = np.full(n, dphi)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        w = D * trap
    else:
        raise ValueError(f"Unknown spreading kind: {spreading.kind!r}")

    # Force the discrete weights to sum to 1 (small <1% renormalisation).
    s_w = w.sum()
    if s_w <= 0:
        raise ValueError(
            f"Spreading quadrature weights collapsed to zero "
            f"(kind={spreading.kind}, s={spreading.s}, sigma_deg={spreading.sigma_deg})"
        )
    w = w / s_w
    angles = theta_bar_rad + phi
    return angles, w


__all__ = [
    "SeaSpreading",
    "cos_2s_norm_const",
    "spreading_quadrature",
]
