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
    Typical values: s = 10-25 for wind sea, s = 25-75 for swell.

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
    kind : "cos_2s" | "gaussian"
    s : cosine-2s exponent (used when kind == "cos_2s"). Default 15
        (DNV-RP-C205 wind-sea typical; ~21 deg one-sigma equivalent).
    sigma_deg : wrapped-Gaussian one-sigma (used when kind ==
        "gaussian").
    n_dir : number of quadrature angles. 31 is enough for trapezoidal
        accuracy at the 1 % level on smooth integrands.
    """

    kind: Literal["cos_2s", "gaussian"] = "cos_2s"
    s: float = 15.0
    sigma_deg: float = 25.0
    n_dir: int = 31

    @classmethod
    def long_crested(cls) -> "SeaSpreading":
        """Single-direction "spreading": one sample, weight 1.

        Use this to recover the long-crested limit deterministically.
        """
        return cls(kind="gaussian", sigma_deg=0.0, n_dir=1)


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
