"""
Wageningen C/D series propeller model.

Implements the Fourier-series based thrust and torque coefficient model
used by the Wageningen C-series (controllable pitch) propellers.

The model is parameterised by:
  - area_ratio:   the blade area ratio (Ae/A0), e.g. 0.55 or 0.70
  - design_pitch: the design P/D ratio (blade geometry)
  - pitch:        the actual operating P/D ratio (blade setting)
  - beta:         the advance angle = atan2(Va, 0.7 * pi * n * D)

CT and CQ are computed from Fourier series in beta, with coefficients
that depend on design_pitch and pitch.

Interpolation strategy:
  - Pitch axis: PCHIP (monotone cubic Hermite) on evaluated CT/CQ values.
    This gives C1-smooth results without overshoot, important for the
    unevenly spaced pitch table.
  - Design pitch axis: PCHIP interpolation between the (few) tabulated
    design pitch values.  The results are mildly sensitive to design pitch
    (typically 5-15%), so interpolation is worthwhile.
  - Blade area ratio axis: linear interpolation between tabulated BAR
    values (typically 0.55 and 0.70).  With only 2 points, PCHIP
    degenerates to linear; the structure supports additional BARs.

Reference: "Open-water test series with modified Wageningen B-screw
series propellers" and associated data publications.
"""

import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PropellerData:
    """Raw Fourier coefficient data for a C/D-series propeller."""

    design_pitches: np.ndarray       # sorted unique design P/D values
    pitches: np.ndarray              # sorted unique operating P/D values
    n_coefficients: int              # number of Fourier terms (k = 0..n-1)

    # Indexed as [design_idx][pitch_idx][k]
    ct_ak: np.ndarray  # shape (n_design, n_pitch, n_coeff)
    ct_bk: np.ndarray
    cq_ak: np.ndarray
    cq_bk: np.ndarray


def load_c_series_data(filepath: str | Path) -> PropellerData:
    """Load C-series Fourier coefficient data from a .dat file.

    Handles both C4-55 and C4-70 format files.  The file is tab-separated
    with columns:
        design  test  pitch  k  ct_bk  ct_ak  cq_bk  cq_ak

    Returns a PropellerData structure with coefficients organised
    as 3D arrays indexed by [design_index, pitch_index, k].
    """
    filepath = Path(filepath)

    designs = []
    pitches_list = []
    k_values = []
    ct_bk_raw = []
    ct_ak_raw = []
    cq_bk_raw = []
    cq_ak_raw = []

    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 8:
                continue
            designs.append(float(row[0]))
            # row[1] is test number, skip
            pitches_list.append(float(row[2]))
            k_values.append(int(row[3]))
            ct_bk_raw.append(float(row[4]))
            ct_ak_raw.append(float(row[5]))
            cq_bk_raw.append(float(row[6]))
            cq_ak_raw.append(float(row[7]))

    unique_designs = np.array(sorted(set(designs)))
    unique_pitches = np.array(sorted(set(pitches_list)))
    n_design = len(unique_designs)
    n_pitch = len(unique_pitches)
    n_coeff = max(k_values) + 1

    # Build index maps
    design_idx = {d: i for i, d in enumerate(unique_designs)}
    pitch_idx = {p: i for i, p in enumerate(unique_pitches)}

    ct_ak = np.zeros((n_design, n_pitch, n_coeff))
    ct_bk = np.zeros((n_design, n_pitch, n_coeff))
    cq_ak = np.zeros((n_design, n_pitch, n_coeff))
    cq_bk = np.zeros((n_design, n_pitch, n_coeff))

    for i in range(len(designs)):
        di = design_idx[designs[i]]
        pi = pitch_idx[pitches_list[i]]
        k = k_values[i]
        ct_ak[di, pi, k] = ct_ak_raw[i]
        ct_bk[di, pi, k] = ct_bk_raw[i]
        cq_ak[di, pi, k] = cq_ak_raw[i]
        cq_bk[di, pi, k] = cq_bk_raw[i]

    return PropellerData(
        design_pitches=unique_designs,
        pitches=unique_pitches,
        n_coefficients=n_coeff,
        ct_ak=ct_ak,
        ct_bk=ct_bk,
        cq_ak=cq_ak,
        cq_bk=cq_bk,
    )


# Backward-compatible alias
load_c4_55_data = load_c_series_data


# ====================================================================
# PCHIP interpolation (Fritsch-Carlson monotone cubic Hermite)
#
# Implemented from scratch so the logic is clear and portable to C++.
# scipy.interpolate.PchipInterpolator does the same thing but we want
# to understand and control the algorithm for eventual brucon porting.
# ====================================================================

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute PCHIP slopes (Fritsch-Carlson method).

    Given data points (x_i, y_i), returns slope d_i at each point
    such that the resulting piecewise cubic is monotone on each interval.

    Parameters
    ----------
    x : array, shape (n,)
        Strictly increasing abscissae.
    y : array, shape (n,)
        Ordinate values.

    Returns
    -------
    d : array, shape (n,)
        Slope at each data point.
    """
    n = len(x)
    d = np.zeros(n)

    # Secant slopes
    h = np.diff(x)
    delta = np.diff(y) / h

    if n == 2:
        d[0] = d[1] = delta[0]
        return d

    # Interior points: weighted harmonic mean of adjacent secants
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            # Same sign: weighted harmonic mean
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            # Different sign or zero: set slope to zero (local extremum)
            d[i] = 0.0

    # End points: one-sided shape-preserving formula
    d[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])

    return d


def _pchip_end_slope(h1: float, h2: float, d1: float, d2: float) -> float:
    """Non-centered three-point formula for end slopes, adjusted
    to preserve shape (Fritsch-Carlson).

    Parameters
    ----------
    h1 : float
        Width of the first (end) interval.
    h2 : float
        Width of the second interval.
    d1 : float
        Secant of the first interval.
    d2 : float
        Secant of the second interval.

    Returns
    -------
    slope : float
        Shape-preserving end slope.
    """
    s = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)

    # Ensure same sign as d1 (shape-preserving)
    if s * d1 <= 0.0:
        return 0.0

    # Ensure monotonicity
    if d1 * d2 <= 0.0 and abs(s) > 3.0 * abs(d1):
        return 3.0 * d1

    return s


def pchip_eval(x: np.ndarray, y: np.ndarray, d: np.ndarray, xi: float) -> float:
    """Evaluate PCHIP interpolant at a single point.

    Parameters
    ----------
    x : array, shape (n,)
        Tabulated abscissae (strictly increasing).
    y : array, shape (n,)
        Tabulated ordinates.
    d : array, shape (n,)
        Slopes at each data point (from _pchip_slopes).
    xi : float
        Point at which to evaluate.

    Returns
    -------
    yi : float
        Interpolated value.
    """
    n = len(x)

    # Extrapolate linearly outside the data range using the end slopes
    if xi <= x[0]:
        return float(y[0] + d[0] * (xi - x[0]))
    if xi >= x[-1]:
        return float(y[-1] + d[-1] * (xi - x[-1]))

    # Binary search for interval
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] > xi:
            hi = mid
        else:
            lo = mid

    # Hermite basis on [x[lo], x[hi]]
    h = x[hi] - x[lo]
    t = (xi - x[lo]) / h

    # Cubic Hermite polynomials
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t ** 2 * (3.0 - 2.0 * t)
    h11 = t ** 2 * (t - 1.0)

    return float(h00 * y[lo] + h10 * h * d[lo] + h01 * y[hi] + h11 * h * d[hi])


class PchipInterpolator:
    """Lightweight PCHIP interpolator for a single 1D dataset.

    Portable implementation suitable for later C++ translation.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._d = _pchip_slopes(self._x, self._y)

    def __call__(self, xi: float) -> float:
        return pchip_eval(self._x, self._y, self._d, xi)


class _SingleBarModel:
    """Internal model for a single blade area ratio.

    Handles design pitch and operating pitch interpolation via PCHIP.
    Not intended for direct use -- use CSeriesPropeller instead.
    """

    def __init__(self, data: PropellerData, design_pitch: float):
        self._data = data
        self._design_pitch = design_pitch

        # Resolve design pitch: find bracketing indices
        # With 2+ design pitches, always use PCHIP (allowing extrapolation
        # via the end slopes for values outside the tabulated range).
        dp_table = data.design_pitches
        n_dp = len(dp_table)
        if n_dp == 1:
            # Only one design pitch available — no interpolation possible
            self._dp_lo = 0
            self._dp_hi = 0
        elif design_pitch <= dp_table[0]:
            # Below range: use first two for PCHIP extrapolation
            self._dp_lo = 0
            self._dp_hi = 1
        elif design_pitch >= dp_table[-1]:
            # Above range: use last two for PCHIP extrapolation
            self._dp_lo = n_dp - 2
            self._dp_hi = n_dp - 1
        else:
            idx = int(np.searchsorted(dp_table, design_pitch, side="right"))
            self._dp_lo = idx - 1
            self._dp_hi = idx

    @property
    def pitch_table(self) -> np.ndarray:
        return self._data.pitches

    @staticmethod
    def _fourier_interpolate(A: np.ndarray, B: np.ndarray, beta: float) -> float:
        """Evaluate Fourier series: sum_k [ A_k sin(k*beta) + B_k cos(k*beta) ]."""
        k = np.arange(len(A))
        return float(np.sum(A * np.sin(k * beta) + B * np.cos(k * beta)))

    def _eval_at_design_pitch(
        self, coeff_ak: np.ndarray, coeff_bk: np.ndarray, pitch_idx: int, beta: float
    ) -> float:
        """Evaluate Fourier series at a specific pitch index, interpolated
        across the design pitch axis."""
        if self._dp_lo == self._dp_hi:
            return self._fourier_interpolate(
                coeff_ak[self._dp_lo, pitch_idx],
                coeff_bk[self._dp_lo, pitch_idx],
                beta,
            )

        dp_table = self._data.design_pitches
        n_dp = len(dp_table)
        vals = np.empty(n_dp)
        for di in range(n_dp):
            vals[di] = self._fourier_interpolate(
                coeff_ak[di, pitch_idx], coeff_bk[di, pitch_idx], beta
            )

        interp = PchipInterpolator(dp_table, vals)
        return interp(self._design_pitch)

    def eval_coefficient(
        self, coeff_ak: np.ndarray, coeff_bk: np.ndarray, pitch: float, beta: float
    ) -> float:
        """Evaluate CT or CQ at arbitrary (design_pitch, pitch, beta).

        Uses PCHIP interpolation on both the pitch and design pitch axes.
        """
        pitches = self._data.pitches
        n_pitch = len(pitches)

        vals = np.empty(n_pitch)
        for pi in range(n_pitch):
            vals[pi] = self._eval_at_design_pitch(coeff_ak, coeff_bk, pi, beta)

        interp = PchipInterpolator(pitches, vals)
        return interp(pitch)

    def CT(self, pitch: float, beta: float) -> float:
        return self.eval_coefficient(self._data.ct_ak, self._data.ct_bk, pitch, beta)

    def CQ(self, pitch: float, beta: float) -> float:
        return self.eval_coefficient(self._data.cq_ak, self._data.cq_bk, pitch, beta)


class CSeriesPropeller:
    """Wageningen C-series controllable-pitch propeller model.

    Supports interpolation across blade area ratio (BAR), design pitch,
    operating pitch, and advance angle.

    Parameters
    ----------
    data : PropellerData or dict[float, PropellerData]
        Fourier coefficient data.  Either:
        - A single PropellerData (single BAR, backward compatible), or
        - A dict mapping BAR values to PropellerData, e.g.
          {0.55: data_55, 0.70: data_70}.
    design_pitch : float
        Design pitch ratio P/D. Interpolated between tabulated values.
    diameter : float
        Propeller diameter [m].
    area_ratio : float, optional
        Blade area ratio Ae/A0.  Required if data is a dict.
        If data is a single PropellerData, this is informational only.
    rho : float
        Water density [kg/m^3], default 1025.
    """

    def __init__(
        self,
        data: PropellerData | dict[float, PropellerData],
        design_pitch: float,
        diameter: float,
        area_ratio: Optional[float] = None,
        rho: float = 1025.0,
    ):
        self._diameter = diameter
        self._rho = rho
        self._design_pitch = design_pitch
        self._area_ratio = area_ratio

        if isinstance(data, dict):
            # Multi-BAR mode
            if area_ratio is None:
                raise ValueError("area_ratio is required when data is a dict of BAR datasets")
            self._bar_table = np.array(sorted(data.keys()))
            self._bar_models = {
                bar: _SingleBarModel(data[bar], design_pitch)
                for bar in self._bar_table
            }
            # Use the union of all pitch tables for the sweep range
            all_pitches = set()
            for d in data.values():
                all_pitches.update(d.pitches.tolist())
            self._pitch_table = np.array(sorted(all_pitches))
        else:
            # Single-BAR mode (backward compatible)
            self._bar_table = None
            self._bar_models = None
            self._single_model = _SingleBarModel(data, design_pitch)
            self._pitch_table = data.pitches

    @property
    def diameter(self) -> float:
        return self._diameter

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def design_pitch(self) -> float:
        return self._design_pitch

    @property
    def area_ratio(self) -> Optional[float]:
        return self._area_ratio

    @property
    def pitch_table(self) -> np.ndarray:
        """Available tabulated pitch ratios (union of all BARs if multi-BAR)."""
        return self._pitch_table

    # ----------------------------------------------------------------
    # Core hydrodynamic calculations
    # ----------------------------------------------------------------

    @staticmethod
    def beta(Va: float, n: float, D: float) -> float:
        """Advance angle [rad].

        beta = atan2(Va, 0.7 * pi * n * D)
        """
        return math.atan2(Va, 0.7 * math.pi * n * D)

    @staticmethod
    def Vr(Va: float, n: float, D: float) -> float:
        """Resultant velocity at 0.7R [m/s]."""
        return math.hypot(Va, 0.7 * math.pi * n * D)

    def _interpolate_bar(self, pitch: float, beta: float, coeff_name: str) -> float:
        """Evaluate CT or CQ with BAR interpolation.

        Parameters
        ----------
        pitch : float
            Operating pitch ratio.
        beta : float
            Advance angle [rad].
        coeff_name : str
            'CT' or 'CQ'.
        """
        if self._bar_models is None:
            # Single-BAR mode
            if coeff_name == 'CT':
                return self._single_model.CT(pitch, beta)
            else:
                return self._single_model.CQ(pitch, beta)

        # Multi-BAR mode: evaluate at each tabulated BAR, then interpolate
        bar_table = self._bar_table
        n_bar = len(bar_table)
        vals = np.empty(n_bar)
        for i, bar in enumerate(bar_table):
            model = self._bar_models[bar]
            if coeff_name == 'CT':
                vals[i] = model.CT(pitch, beta)
            else:
                vals[i] = model.CQ(pitch, beta)

        if n_bar == 1:
            return float(vals[0])
        elif n_bar == 2:
            # Linear interpolation / extrapolation
            # With only 2 BAR points we use linear inter/extrapolation.
            # Extrapolation outside the table range is allowed but should
            # be used cautiously (e.g. BAR=0.432 from 0.55/0.70 data).
            bar_lo, bar_hi = bar_table[0], bar_table[1]
            w = (self._area_ratio - bar_lo) / (bar_hi - bar_lo)
            return float(vals[0] * (1.0 - w) + vals[1] * w)
        else:
            # PCHIP across BAR for 3+ points
            interp = PchipInterpolator(bar_table, vals)
            return interp(self._area_ratio)

    def CT(self, pitch: float, beta: float) -> float:
        """Thrust loading coefficient CT at given pitch ratio and advance angle."""
        return self._interpolate_bar(pitch, beta, 'CT')

    def CQ(self, pitch: float, beta: float) -> float:
        """Torque loading coefficient CQ at given pitch ratio and advance angle."""
        return self._interpolate_bar(pitch, beta, 'CQ')

    def thrust(self, pitch: float, n: float, Va: float) -> float:
        """Propeller thrust [N].

        Parameters
        ----------
        pitch : float
            Operating pitch ratio P/D.
        n : float
            Shaft speed [rev/s] (NOT rpm).
        Va : float
            Advance velocity [m/s].
        """
        b = self.beta(Va, n, self._diameter)
        vr = self.Vr(Va, n, self._diameter)
        ct = self.CT(pitch, b)
        return ct * 0.5 * self._rho * vr**2 * (math.pi / 4.0) * self._diameter**2

    def torque(self, pitch: float, n: float, Va: float) -> float:
        """Propeller torque [Nm].

        Parameters
        ----------
        pitch : float
            Operating pitch ratio P/D.
        n : float
            Shaft speed [rev/s].
        Va : float
            Advance velocity [m/s].
        """
        b = self.beta(Va, n, self._diameter)
        vr = self.Vr(Va, n, self._diameter)
        cq = self.CQ(pitch, b)
        return cq * 0.5 * self._rho * vr**2 * (math.pi / 4.0) * self._diameter**3

    def power(self, pitch: float, n: float, Va: float) -> float:
        """Shaft power [W] = torque * 2*pi*n.

        Parameters
        ----------
        pitch : float
            Operating pitch ratio P/D.
        n : float
            Shaft speed [rev/s].
        Va : float
            Advance velocity [m/s].
        """
        return self.torque(pitch, n, Va) * 2.0 * math.pi * n

    # ----------------------------------------------------------------
    # KT / KQ (non-dimensional coefficients)
    # ----------------------------------------------------------------

    def KT(self, T: float, n: float) -> float:
        """Thrust coefficient KT = T / (rho * n^2 * D^4)."""
        return T / (self._rho * n**2 * self._diameter**4)

    def KQ(self, Q: float, n: float) -> float:
        """Torque coefficient KQ = Q / (rho * n^2 * D^5)."""
        return Q / (self._rho * n**2 * self._diameter**5)

    def eta0(self, pitch: float, n: float, Va: float) -> float:
        """Open-water efficiency eta_0 = T * Va / (2*pi*n*Q).

        Returns 0 if torque is zero or negative advance speed.
        """
        T = self.thrust(pitch, n, Va)
        Q = self.torque(pitch, n, Va)
        if abs(Q) < 1e-12 or abs(n) < 1e-12:
            return 0.0
        return T * Va / (2.0 * math.pi * n * Q)
