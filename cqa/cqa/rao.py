"""RAO loader and 6-DOF interpolator for pdstrip .dat exports.

Loads a brucon-style pdstrip ``.dat`` file (tab-separated, columns
``freq enc angle speed surge_r surge_i sway_r sway_i heave_r heave_i
roll_r roll_i pitch_r pitch_i yaw_r yaw_i surge_d sway_d yaw_d``) and
exposes a complex 6-DOF transfer function ``H(omega, beta)`` returning
the per-metre wave-amplitude response

    [surge, sway, heave, roll, pitch, yaw]

with translations in m/m and rotations in rad/m, evaluated at zero
forward speed (DP operation).

Conventions
-----------
* ``omega`` is wave angular frequency in rad/s. At zero speed encounter
  frequency equals wave frequency; we filter on ``speed == 0`` rows.
* ``beta`` is the wave direction in degrees in the pdstrip convention:
  180° = head sea, 0° = following, +90° = beam from port (wave going to
  starboard), -90° = beam from starboard. Look-up is wrapped to
  [0, 360).
* Below the lowest tabulated frequency the RAOs are extrapolated as
  zero (mirroring brucon's "outside valid range" handling for
  ``enc < 0.2 rad/s``); above the highest, also zero (high-frequency
  components carry vanishing energy in realistic spectra).

Production / prototype boundary
-------------------------------
In the real DP system:

* Wind, currents, vessel position/heading and thruster forces are
  measured or estimated.
* The wave **spectrum** is NOT directly measured. It comes from
  forecast (NORA3 / WW3), ship-mounted wave radar, or a wind-sea
  analogy via ``hydro_tools/environment/wave_buoy.py``.
* The 1st-order wave-frequency motion at the gangway base is computed
  on-the-fly from RAO × spectrum, not read from MRU.

This module is therefore the natural place where the C++ port will
plug in a ``WaveSpectrumProvider`` interface alongside the RAO data
that ships with the vessel config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Encounter frequency below which pdstrip results are unreliable;
# matches brucon::dp::ResponseFunctions::FillResponseFunctionVectors.
PDSTRIP_MIN_ENC_FREQ_RAD_S = 0.2


@dataclass(frozen=True)
class RaoTable:
    """Tabulated complex RAOs at zero forward speed, RAW pdstrip values.

    Attributes
    ----------
    omega : (n_omega,) float64, strictly increasing rad/s.
    beta_deg : (n_beta,) float64, strictly increasing degrees in
        [0, 360) after wrapping.
    H : (n_omega, n_beta, 6) complex128 -- RAW pdstrip column values.
        Order: [surge, sway, heave, roll, pitch, yaw].

        IMPORTANT: pdstrip stores the rotational columns (roll, pitch,
        yaw) per unit wave **slope**, not per unit wave **amplitude**.
        To get the physical RAO in rad per metre of wave amplitude, the
        rotational columns must be multiplied by the wavenumber
        k = omega^2 / g (deep water). The translational columns
        (surge, sway, heave) are already in m per metre of wave
        amplitude and need no scaling. ``evaluate_rao()`` applies the
        wavenumber factor for you; if you read ``H`` directly, you
        must apply it yourself.

        This convention matches brucon's
        ``WaveResponse::CalculateLinearResponse`` which applies
        ``angle_factor = k`` for ``dof > 3``.

    drift : (n_omega, n_beta, 3) float64 -- mean drift force/moment
        coefficients [surge_d, sway_d, yaw_d] as reported by pdstrip
        (units N/m^2 and Nm/m^2 respectively, per unit wave amplitude
        squared). These are physical and need no wavenumber scaling.
    source_path : original file path (informational).
    """

    omega: np.ndarray
    beta_deg: np.ndarray
    H: np.ndarray
    drift: np.ndarray
    source_path: Optional[Path] = None

    @property
    def n_omega(self) -> int:
        return int(self.omega.size)

    @property
    def n_beta(self) -> int:
        return int(self.beta_deg.size)


# Standard gravity used for the deep-water dispersion k = omega^2 / g.
# Matches brucon::util::Math::g.
G_STD = 9.80665


_DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_PDSTRIP_COLUMNS = (
    "freq", "enc", "angle", "speed",
    "surge_r", "surge_i", "sway_r", "sway_i",
    "heave_r", "heave_i", "roll_r", "roll_i",
    "pitch_r", "pitch_i", "yaw_r", "yaw_i",
    "surge_d", "sway_d", "yaw_d",
)


def load_pdstrip_rao(path: str | Path, speed: float = 0.0) -> RaoTable:
    """Parse a pdstrip ``.dat`` file and build a 6-DOF RAO table.

    Parameters
    ----------
    path : path to the ``.dat`` file (e.g.
        ``$BRUCON/build/bin/vessel_simulator_config/csov_pdstrip.dat``).
    speed : forward-speed row to extract, m/s. Default 0.0 (DP).

    Returns
    -------
    RaoTable

    Notes
    -----
    * The file may contain a leading "speed = -1.0" block which brucon
      uses as a placeholder; we ignore it by filtering on the requested
      speed value with a 1e-3 m/s tolerance.
    * Frequencies below ``PDSTRIP_MIN_ENC_FREQ_RAD_S`` are zeroed (same
      rule brucon applies on encounter frequency).
    """
    path = Path(path)
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] != len(_PDSTRIP_COLUMNS):
        raise ValueError(
            f"Unexpected pdstrip column count: got {data.shape[1]}, "
            f"expected {len(_PDSTRIP_COLUMNS)}"
        )

    speed_col = data[:, 3]
    mask = np.isclose(speed_col, speed, atol=1e-3)
    if not mask.any():
        speeds_present = np.unique(speed_col)
        raise ValueError(
            f"No rows at speed={speed} m/s in {path}; available speeds: "
            f"{speeds_present.tolist()}"
        )
    sub = data[mask]

    omega_all = sub[:, 0]
    angle_all = sub[:, 2]

    omega = np.unique(np.round(omega_all, 6))
    omega.sort()
    beta = np.unique(np.round(angle_all, 6))
    # Wrap to [0, 360) and re-sort for deterministic interpolation.
    beta_wrapped = np.mod(beta, 360.0)
    order = np.argsort(beta_wrapped)
    beta_sorted = beta_wrapped[order]
    if beta_sorted.size != np.unique(beta_sorted).size:
        raise ValueError(
            f"pdstrip angle grid has duplicates after wrapping to [0,360): "
            f"{beta_sorted}"
        )

    n_omega = omega.size
    n_beta = beta_sorted.size
    if sub.shape[0] != n_omega * n_beta:
        raise ValueError(
            f"pdstrip rows ({sub.shape[0]}) at speed={speed} do not form a "
            f"complete {n_omega}x{n_beta} (omega,beta) grid"
        )

    H = np.zeros((n_omega, n_beta, 6), dtype=np.complex128)
    drift = np.zeros((n_omega, n_beta, 3), dtype=np.float64)

    # Index lookup (use integer keys via rounding to match the unique() above).
    omega_idx = {float(v): i for i, v in enumerate(omega)}
    beta_idx = {float(v): i for i, v in enumerate(beta_sorted)}

    for row in sub:
        i_omega = omega_idx[float(np.round(row[0], 6))]
        b_wrapped = float(np.round(np.mod(row[2], 360.0), 6))
        j_beta = beta_idx[b_wrapped]
        # Real/imag pairs: cols 4..15 → 6 complex DOFs.
        complex_block = row[4:16].reshape(6, 2)
        H[i_omega, j_beta, :] = complex_block[:, 0] + 1j * complex_block[:, 1]
        drift[i_omega, j_beta, :] = row[16:19]

    # Zero out frequencies below pdstrip's valid range.
    invalid = omega < PDSTRIP_MIN_ENC_FREQ_RAD_S
    if invalid.any():
        H[invalid, :, :] = 0.0
        drift[invalid, :, :] = 0.0

    return RaoTable(omega=omega, beta_deg=beta_sorted, H=H, drift=drift, source_path=path)


def evaluate_rao(table: RaoTable, omega: np.ndarray, beta_deg: float) -> np.ndarray:
    """Bilinear interpolation of the complex RAOs, returned in PHYSICAL units.

    Parameters
    ----------
    table : RaoTable.
    omega : (n,) array of wave angular frequencies in rad/s.
    beta_deg : wave direction in degrees (pdstrip convention). Wrapped
        to [0, 360) before look-up.

    Returns
    -------
    (n, 6) complex128 array, in DOF order [surge, sway, heave, roll,
    pitch, yaw], in physical per-wave-amplitude units:
        translations: m / m  (m of motion per m of wave amplitude)
        rotations:    rad / m

    Frequencies outside the tabulated range return 0.

    Notes
    -----
    * The rotational DOF columns in pdstrip are stored per unit wave
      *slope*, so we multiply them by the deep-water wavenumber
      k = omega^2 / g to convert to per unit wave amplitude. Mirrors
      brucon's WaveResponse::CalculateLinearResponse (``angle_factor = k``
      for ``dof > 3``).
    * Real and imaginary parts are interpolated separately. Standard
      practice for narrowly-spaced RAO grids and matches what brucon
      does internally.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.zeros((omega.size, 6), dtype=np.complex128)

    # Beta index (linear, with wrap).
    beta_q = float(np.mod(beta_deg, 360.0))
    bgrid = table.beta_deg
    if beta_q <= bgrid[0] or beta_q >= bgrid[-1]:
        # Wrap-around between bgrid[-1] and bgrid[0]+360.
        b_lo = bgrid[-1]
        b_hi = bgrid[0] + 360.0
        b_q_eff = beta_q if beta_q >= bgrid[-1] else beta_q + 360.0
        wb = (b_q_eff - b_lo) / (b_hi - b_lo)
        j_lo = bgrid.size - 1
        j_hi = 0
    else:
        j_hi = int(np.searchsorted(bgrid, beta_q))
        j_lo = j_hi - 1
        wb = (beta_q - bgrid[j_lo]) / (bgrid[j_hi] - bgrid[j_lo])

    # Per-frequency interp (vectorised over omega).
    ogrid = table.omega
    # Mark out-of-range frequencies.
    in_range = (omega >= ogrid[0]) & (omega <= ogrid[-1])
    if not in_range.any():
        return out

    omega_in = omega[in_range]
    i_hi = np.searchsorted(ogrid, omega_in)
    # Clamp upper edge: omega == ogrid[-1] gives i_hi == n; treat as last segment.
    i_hi = np.clip(i_hi, 1, ogrid.size - 1)
    i_lo = i_hi - 1
    wo = (omega_in - ogrid[i_lo]) / (ogrid[i_hi] - ogrid[i_lo])

    H = table.H  # (n_omega, n_beta, 6)
    H00 = H[i_lo, j_lo, :]
    H01 = H[i_lo, j_hi, :]
    H10 = H[i_hi, j_lo, :]
    H11 = H[i_hi, j_hi, :]
    interp = (
        (1.0 - wo[:, None]) * (1.0 - wb) * H00
        + (1.0 - wo[:, None]) * wb * H01
        + wo[:, None] * (1.0 - wb) * H10
        + wo[:, None] * wb * H11
    )

    # Convert rotational DOFs from per-wave-slope to per-wave-amplitude
    # by multiplying by the deep-water wavenumber k = omega^2 / g.
    # Translational DOFs (cols 0,1,2) are already per-wave-amplitude.
    k_in = (omega_in ** 2) / G_STD
    interp[:, 3] *= k_in
    interp[:, 4] *= k_in
    interp[:, 5] *= k_in

    out[in_range, :] = interp
    return out


def evaluate_drift(table: RaoTable, omega: np.ndarray, beta_deg: float) -> np.ndarray:
    """Bilinear interpolation of pdstrip mean-drift coefficients.

    Parameters
    ----------
    table : RaoTable.
    omega : (n,) array of wave angular frequencies in rad/s.
    beta_deg : wave direction in degrees (pdstrip convention). Wrapped
        to [0, 360) before look-up, identical wrap rule to
        :func:`evaluate_rao`.

    Returns
    -------
    (n, 3) float64 array, in DOF order [surge_d, sway_d, yaw_d], in
    physical per-(wave-amplitude)^2 units:
        surge_d, sway_d : N / m^2
        yaw_d           : N m / m^2

    Frequencies outside the tabulated range return 0.

    Notes
    -----
    * Drift coefficients in pdstrip are physical (Faltinsen eq. 5.41,
      diagonal QTF) and need NO wavenumber correction. They are signed
      and oriented in the vessel body frame.
    * 19-column pdstrip files have no roll-drift coefficient. cqa
      currently ignores roll drift; for the SOV beam/GM combination it
      sits well below the restoring stiffness.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.zeros((omega.size, 3), dtype=np.float64)

    # Beta index (linear, with wrap) -- identical logic to evaluate_rao
    # so wrap-around behaviour matches.
    beta_q = float(np.mod(beta_deg, 360.0))
    bgrid = table.beta_deg
    if beta_q <= bgrid[0] or beta_q >= bgrid[-1]:
        b_lo = bgrid[-1]
        b_hi = bgrid[0] + 360.0
        b_q_eff = beta_q if beta_q >= bgrid[-1] else beta_q + 360.0
        wb = (b_q_eff - b_lo) / (b_hi - b_lo)
        j_lo = bgrid.size - 1
        j_hi = 0
    else:
        j_hi = int(np.searchsorted(bgrid, beta_q))
        j_lo = j_hi - 1
        wb = (beta_q - bgrid[j_lo]) / (bgrid[j_hi] - bgrid[j_lo])

    ogrid = table.omega
    in_range = (omega >= ogrid[0]) & (omega <= ogrid[-1])
    if not in_range.any():
        return out

    omega_in = omega[in_range]
    i_hi = np.searchsorted(ogrid, omega_in)
    i_hi = np.clip(i_hi, 1, ogrid.size - 1)
    i_lo = i_hi - 1
    wo = (omega_in - ogrid[i_lo]) / (ogrid[i_hi] - ogrid[i_lo])

    D = table.drift  # (n_omega, n_beta, 3)
    D00 = D[i_lo, j_lo, :]
    D01 = D[i_lo, j_hi, :]
    D10 = D[i_hi, j_lo, :]
    D11 = D[i_hi, j_hi, :]
    interp = (
        (1.0 - wo[:, None]) * (1.0 - wb) * D00
        + (1.0 - wo[:, None]) * wb * D01
        + wo[:, None] * (1.0 - wb) * D10
        + wo[:, None] * wb * D11
    )

    out[in_range, :] = interp
    return out


__all__ = [
    "PDSTRIP_MIN_ENC_FREQ_RAD_S",
    "RaoTable",
    "load_pdstrip_rao",
    "evaluate_rao",
    "evaluate_drift",
]
