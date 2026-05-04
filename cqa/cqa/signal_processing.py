"""Signal processing helpers for the cqa pipeline.

Currently a single utility: zero-phase low-pass / band split of a
scalar 1-D channel into a low-frequency component and the residual
high-frequency component.

Design rationale
----------------
The DP intact-prior pipeline produces two well-separated bands on the
gangway-telescope channel:

  * Slow band:  closed-loop response to wind / drift / current
                disturbances. Energy concentrated below ~0.15 rad/s.
  * Wave band:  1st-order vessel motion driven by the wave RAO.
                Energy concentrated around 0.5-1.0 rad/s, falls off
                sharply by ~1.3 rad/s.

The two bands are separated by more than 2 octaves, so a moderate-
order Butterworth low-pass at ~0.3 rad/s gives a clean split:

    x_lf(t) = LP_{omega_split}(x(t))
    x_wf(t) = x(t) - x_lf(t)

We use ``scipy.signal.filtfilt`` for zero-phase filtering: the data
is filtered forwards then backwards, so the effective frequency
response is |H(omega)|^2 (steeper roll-off, no phase distortion).

This is OFFLINE only -- ``filtfilt`` requires the full series. For
the online estimator we'd use a causal IIR filter and book-keep its
group delay; that's a separate implementation deferred to the C++
port.

References
----------
Oppenheim & Schafer, "Discrete-Time Signal Processing", 3rd ed.,
ch. 7 (filter design) and 6.6 (zero-phase filtering).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt


def bandsplit_lowpass(
    x: np.ndarray,
    fs_hz: float,
    omega_split_rad_s: float,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1-D signal into low-frequency and high-frequency bands.

    Uses a zero-phase Butterworth low-pass: the LF component is the
    forward-backward filtered signal, the HF component is the residual
    ``x - x_lf``. By construction ``x_lf + x_wf == x`` at machine
    precision.

    Parameters
    ----------
    x : 1-D ndarray of samples.
    fs_hz : float. Sample rate [Hz].
    omega_split_rad_s : float. Cutoff angular frequency [rad/s] for the
        low-pass. The Butterworth -3 dB frequency. Choose well above
        the LF band energy and well below the HF band energy.
    order : int, default 4. Butterworth filter order. Effective order
        with filtfilt is ``2 * order`` (forward then backward), so
        order=4 gives a 48 dB/octave roll-off.

    Returns
    -------
    (x_lf, x_wf) tuple of two ndarrays, each the same shape as ``x``.

    Notes
    -----
    * filtfilt requires ``len(x) > padlen`` where padlen ~ 3*order*N_b.
      For order=4 this is ~12 samples; the function raises a clear
      error if the input is too short.

    * Cutoff frequency is given in rad/s for cqa convention; converted
      internally to a normalised digital frequency
      ``Wn = (omega_split / (2 pi)) / (fs / 2) = omega_split / (pi fs)``.

    * Group delay of the resulting filter is zero (filtfilt
      property), so x_lf and x_wf are time-aligned with the original
      x. This is the key property that makes the sum-back-to-x
      identity exact.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1.0
    >>> t = np.arange(0, 600, 1/fs)
    >>> x_slow = np.sin(2 * np.pi * 0.01 * t)        # 100-s period
    >>> x_wave = np.sin(2 * np.pi * 0.1 * t)         # 10-s period
    >>> x = x_slow + x_wave
    >>> lf, wf = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=0.3)
    >>> # lf ~ x_slow, wf ~ x_wave; lf+wf == x exactly.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be > 0, got {fs_hz}")
    if omega_split_rad_s <= 0.0:
        raise ValueError(
            f"omega_split_rad_s must be > 0, got {omega_split_rad_s}"
        )
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    # Convert rad/s cutoff to normalised digital frequency.
    nyquist_rad_s = np.pi * fs_hz
    if omega_split_rad_s >= nyquist_rad_s:
        raise ValueError(
            f"omega_split_rad_s={omega_split_rad_s} must be below Nyquist "
            f"= pi * fs = {nyquist_rad_s:.3f} rad/s"
        )
    Wn = omega_split_rad_s / nyquist_rad_s   # in (0, 1)

    b, a = butter(order, Wn, btype="lowpass")

    # filtfilt's default padlen is 3 * max(len(a), len(b)). For order
    # 4 Butterworth this is ~15. Use the default; raise our own error
    # with a clearer message if too short.
    min_len = 3 * max(len(a), len(b))
    if x.size <= min_len:
        raise ValueError(
            f"x is too short for filtfilt: len(x)={x.size} but need "
            f">= {min_len + 1} for order={order}. Either lengthen the "
            f"window or lower the order."
        )

    x_lf = filtfilt(b, a, x)
    x_wf = x - x_lf
    return x_lf, x_wf


__all__ = ["bandsplit_lowpass"]
