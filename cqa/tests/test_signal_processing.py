"""Tests for cqa.signal_processing."""

from __future__ import annotations

import numpy as np
import pytest

from cqa.signal_processing import bandsplit_lowpass


def test_sum_back_to_input_is_exact():
    """By construction x_lf + x_wf == x to machine precision."""
    rng = np.random.default_rng(0)
    fs = 1.0
    N = 1000
    x = rng.standard_normal(N)
    x_lf, x_wf = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=0.3)
    np.testing.assert_allclose(x_lf + x_wf, x, atol=1e-12, rtol=0)


def test_recovers_low_frequency_sinusoid_into_lf_band():
    """A pure tone well below the cutoff should land in x_lf with
    near-unit gain, leaving x_wf ~ 0."""
    fs = 1.0
    t = np.arange(0, 2000, 1.0 / fs)
    omega_slow = 0.02      # period ~314 s, well below 0.3 rad/s cutoff
    x = np.sin(omega_slow * t)
    x_lf, x_wf = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=0.3)
    # Drop the first/last 100 s: filtfilt has edge transients there.
    e = int(100 * fs)
    rms_lf = np.sqrt(np.mean(x_lf[e:-e] ** 2))
    rms_wf = np.sqrt(np.mean(x_wf[e:-e] ** 2))
    rms_x  = np.sqrt(np.mean(x[e:-e] ** 2))
    assert rms_lf == pytest.approx(rms_x, rel=1e-3), (
        f"slow tone RMS leaked: lf={rms_lf:.4f} vs x={rms_x:.4f}"
    )
    assert rms_wf < 1e-3 * rms_x, (
        f"slow tone leaked into wf band: rms_wf={rms_wf:.4e} vs rms_x={rms_x:.4e}"
    )


def test_recovers_high_frequency_sinusoid_into_wf_band():
    """A pure tone well above the cutoff should land in x_wf with
    near-unit gain, leaving x_lf ~ 0."""
    fs = 1.0
    t = np.arange(0, 2000, 1.0 / fs)
    omega_wave = 0.7       # period ~9 s, well above 0.3 rad/s cutoff
    x = np.sin(omega_wave * t)
    x_lf, x_wf = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=0.3)
    e = int(100 * fs)
    rms_wf = np.sqrt(np.mean(x_wf[e:-e] ** 2))
    rms_lf = np.sqrt(np.mean(x_lf[e:-e] ** 2))
    rms_x  = np.sqrt(np.mean(x[e:-e] ** 2))
    assert rms_wf == pytest.approx(rms_x, rel=1e-3), (
        f"wave tone RMS leaked: wf={rms_wf:.4f} vs x={rms_x:.4f}"
    )
    assert rms_lf < 1e-3 * rms_x


def test_separates_two_well_separated_tones():
    """Slow + wave tone (a 30x frequency separation): both bands
    should recover the right amplitude with cross-leakage <1%."""
    fs = 1.0
    t = np.arange(0, 2000, 1.0 / fs)
    omega_slow = 0.02
    omega_wave = 0.6
    A_slow, A_wave = 1.5, 0.4
    x_slow = A_slow * np.sin(omega_slow * t)
    x_wave = A_wave * np.cos(omega_wave * t)
    x = x_slow + x_wave
    x_lf, x_wf = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=0.3)
    e = int(100 * fs)

    # RMS of a pure tone is A/sqrt(2).
    rms_slow_expected = A_slow / np.sqrt(2.0)
    rms_wave_expected = A_wave / np.sqrt(2.0)
    rms_lf = np.sqrt(np.mean(x_lf[e:-e] ** 2))
    rms_wf = np.sqrt(np.mean(x_wf[e:-e] ** 2))
    assert rms_lf == pytest.approx(rms_slow_expected, rel=1.5e-2)
    assert rms_wf == pytest.approx(rms_wave_expected, rel=1.5e-2)

    # Direct waveform recovery (within edge trim).
    np.testing.assert_allclose(x_lf[e:-e], x_slow[e:-e], atol=2e-2)
    np.testing.assert_allclose(x_wf[e:-e], x_wave[e:-e], atol=2e-2)


def test_passband_attenuation_sharp_butterworth():
    """At the -3 dB cutoff, filtfilt's effective response is |H|^2
    so the gain is -6 dB (factor 0.5 in amplitude squared, sqrt(0.5)
    ~ 0.707 in amplitude)."""
    fs = 1.0
    t = np.arange(0, 4000, 1.0 / fs)
    omega_c = 0.3
    x = np.sin(omega_c * t)
    x_lf, _ = bandsplit_lowpass(x, fs_hz=fs, omega_split_rad_s=omega_c, order=4)
    e = int(200 * fs)
    rms_lf = np.sqrt(np.mean(x_lf[e:-e] ** 2))
    rms_x  = np.sqrt(np.mean(x[e:-e] ** 2))
    # filtfilt with order-4 Butterworth at the design cutoff gives
    # amplitude ratio (1/sqrt(2))^2 = 0.5 (because |H|^2 instead of |H|).
    # Be lenient (5%): the design cutoff is the -3 dB point of |H|, the
    # actual amplitude attenuation depends on order.
    expected = 0.5 * rms_x
    assert rms_lf == pytest.approx(expected, rel=0.05), (
        f"Cutoff gain: lf/x={rms_lf/rms_x:.3f}, expected ~0.5 (filtfilt -6 dB)"
    )


# ---- error paths ----


def test_2d_input_raises():
    with pytest.raises(ValueError, match="must be 1-D"):
        bandsplit_lowpass(np.zeros((10, 10)), fs_hz=1.0,
                          omega_split_rad_s=0.3)


def test_negative_fs_raises():
    with pytest.raises(ValueError, match="fs_hz must be > 0"):
        bandsplit_lowpass(np.zeros(100), fs_hz=-1.0,
                          omega_split_rad_s=0.3)


def test_omega_split_above_nyquist_raises():
    # fs=1 Hz Nyquist is pi rad/s; ask for cutoff at 4 rad/s -> error.
    with pytest.raises(ValueError, match="below Nyquist"):
        bandsplit_lowpass(np.zeros(100), fs_hz=1.0,
                          omega_split_rad_s=4.0)


def test_too_short_signal_raises():
    with pytest.raises(ValueError, match="too short for filtfilt"):
        bandsplit_lowpass(np.zeros(5), fs_hz=1.0,
                          omega_split_rad_s=0.3, order=4)
