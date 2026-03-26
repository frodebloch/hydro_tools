#!/usr/bin/env python3
"""
Vessel-as-wave-buoy: Estimate wave spectrum from measured vessel motions.

Reads exported MRU/navigation data (long-format CSV from Brunvoll systems),
loads PdStrip RAOs, and estimates wave parameters from motion spectra.

Usage:
    python wave_buoy.py /path/to/exported_data.csv --rao /path/to/pdstrip.dat
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal


# ---------------------------------------------------------------------------
# 1. CSV Parser — long-format with European decimals
# ---------------------------------------------------------------------------

# Signals of interest, grouped by source
MRU_SIGNALS = [
    "VesselHeave", "VesselHeaveVel", "VesselHeaveAcc",
    "VesselPitch", "VesselPitchRate", "VesselPitchAcc",
    "VesselRoll", "VesselRollRate", "VesselRollAcc",
    "VesselYaw", "VesselYawRate", "VesselYawAcc",
    "VesselSurge", "VesselSurgeVel", "VesselSurgeAcc",
    "VesselSway", "VesselSwayVel", "VesselSwayAcc",
    "VesselAmplitudeX", "VesselAmplitudeY", "VesselAmplitudeZ",
    "VesselPeriodX", "VesselPeriodY", "VesselPeriodZ",
]

NAV_SIGNALS = [
    "HeadingDeg", "Latitude", "Longitude",
    "SpeedOverGround", "SpeedThroughWater",
    "RateOfTurn", "WaterDepthBelowKeel",
    "WindSpeed", "WindAngleRel",
]


def parse_csv(csv_path: str, signals: list[str] | None = None) -> dict[str, pd.Series]:
    """Parse long-format CSV with European decimal notation.

    Returns dict of signal_name -> pd.Series(float, index=DatetimeIndex).
    """
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        sep=";",
        quotechar='"',
        header=0,
        names=["time", "metric", "value"],
        dtype={"time": str, "metric": str, "value": str},
        low_memory=False,
    )
    print(f"  {len(df):,} rows loaded")

    # Strip device prefix (e.g. "11324 VesselPitch" -> "VesselPitch")
    df["signal"] = df["metric"].str.replace(r"^\d+\s+", "", regex=True)

    # Filter to signals of interest
    if signals is None:
        signals = MRU_SIGNALS + NAV_SIGNALS
    df = df[df["signal"].isin(signals)].copy()
    print(f"  {len(df):,} rows after filtering to {len(signals)} signals")

    # Parse values (European decimal: comma -> dot)
    df["value"] = df["value"].str.replace(",", ".").astype(float)

    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Build per-signal time series
    result = {}
    for sig_name, group in df.groupby("signal"):
        s = group.set_index("time")["value"].sort_index()
        s.name = sig_name
        # Remove exact duplicate timestamps (keep first)
        s = s[~s.index.duplicated(keep="first")]
        result[sig_name] = s

    return result


# ---------------------------------------------------------------------------
# 2. Resample to uniform grid
# ---------------------------------------------------------------------------

def resample_uniform(s: pd.Series, fs: float = 1.0) -> pd.Series:
    """Resample an irregularly-sampled signal to a uniform grid at fs Hz.

    Uses linear interpolation. Returns new Series with uniform DatetimeIndex.
    """
    t0 = s.index[0]
    t1 = s.index[-1]
    dt = pd.Timedelta(seconds=1.0 / fs)
    new_index = pd.date_range(start=t0, end=t1, freq=dt)

    # Convert to float seconds for interpolation
    t_orig = (s.index - t0).total_seconds().values
    t_new = (new_index - t0).total_seconds().values
    v_new = np.interp(t_new, t_orig, s.values)

    return pd.Series(v_new, index=new_index, name=s.name)


# ---------------------------------------------------------------------------
# 3. Motion spectra (Welch)
# ---------------------------------------------------------------------------

def compute_spectrum(s: pd.Series, fs: float = 1.0, segment_sec: float = 256.0,
                     overlap_frac: float = 0.5, detrend: str = "linear"):
    """Compute one-sided power spectral density using Welch's method.

    Returns (freq_hz, psd) where psd is in [unit^2/Hz].
    """
    nperseg = int(segment_sec * fs)
    noverlap = int(nperseg * overlap_frac)
    f, psd = signal.welch(
        s.values,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        window="hann",
        scaling="density",
    )
    return f, psd


def spectral_moments(f, psd):
    """Compute spectral moments m0, m1, m2, m4 and derived parameters."""
    df = np.diff(f)
    fc = 0.5 * (f[:-1] + f[1:])
    sc = 0.5 * (psd[:-1] + psd[1:])

    m0 = np.sum(sc * df)
    m1 = np.sum(fc * sc * df)
    m2 = np.sum(fc**2 * sc * df)
    m4 = np.sum(fc**4 * sc * df)

    Hs = 4.0 * np.sqrt(m0)  # significant height (for displacement spectra)
    Tm01 = m1 / m0 if m0 > 0 else 0.0  # mean period
    Tm02 = np.sqrt(m2 / m0) if m0 > 0 else 0.0  # zero-crossing period
    Tp = 1.0 / fc[np.argmax(sc)] if np.max(sc) > 0 else 0.0  # peak period

    return dict(m0=m0, m1=m1, m2=m2, m4=m4, Hs=Hs,
                Tm01=1.0/Tm01 if Tm01 > 0 else 0.0,
                Tm02=1.0/Tm02 if Tm02 > 0 else 0.0, Tp=Tp)


# ---------------------------------------------------------------------------
# 4. RAO loader (PdStrip pdstrip.dat format)
# ---------------------------------------------------------------------------

def load_rao(dat_path: str, speed_ms: float = 0.0) -> dict:
    """Load PdStrip RAO table from pdstrip.dat.

    Returns dict with:
        freq_rad: array of wave frequencies (rad/s)
        angles_deg: array of wave encounter angles (deg)
        heave: complex RAO array [n_freq, n_angle] (heave_r + j*heave_i)
        pitch: complex RAO array [n_freq, n_angle] (pitch_r + j*pitch_i)
        roll:  complex RAO array [n_freq, n_angle] (roll_r + j*roll_i)
        surge_drift, sway_drift, yaw_drift: mean drift force arrays
    """
    print(f"Loading RAOs from {dat_path} ...")
    df = pd.read_csv(dat_path, sep=r"\s+", header=0)

    # Columns: freq enc angle speed surge_r surge_i sway_r sway_i
    #          heave_r heave_i roll_r roll_i pitch_r pitch_i yaw_r yaw_i
    #          surge_d sway_d yaw_d roll_d

    # Select speed closest to requested
    speeds = sorted(df["speed"].unique())
    # speed = -1 is the zero-speed case in PdStrip (used for drift forces etc.)
    # speed = 0 also exists and is valid zero-speed
    # Map -1 -> 0 for distance comparison, but prefer speed=0 over speed=-1
    speed_map = {s: (0.0 if s < 0 else s) for s in speeds}
    best_speed = min(speeds, key=lambda s: abs(speed_map[s] - speed_ms) + (0.01 if s < 0 else 0))
    print(f"  Available speeds: {speeds}")
    print(f"  Selected speed={best_speed} (mapped to {speed_map[best_speed]:.1f} m/s)")

    sub = df[df["speed"] == best_speed].copy()

    freqs = sorted(sub["freq"].unique())
    angles = sorted(sub["angle"].unique())
    n_freq = len(freqs)
    n_angle = len(angles)
    print(f"  Grid: {n_freq} frequencies x {n_angle} angles")
    print(f"  Freq range: {freqs[0]:.3f} - {freqs[-1]:.3f} rad/s")
    print(f"  Angle range: {angles[0]:.0f} - {angles[-1]:.0f} deg")

    # Build 2D arrays
    heave = np.zeros((n_freq, n_angle), dtype=complex)
    pitch = np.zeros((n_freq, n_angle), dtype=complex)
    roll = np.zeros((n_freq, n_angle), dtype=complex)
    surge_d = np.zeros((n_freq, n_angle))
    sway_d = np.zeros((n_freq, n_angle))
    yaw_d = np.zeros((n_freq, n_angle))
    roll_d = np.zeros((n_freq, n_angle))

    freq_idx = {f: i for i, f in enumerate(freqs)}
    angle_idx = {a: i for i, a in enumerate(angles)}

    for _, row in sub.iterrows():
        fi = freq_idx[row["freq"]]
        ai = angle_idx[row["angle"]]
        heave[fi, ai] = row["heave_r"] + 1j * row["heave_i"]
        pitch[fi, ai] = row["pitch_r"] + 1j * row["pitch_i"]
        roll[fi, ai] = row["roll_r"] + 1j * row["roll_i"]
        surge_d[fi, ai] = row["surge_d"]
        sway_d[fi, ai] = row["sway_d"]
        yaw_d[fi, ai] = row["yaw_d"]
        roll_d[fi, ai] = row["roll_d"]

    return dict(
        freq_rad=np.array(freqs),
        angles_deg=np.array(angles),
        heave=heave,
        pitch=pitch,
        roll=roll,
        surge_drift=surge_d,
        sway_drift=sway_d,
        yaw_drift=yaw_d,
        roll_drift=roll_d,
    )


# ---------------------------------------------------------------------------
# 5. Wave estimation — parametric (single direction, JONSWAP fit)
# ---------------------------------------------------------------------------

def estimate_wave_from_pitch(f_hz, psd_pitch_deg2, rao: dict, heading_deg: float,
                             wave_dir_search: np.ndarray | None = None):
    """Estimate wave spectrum from pitch motion spectrum + RAO.

    PdStrip pitch RAO is NONDIMENSIONAL: pitch_amplitude / (k * wave_amplitude)
    where k = omega^2/g (deep water wavenumber).

    So: pitch_amp [rad] = RAO * k * wave_amp
    And: S_pitch(f) [rad^2/Hz] = |RAO(f)|^2 * k(f)^2 * S_wave(f) [m^2/Hz]
    => S_wave(f) = S_pitch(f) / (|RAO|^2 * k^2)

    Args:
        f_hz: frequency array (Hz)
        psd_pitch_deg2: pitch PSD in [deg^2/Hz]
        rao: dict from load_rao()
        heading_deg: vessel heading (deg, compass)
        wave_dir_search: candidate wave directions to try (deg true, 0=N)

    Returns list of dicts sorted by score, best first.
    """
    g = 9.81

    # Convert pitch PSD from deg^2/Hz to rad^2/Hz
    psd_pitch_rad2 = psd_pitch_deg2 * (np.pi / 180.0) ** 2

    omega = 2.0 * np.pi * f_hz  # rad/s
    # Deep water wavenumber
    k = np.where(omega > 0, omega**2 / g, 0.0)

    rao_freqs = rao["freq_rad"]
    rao_angles = rao["angles_deg"]

    # PdStrip angle convention: 0 = head seas (waves from bow), 180 = following
    # We need to convert from (wave_dir_true, heading) to encounter angle
    # encounter_angle = wave_dir_true - heading (then map to PdStrip convention)
    # PdStrip: angle is measured from bow, positive to port
    # 0 = head seas, 90 = beam from port, 180 = following

    if wave_dir_search is None:
        wave_dir_search = np.arange(0, 360, 10)  # every 10 degrees

    results = []
    for wave_dir in wave_dir_search:
        # Relative wave direction (where waves come FROM relative to bow)
        # encounter_angle = heading - wave_dir (head seas when waves come from heading direction)
        rel_angle = heading_deg - wave_dir
        # Normalize to PdStrip convention
        # PdStrip 0 = head seas = waves from ahead
        enc_angle = rel_angle % 360
        if enc_angle > 260:
            enc_angle -= 360  # map to [-90, 260] range

        # Check if this angle is within RAO range
        if enc_angle < rao_angles[0] or enc_angle > rao_angles[-1]:
            continue

        # Interpolate pitch RAO magnitude onto our frequency grid
        # For each frequency, interpolate in angle
        rao_pitch_mag2 = np.zeros_like(f_hz)
        for i, om in enumerate(omega):
            if om < rao_freqs[0] or om > rao_freqs[-1]:
                rao_pitch_mag2[i] = 0.0
                continue

            # Bilinear interpolation on (freq, angle)
            fi = np.searchsorted(rao_freqs, om) - 1
            fi = np.clip(fi, 0, len(rao_freqs) - 2)
            ai = np.searchsorted(rao_angles, enc_angle) - 1
            ai = np.clip(ai, 0, len(rao_angles) - 2)

            wf = (om - rao_freqs[fi]) / (rao_freqs[fi + 1] - rao_freqs[fi])
            wa = (enc_angle - rao_angles[ai]) / (rao_angles[ai + 1] - rao_angles[ai])
            wf = np.clip(wf, 0, 1)
            wa = np.clip(wa, 0, 1)

            # Interpolate complex RAO, then take magnitude squared
            c00 = rao["pitch"][fi, ai]
            c01 = rao["pitch"][fi, ai + 1]
            c10 = rao["pitch"][fi + 1, ai]
            c11 = rao["pitch"][fi + 1, ai + 1]
            c = (1 - wf) * (1 - wa) * c00 + (1 - wf) * wa * c01 + wf * (1 - wa) * c10 + wf * wa * c11
            rao_pitch_mag2[i] = abs(c) ** 2

        # Deconvolve: S_wave = S_pitch / (|RAO_pitch|^2 * k^2)
        # The factor k^2 accounts for pitch RAO being nondimensional
        min_rao2_k2 = 1e-12
        rao_k2 = rao_pitch_mag2 * k**2
        valid = rao_k2 > min_rao2_k2
        s_wave = np.zeros_like(f_hz)
        s_wave[valid] = psd_pitch_rad2[valid] / rao_k2[valid]

        # Compute wave spectral moments (only within RAO-valid range)
        if np.sum(valid) < 5:
            continue

        moments = spectral_moments(f_hz[valid], s_wave[valid])

        results.append(dict(
            wave_dir=wave_dir,
            enc_angle=enc_angle,
            Hs=moments["Hs"],
            Tp=moments["Tp"],
            Tm02=moments["Tm02"],
            s_wave=s_wave,
            rao_pitch_mag2=rao_pitch_mag2,
            rao_k2=rao_k2,
            valid=valid,
            moments=moments,
        ))

    if not results:
        print("  WARNING: No valid wave direction found!")
        return None

    # Direction selection: we need a criterion that rewards directions where
    # the RAO is large enough to give a well-conditioned deconvolution,
    # AND the resulting wave spectrum has a plausible shape.
    #
    # Key insight: dividing by a small RAO amplifies noise, producing
    # flat or rising spectra at high frequencies. A good estimate should
    # have energy concentrated in a peak with spectral rolloff.
    #
    # Score = mean_rao_mag * spectral_peakedness
    # where spectral_peakedness = peak_psd / mean_psd (high = peaked spectrum)
    # and mean_rao_mag penalizes directions with tiny RAOs.

    for r in results:
        s = r["s_wave"]
        valid = r["valid"]
        rao_k = np.sqrt(r["rao_k2"])  # effective transfer function including k

        # Mean effective transfer function in valid band (penalize small values)
        mean_rao = np.mean(rao_k[valid]) if np.any(valid) else 0.0

        # Spectral peakedness: ratio of peak to mean PSD
        s_valid = s[valid]
        if len(s_valid) > 5 and np.mean(s_valid) > 0:
            peakedness = np.max(s_valid) / np.mean(s_valid)
        else:
            peakedness = 0.0

        # Penalize unrealistic Hs
        hs_penalty = 1.0 if r["Hs"] < 10.0 else 0.1

        r["score"] = mean_rao * peakedness * hs_penalty
        r["mean_rao"] = mean_rao
        r["peakedness"] = peakedness

    results.sort(key=lambda r: r["score"], reverse=True)

    # Filter out unrealistic results
    results = [r for r in results if r["Hs"] < 15.0]
    if not results:
        print("  WARNING: All estimates unrealistic (Hs > 15m)")
        return None

    return results


def estimate_wave_from_heave(f_hz, psd_heave_m2, rao: dict, heading_deg: float,
                              wave_dir_deg: float, min_rao: float = 0.5):
    """Estimate wave spectrum from heave motion spectrum + RAO at known direction.

    Heave RAO is heave_amplitude / wave_amplitude (m/m, dimensionless).
    S_heave(f) = |RAO_heave(f, theta)|^2 * S_wave(f)
    => S_wave(f) = S_heave(f) / |RAO_heave(f)|^2

    Only uses frequency bins where |RAO| > min_rao to avoid noise amplification.

    Returns estimated wave spectrum.
    """
    omega = 2.0 * np.pi * f_hz
    rao_freqs = rao["freq_rad"]
    rao_angles = rao["angles_deg"]

    rel_angle = heading_deg - wave_dir_deg
    enc_angle = rel_angle % 360
    if enc_angle > 260:
        enc_angle -= 360

    if enc_angle < rao_angles[0] or enc_angle > rao_angles[-1]:
        print(f"  WARNING: encounter angle {enc_angle:.0f} outside RAO range")
        return None

    rao_heave_mag2 = np.zeros_like(f_hz)
    for i, om in enumerate(omega):
        if om < rao_freqs[0] or om > rao_freqs[-1]:
            continue
        fi = np.searchsorted(rao_freqs, om) - 1
        fi = np.clip(fi, 0, len(rao_freqs) - 2)
        ai = np.searchsorted(rao_angles, enc_angle) - 1
        ai = np.clip(ai, 0, len(rao_angles) - 2)

        wf = (om - rao_freqs[fi]) / (rao_freqs[fi + 1] - rao_freqs[fi])
        wa = (enc_angle - rao_angles[ai]) / (rao_angles[ai + 1] - rao_angles[ai])
        wf = np.clip(wf, 0, 1)
        wa = np.clip(wa, 0, 1)

        c00 = rao["heave"][fi, ai]
        c01 = rao["heave"][fi, ai + 1]
        c10 = rao["heave"][fi + 1, ai]
        c11 = rao["heave"][fi + 1, ai + 1]
        c = (1 - wf) * (1 - wa) * c00 + (1 - wf) * wa * c01 + wf * (1 - wa) * c10 + wf * wa * c11
        rao_heave_mag2[i] = abs(c) ** 2

    min_rao2 = min_rao ** 2
    valid = rao_heave_mag2 > min_rao2
    s_wave = np.zeros_like(f_hz)
    s_wave[valid] = psd_heave_m2[valid] / rao_heave_mag2[valid]

    moments = spectral_moments(f_hz[valid], s_wave[valid])
    return dict(
        s_wave=s_wave,
        rao_heave_mag2=rao_heave_mag2,
        valid=valid,
        moments=moments,
        Hs=moments["Hs"],
        Tp=moments["Tp"],
    )


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot_time_series(signals: dict, out_dir: Path, segment: str = "full"):
    """Plot overview time series of key MRU signals."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    pairs = [
        ("VesselPitch", "Pitch [deg]"),
        ("VesselHeave", "Heave [m]"),
        ("VesselRoll", "Roll [deg]"),
        ("HeadingDeg", "Heading [deg]"),
    ]

    for ax, (sig, ylabel) in zip(axes, pairs):
        if sig in signals:
            s = signals[sig]
            t_min = (s.index - s.index[0]).total_seconds() / 60.0
            ax.plot(t_min, s.values, linewidth=0.3, color="C0")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            # Show statistics
            ax.text(0.01, 0.95, f"mean={s.mean():.2f}  std={s.std():.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        else:
            ax.text(0.5, 0.5, f"{sig} not available", transform=ax.transAxes,
                    ha="center", va="center")

    axes[-1].set_xlabel("Time [minutes from start]")
    axes[0].set_title(f"MRU Time Series — {segment}")
    plt.tight_layout()
    path = out_dir / f"01_time_series_{segment}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_motion_spectra(spectra: dict, out_dir: Path, segment: str = "full"):
    """Plot motion power spectra for pitch, heave, roll."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    items = [
        ("pitch", "Pitch PSD [deg²/Hz]"),
        ("heave", "Heave PSD [m²/Hz]"),
        ("roll", "Roll PSD [deg²/Hz]"),
    ]

    for ax, (key, ylabel) in zip(axes, items):
        if key in spectra:
            f, psd = spectra[key]
            ax.semilogy(f, psd, linewidth=0.8, color="C0")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, which="both")

            # Mark peak
            ipk = np.argmax(psd[1:]) + 1  # skip DC
            ax.axvline(f[ipk], color="red", linewidth=0.5, alpha=0.5)
            ax.text(f[ipk], psd[ipk], f"  Tp={1/f[ipk]:.1f}s", fontsize=8, color="red")

            # Show spectral moments
            moments = spectral_moments(f[1:], psd[1:])  # skip DC
            label = f"Hs={moments['Hs']:.2f}  Tp={moments['Tp']:.1f}s  Tm02={moments['Tm02']:.1f}s"
            ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    axes[-1].set_xlabel("Frequency [Hz]")
    axes[-1].set_xlim(0, 0.5)
    axes[0].set_title(f"Motion Power Spectra (Welch, 256s segments) — {segment}")
    plt.tight_layout()
    path = out_dir / f"02_motion_spectra_{segment}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_rao_overview(rao: dict, out_dir: Path):
    """Plot RAO magnitudes for heave, pitch, roll at key angles."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    items = [("heave", "Heave RAO [m/m]"), ("pitch", "Pitch RAO [rad/m]"), ("roll", "Roll RAO [rad/m]")]
    angles_to_show = [0, 30, 60, 90, 120, 150, 180]

    for ax, (key, ylabel) in zip(axes, items):
        rao_data = rao[key]
        freqs = rao["freq_rad"]
        angles = rao["angles_deg"]
        for target_angle in angles_to_show:
            ai = np.argmin(np.abs(angles - target_angle))
            mag = np.abs(rao_data[:, ai])
            ax.plot(freqs, mag, label=f"{angles[ai]:.0f}°", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=4, title="Encounter angle")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wave frequency [rad/s]")
    axes[0].set_title("PdStrip RAOs — Geir (zero speed)")
    plt.tight_layout()
    path = out_dir / "03_rao_overview.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_wave_estimate(f_hz, pitch_results, heave_result, out_dir: Path, segment: str = "full"):
    """Plot estimated wave spectra from pitch and heave deconvolution."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Top: estimated wave spectra from pitch (top 5 directions)
    ax = axes[0]
    if pitch_results:
        for i, r in enumerate(pitch_results[:5]):
            ax.plot(f_hz, r["s_wave"], linewidth=0.8,
                    label=f"dir={r['wave_dir']:.0f}° Hs={r['Hs']:.2f}m Tp={r['Tp']:.1f}s")
        ax.set_ylabel("Wave PSD [m²/Hz]")
        ax.set_title(f"Wave Spectrum Estimated from Pitch — {segment}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.3)

    # Middle: estimated wave spectrum from heave at best direction
    ax = axes[1]
    if heave_result:
        ax.plot(f_hz, heave_result["s_wave"], linewidth=1.0, color="C1",
                label=f"Heave: Hs={heave_result['Hs']:.2f}m Tp={heave_result['Tp']:.1f}s")
        ax.set_ylabel("Wave PSD [m²/Hz]")
        ax.set_title(f"Wave Spectrum Estimated from Heave — {segment}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.3)
    else:
        ax.text(0.5, 0.5, "Heave estimate not available", transform=ax.transAxes,
                ha="center", va="center")

    # Bottom: effective transfer functions used (RAO * k for pitch)
    ax = axes[2]
    if pitch_results:
        best = pitch_results[0]
        ax.plot(f_hz, np.sqrt(best["rao_k2"]), linewidth=1.0, color="C0",
                label=f"Pitch |RAO*k| @ {best['enc_angle']:.0f}° enc")
        ax.plot(f_hz, np.sqrt(best["rao_pitch_mag2"]), linewidth=0.5, color="C0",
                linestyle="--", alpha=0.5,
                label=f"Pitch |RAO| (nondim)")
    if heave_result:
        ax.plot(f_hz, np.sqrt(heave_result["rao_heave_mag2"]), linewidth=1.0, color="C1",
                label="Heave RAO")
    ax.set_ylabel("Transfer function magnitude")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Transfer Functions Used")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.3)

    plt.tight_layout()
    path = out_dir / f"04_wave_estimate_{segment}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 7. Lever arm check: heave-pitch cross-contamination diagnostic
# ---------------------------------------------------------------------------

def lever_arm_diagnostic(f_hz, heave_psd, pitch_psd, cross_psd,
                         rao: dict, heading_deg: float, wave_dir_deg: float,
                         out_dir: Path, segment: str = "full"):
    """Check for lever arm errors by examining heave-pitch coherence.

    If MRU is offset from CoG by dx (forward), then:
        heave_measured = heave_true + dx * pitch (in radians)

    This creates excess coherence between heave and pitch beyond what
    the wave-induced correlation (through RAOs) would predict.

    Returns estimated dx lever arm error.
    """
    # Coherence = |Sxy|^2 / (Sxx * Syy)
    coherence = np.abs(cross_psd) ** 2 / (heave_psd * pitch_psd + 1e-30)

    # Transfer function H = Sxy / Sxx (heave per unit pitch)
    H = cross_psd / (pitch_psd + 1e-30)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(f_hz, coherence, linewidth=0.8)
    axes[0].set_ylabel("Coherence")
    axes[0].set_title(f"Heave-Pitch Cross-Spectral Analysis — {segment}")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_hz, np.abs(H), linewidth=0.8)
    axes[1].set_ylabel("|H(f)| = |S_hp| / S_pp")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f_hz, np.angle(H, deg=True), linewidth=0.8)
    axes[2].set_ylabel("Phase(H) [deg]")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"05_lever_arm_diagnostic_{segment}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    return coherence, H


def plot_summary(f_hz, heave_results_all, spectra, rao, heading, out_dir,
                 segment="full", best_dir_info=None):
    """Summary plot: heave-derived wave spectrum vs motion spectra and RAOs.

    Args:
        best_dir_info: optional dict with 'wave_dir', 'method' from direction estimation
    """
    if not heave_results_all:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Pick best direction: use direction estimate if available, else median Hs
    hs_vals = [r["Hs"] for r in heave_results_all]
    median_hs = np.median(hs_vals)
    if best_dir_info is not None:
        best = min(heave_results_all,
                   key=lambda r: abs(r["wave_dir"] - best_dir_info["wave_dir"]))
        dir_note = f" ({best_dir_info['method']})"
    else:
        best = min(heave_results_all, key=lambda r: abs(r["Hs"] - median_hs))
        dir_note = " (median Hs)"

    # Top-left: Heave motion spectrum and derived wave spectrum (best direction)
    ax = axes[0, 0]
    f_h, psd_h = spectra["heave"]
    ax.plot(f_h, psd_h, linewidth=1.0, color="C0", label="Heave motion S(f)")
    ax.plot(f_hz, best["s_wave"], linewidth=1.0, color="C1",
            label=f"Wave S(f) (dir={best['wave_dir']:.0f}°{dir_note})")
    ax.set_ylabel("PSD [m²/Hz]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_xlim(0, 0.25)
    ax.set_title("Heave Motion -> Wave Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Hs vs wave direction
    ax = axes[0, 1]
    dirs = [r["wave_dir"] for r in heave_results_all]
    hs_list = [r["Hs"] for r in heave_results_all]
    ax.plot(dirs, hs_list, "o-", markersize=3, linewidth=0.8)
    ax.axhline(median_hs, color="red", linewidth=0.5, linestyle="--",
               label=f"Median Hs={median_hs:.2f}m")
    if best_dir_info is not None:
        ax.axvline(best_dir_info["wave_dir"], color="green", linewidth=1.0,
                   linestyle=":", label=f"Est. dir={best_dir_info['wave_dir']:.0f}°")
    ax.set_xlabel("Wave direction [deg true]")
    ax.set_ylabel("Estimated Hs [m]")
    ax.set_title("Hs Sensitivity to Wave Direction (Heave)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Pitch motion spectrum
    ax = axes[1, 0]
    if "pitch" in spectra:
        f_p, psd_p = spectra["pitch"]
        ax.semilogy(f_p, psd_p, linewidth=1.0, color="C2")
        m = spectral_moments(f_p[1:], psd_p[1:])
        ax.set_title(f"Pitch Motion Spectrum (RMS={np.sqrt(m['m0']):.2f} deg, Tp={m['Tp']:.1f}s)")
    else:
        ax.text(0.5, 0.5, "Pitch not available", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_ylabel("PSD [deg^2/Hz]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_xlim(0, 0.25)
    ax.grid(True, alpha=0.3, which="both")

    # Bottom-right: Heave RAO at selected angles
    ax = axes[1, 1]
    for target_angle in [0, 30, 60, 90, 120, 150, 180]:
        ai = np.argmin(np.abs(rao["angles_deg"] - target_angle))
        mag = np.abs(rao["heave"][:, ai])
        f_rao = rao["freq_rad"] / (2 * np.pi)  # convert to Hz
        ax.plot(f_rao, mag, label=f"{rao['angles_deg'][ai]:.0f} deg", linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Heave RAO [m/m]")
    ax.set_title("Heave RAO (PdStrip)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)

    fig.suptitle(f"Vessel-as-Wave-Buoy Summary -- {segment}\n"
                 f"Heading={heading:.0f} deg | Hs={best['Hs']:.2f}m  Tp={best['Tp']:.1f}s"
                 f"  Dir={best['wave_dir']:.0f} deg{dir_note}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"06_summary_{segment}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 8. Direction estimation from heave-pitch cross-spectrum
# ---------------------------------------------------------------------------

def estimate_direction_heave_pitch(f_hz, heave_psd, pitch_psd_rad2, cross_psd,
                                   rao: dict, heading_deg: float,
                                   f_band: tuple[float, float] = (0.05, 0.20)):
    """Estimate wave direction from heave-pitch cross-spectrum + RAOs.

    Principle: For a unidirectional sea at angle theta, the cross-spectrum
    between heave and pitch is:
        S_hp(f) = RAO_heave*(f,theta) * RAO_pitch(f,theta) * k(f) * S_wave(f)

    The phase of S_hp tells us the relative phase between heave and pitch RAOs.
    For head seas (theta=0), heave and pitch are ~90 deg out of phase at resonance.
    For following seas (theta=180), the sign flips.

    We score each candidate direction by comparing the measured cross-spectrum
    phase with the predicted phase from RAOs, weighted by coherence.

    Args:
        f_hz: frequency array
        heave_psd: heave auto-spectrum [m^2/Hz]
        pitch_psd_rad2: pitch auto-spectrum [rad^2/Hz]
        cross_psd: complex cross-spectrum S_hp [m*rad/Hz]
        rao: dict from load_rao()
        heading_deg: vessel heading (compass deg)
        f_band: frequency band (Hz) for direction scoring

    Returns:
        best_dir: estimated wave direction (deg true)
        scores: list of (wave_dir, score) tuples
    """
    g = 9.81
    omega = 2.0 * np.pi * f_hz
    k = np.where(omega > 0, omega**2 / g, 0.0)

    # Band mask
    band = (f_hz >= f_band[0]) & (f_hz <= f_band[1])
    if np.sum(band) < 5:
        print("  WARNING: Too few points in frequency band for direction estimation")
        return None, []

    # Measured coherence and phase in band
    coherence = np.abs(cross_psd)**2 / (heave_psd * pitch_psd_rad2 + 1e-30)
    meas_phase = np.angle(cross_psd)  # radians

    rao_freqs = rao["freq_rad"]
    rao_angles = rao["angles_deg"]

    scores = []
    for wave_dir in np.arange(0, 360, 5):
        rel_angle = heading_deg - wave_dir
        enc_angle = rel_angle % 360
        if enc_angle > 260:
            enc_angle -= 360
        if enc_angle < rao_angles[0] or enc_angle > rao_angles[-1]:
            continue

        # Interpolate heave and pitch RAOs onto frequency grid
        rao_h = np.zeros(len(f_hz), dtype=complex)
        rao_p = np.zeros(len(f_hz), dtype=complex)
        for i, om in enumerate(omega):
            if om < rao_freqs[0] or om > rao_freqs[-1]:
                continue
            fi = np.searchsorted(rao_freqs, om) - 1
            fi = np.clip(fi, 0, len(rao_freqs) - 2)
            ai = np.searchsorted(rao_angles, enc_angle) - 1
            ai = np.clip(ai, 0, len(rao_angles) - 2)
            wf = np.clip((om - rao_freqs[fi]) / (rao_freqs[fi+1] - rao_freqs[fi]), 0, 1)
            wa = np.clip((enc_angle - rao_angles[ai]) / (rao_angles[ai+1] - rao_angles[ai]), 0, 1)
            # Bilinear interp
            for key, arr in [("heave", rao_h), ("pitch", rao_p)]:
                c00 = rao[key][fi, ai]
                c01 = rao[key][fi, ai + 1]
                c10 = rao[key][fi + 1, ai]
                c11 = rao[key][fi + 1, ai + 1]
                arr[i] = (1-wf)*(1-wa)*c00 + (1-wf)*wa*c01 + wf*(1-wa)*c10 + wf*wa*c11

        # Predicted cross-spectrum phase:
        # S_hp = conj(RAO_heave) * (RAO_pitch * k) * S_wave
        # phase(S_hp) = phase(RAO_pitch * k) - phase(RAO_heave)
        #             = phase(RAO_pitch) - phase(RAO_heave)  (k is real positive)
        pred_phase = np.angle(rao_p) - np.angle(rao_h)

        # Phase difference (measured - predicted), wrapped to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * (meas_phase - pred_phase)))

        # Score: coherence-weighted mean cos(phase_diff) in band
        # cos=1 when phases match, cos=-1 when they're opposite
        weights = coherence[band]
        if np.sum(weights) < 1e-10:
            continue
        score = np.average(np.cos(phase_diff[band]), weights=weights)

        # Also penalize directions where RAO magnitudes are very small
        # (unreliable phase estimates)
        mean_h_mag = np.mean(np.abs(rao_h[band]))
        mean_p_k_mag = np.mean(np.abs(rao_p[band]) * k[band])
        rao_penalty = min(mean_h_mag, 1.0) * min(mean_p_k_mag * 20, 1.0)

        scores.append((wave_dir, score * rao_penalty, score, rao_penalty))

    if not scores:
        return None, []

    scores.sort(key=lambda x: x[1], reverse=True)
    best_dir = scores[0][0]

    print(f"\n  Direction estimation (heave-pitch cross-spectrum):")
    print(f"    Band: {f_band[0]:.3f} - {f_band[1]:.3f} Hz")
    print(f"    Best direction: {best_dir:.0f} deg true (score={scores[0][1]:.3f})")
    print(f"    Top 5:")
    for d, s_total, s_phase, s_rao in scores[:5]:
        print(f"      dir={d:5.0f} deg  score={s_total:.3f}  "
              f"(phase={s_phase:.3f}, rao={s_rao:.3f})")

    return best_dir, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vessel-as-wave-buoy analysis")
    parser.add_argument("csv", help="Path to exported data CSV")
    parser.add_argument("--rao", default=None,
                        help="Path to PdStrip pdstrip.dat file")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Vessel speed for RAO selection (m/s, default=0)")
    parser.add_argument("--fs", type=float, default=1.0,
                        help="Resample frequency (Hz, default=1.0)")
    parser.add_argument("--segment-sec", type=float, default=512.0,
                        help="Welch segment length (seconds, default=512)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots")
    parser.add_argument("--window-min", type=float, default=0.0,
                        help="Analysis window: minutes from start (0=full)")
    parser.add_argument("--window-dur", type=float, default=0.0,
                        help="Analysis window duration in minutes (0=full)")
    parser.add_argument("--min-rao", type=float, default=0.5,
                        help="Minimum heave RAO for valid deconvolution (default=0.5)")
    parser.add_argument("--wave-dir", type=float, default=None,
                        help="Override wave direction (deg true, 0=N). Skip direction estimation.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).parent / "plots" / "wave_buoy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse CSV ---
    signals = parse_csv(str(csv_path))
    print(f"\nAvailable signals: {sorted(signals.keys())}")
    for name, s in sorted(signals.items()):
        dt = s.index[-1] - s.index[0]
        print(f"  {name:30s}: {len(s):8,} samples, {dt.total_seconds()/60:.1f} min, "
              f"~{len(s)/dt.total_seconds():.1f} Hz")

    # --- Resample to uniform grid ---
    print(f"\nResampling to {args.fs} Hz uniform grid ...")
    resampled = {}
    for sig_name in MRU_SIGNALS:
        if sig_name in signals and len(signals[sig_name]) > 100:
            resampled[sig_name] = resample_uniform(signals[sig_name], fs=args.fs)

    # Apply analysis window if requested
    segment_label = "full"
    if args.window_min > 0 or args.window_dur > 0:
        t0 = next(iter(resampled.values())).index[0]
        win_start = t0 + pd.Timedelta(minutes=args.window_min)
        if args.window_dur > 0:
            win_end = win_start + pd.Timedelta(minutes=args.window_dur)
        else:
            win_end = next(iter(resampled.values())).index[-1]
        for key in list(resampled.keys()):
            resampled[key] = resampled[key][win_start:win_end]
        segment_label = f"t{args.window_min:.0f}-{args.window_min + args.window_dur:.0f}min"
        print(f"  Window: {win_start} to {win_end} ({segment_label})")

    # Also keep nav signals for plotting
    for sig_name in NAV_SIGNALS:
        if sig_name in signals and sig_name not in resampled:
            resampled[sig_name] = resample_uniform(signals[sig_name], fs=args.fs)
            if args.window_min > 0 or args.window_dur > 0:
                resampled[sig_name] = resampled[sig_name][win_start:win_end]

    # --- Location and heading (from windowed/resampled data) ---
    if "Latitude" in resampled:
        lat = resampled["Latitude"].median()
        lon = resampled["Longitude"].median()
        print(f"\nPosition (segment): {lat:.4f}°N, {lon:.4f}°E")
    if "HeadingDeg" in resampled:
        hdg = resampled["HeadingDeg"].median()
        hdg_std = resampled["HeadingDeg"].std()
        print(f"Heading (segment): {hdg:.1f}° (std={hdg_std:.1f}°)")
    if "SpeedOverGround" in resampled:
        sog = resampled["SpeedOverGround"]
        print(f"SOG (segment): mean={sog.mean():.1f} kn, std={sog.std():.1f} kn")
    if "WindSpeed" in resampled:
        ws = resampled["WindSpeed"]
        wa = resampled["WindAngleRel"]
        print(f"Wind (segment): speed={ws.mean():.1f} m/s (std={ws.std():.1f}), "
              f"rel angle={wa.mean():.0f}° (std={wa.std():.0f}°)")

    # --- Time series plot ---
    print("\nPlotting time series ...")
    plot_time_series(resampled, out_dir, segment=segment_label)

    # --- Compute motion spectra ---
    print(f"\nComputing motion spectra (Welch, {args.segment_sec}s segments) ...")
    spectra = {}
    if "VesselPitch" in resampled:
        f, psd = compute_spectrum(resampled["VesselPitch"], fs=args.fs,
                                  segment_sec=args.segment_sec)
        spectra["pitch"] = (f, psd)
        m = spectral_moments(f[1:], psd[1:])
        print(f"  Pitch: RMS={np.sqrt(m['m0']):.2f}°, peak period={m['Tp']:.1f}s")

    if "VesselHeave" in resampled:
        f, psd = compute_spectrum(resampled["VesselHeave"], fs=args.fs,
                                  segment_sec=args.segment_sec)
        spectra["heave"] = (f, psd)
        m = spectral_moments(f[1:], psd[1:])
        print(f"  Heave: Hs_motion={m['Hs']:.2f}m, peak period={m['Tp']:.1f}s")

    if "VesselRoll" in resampled:
        f, psd = compute_spectrum(resampled["VesselRoll"], fs=args.fs,
                                  segment_sec=args.segment_sec)
        spectra["roll"] = (f, psd)
        m = spectral_moments(f[1:], psd[1:])
        print(f"  Roll: RMS={np.sqrt(m['m0']):.2f}°, peak period={m['Tp']:.1f}s")

    plot_motion_spectra(spectra, out_dir, segment=segment_label)

    # --- RAO-based wave estimation ---
    if args.rao:
        print(f"\nLoading RAOs ...")
        rao = load_rao(args.rao, speed_ms=args.speed)
        plot_rao_overview(rao, out_dir)

        heading = resampled["HeadingDeg"].median() if "HeadingDeg" in resampled else 0.0

        # -- Heave-based estimation (PRIMARY) --
        heave_result = None
        heave_results_all = []
        best_dir_info = None

        if "heave" in spectra:
            f_heave, psd_heave = spectra["heave"]
            print(f"\nEstimating wave spectrum from heave (heading={heading:.1f} deg, "
                  f"min_rao={args.min_rao:.2f}) ...")
            for wave_dir in np.arange(0, 360, 10):
                hr = estimate_wave_from_heave(
                    f_heave, psd_heave, rao, heading, wave_dir,
                    min_rao=args.min_rao)
                if hr and hr["Hs"] < 15.0:
                    hr["wave_dir"] = wave_dir
                    heave_results_all.append(hr)

            if heave_results_all:
                hs_vals = [r["Hs"] for r in heave_results_all]
                median_hs = np.median(hs_vals)
                print(f"\n  Heave Hs across all directions:")
                print(f"    Range: {min(hs_vals):.2f} - {max(hs_vals):.2f} m")
                print(f"    Median: {median_hs:.2f} m")

        # -- Direction estimation from heave-pitch cross-spectrum --
        if args.wave_dir is not None:
            best_dir_info = {"wave_dir": args.wave_dir, "method": "user override"}
            print(f"\n  Using user-specified wave direction: {args.wave_dir:.0f} deg true")
        elif ("heave" in spectra and "pitch" in spectra and
                "VesselHeave" in resampled and "VesselPitch" in resampled):
            print("\nEstimating wave direction from heave-pitch cross-spectrum ...")
            nperseg_dir = int(args.segment_sec * args.fs)
            noverlap_dir = nperseg_dir // 2
            f_cross, Pxy = signal.csd(
                resampled["VesselHeave"].values,
                resampled["VesselPitch"].values * np.pi / 180.0,
                fs=args.fs, nperseg=nperseg_dir, noverlap=noverlap_dir,
                detrend="linear", window="hann",
            )
            _, Pxx = signal.welch(resampled["VesselHeave"].values, fs=args.fs,
                                  nperseg=nperseg_dir, noverlap=noverlap_dir,
                                  detrend="linear", window="hann")
            _, Pyy = signal.welch(resampled["VesselPitch"].values * np.pi / 180.0,
                                  fs=args.fs, nperseg=nperseg_dir, noverlap=noverlap_dir,
                                  detrend="linear", window="hann")

            # Determine frequency band from heave spectrum peak
            f_h, psd_h = spectra["heave"]
            ipk = np.argmax(psd_h[1:]) + 1
            f_peak = f_h[ipk]
            f_band = (max(0.03, f_peak * 0.5), min(0.5, f_peak * 2.0))

            est_dir, dir_scores = estimate_direction_heave_pitch(
                f_cross, Pxx, Pyy, Pxy, rao, heading, f_band=f_band)

            if est_dir is not None:
                best_dir_info = {"wave_dir": est_dir, "method": "heave-pitch cross-spectrum"}

            # Lever arm diagnostic plot (now with actual RAO dict)
            lever_arm_diagnostic(f_cross, Pxx, Pyy, Pxy, rao, heading,
                                 est_dir if est_dir is not None else 0.0,
                                 out_dir, segment_label)

        # -- Select best heave result based on direction --
        if heave_results_all:
            if best_dir_info is not None:
                heave_result = min(heave_results_all,
                                   key=lambda r: abs(r["wave_dir"] - best_dir_info["wave_dir"]))
            else:
                hs_vals = [r["Hs"] for r in heave_results_all]
                heave_result = min(heave_results_all,
                                   key=lambda r: abs(r["Hs"] - np.median(hs_vals)))

            print(f"\n  === PRIMARY RESULT (Heave) ===")
            dir_str = (f"{heave_result['wave_dir']:.0f} deg true"
                       f" ({best_dir_info['method']})" if best_dir_info
                       else "direction insensitive (median Hs)")
            print(f"    Hs  = {heave_result['Hs']:.2f} m")
            print(f"    Tp  = {heave_result['Tp']:.1f} s")
            print(f"    Dir = {dir_str}")

        # -- Pitch-based estimation (SECONDARY, for comparison) --
        pitch_results = None
        if "pitch" in spectra:
            f_pitch, psd_pitch = spectra["pitch"]
            print(f"\nEstimating wave spectrum from pitch (secondary) ...")
            pitch_results = estimate_wave_from_pitch(f_pitch, psd_pitch, rao, heading)
            if pitch_results:
                best_pitch = pitch_results[0]
                print(f"  Pitch estimate (best direction):")
                print(f"    Dir={best_pitch['wave_dir']:.0f} deg  "
                      f"Hs={best_pitch['Hs']:.2f}m  Tp={best_pitch['Tp']:.1f}s  "
                      f"(score={best_pitch.get('score', 0):.3f})")
                print(f"  NOTE: Pitch poorly conditioned for Hs (|RAO*k| ~ 0.05 at peak)")

        # -- Plots --
        if "pitch" in spectra and pitch_results:
            f_pitch, _ = spectra["pitch"]
            plot_wave_estimate(f_pitch, pitch_results, heave_result, out_dir, segment_label)

        if heave_results_all:
            f_heave, _ = spectra["heave"]
            plot_summary(f_heave, heave_results_all, spectra, rao, heading,
                         out_dir, segment_label, best_dir_info)

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
