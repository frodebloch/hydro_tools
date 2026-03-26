#!/usr/bin/env python3
"""
Hindcast validation: compare vessel-as-wave-buoy estimates against NORA3.

Runs sliding-window wave estimation on measured vessel motions and overlays
the NORA3 wave hindcast time series at the nearest grid point.

Usage:
    python validate_hindcast.py /path/to/exported_data.csv \
        --rao /path/to/pdstrip.dat \
        --nora3-month 202106

The script:
  1. Parses vessel motion data and identifies stationary heading segments
  2. Runs heave-based wave estimation on overlapping windows
  3. Fetches NORA3 wave parameters (Hs, Tp, direction) via OPeNDAP
  4. Produces comparison plots

Requires: environment/.venv with xarray, netcdf4, numpy, scipy, matplotlib, pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal as sp_signal

# Import from wave_buoy.py (same directory)
from wave_buoy import (
    parse_csv, resample_uniform, compute_spectrum, spectral_moments,
    load_rao, estimate_wave_from_heave, MRU_SIGNALS, NAV_SIGNALS,
)
from common import nora3_wave_url

# Full-domain NORA3/Windsurfer WAM 3km hindcast (1959-present, ~2000x2400 grid)
NORA3_WAM3KM_AGG = ("https://thredds.met.no/thredds/dodsC/windsurfer/"
                     "mywavewam3km_agg/wam3kmhindcastaggregated.ncml")


# ---------------------------------------------------------------------------
# 1. NORA3 nearest-point extraction
# ---------------------------------------------------------------------------

def fetch_nora3_nearest(yyyymm: str, lat: float, lon: float,
                        time_start: str | None = None,
                        time_end: str | None = None,
                        use_full_domain: bool = True) -> pd.DataFrame:
    """Fetch NORA3 wave parameters at the nearest grid point via OPeNDAP.

    By default uses the full-domain WAM 3km aggregated dataset (Windsurfer),
    which has ~2000x2400 grid points covering 36-90N. Falls back to the
    NORA3 monthly subset files if use_full_domain=False.

    Returns DataFrame with columns: hs, tp, thq, hs_sea, hs_swell, tp_swell,
    thq_swell, ff, dd, indexed by time (UTC).
    """
    import xarray as xr

    if use_full_domain:
        url = NORA3_WAM3KM_AGG
        print(f"Opening NORA3 WAM 3km (full domain, aggregated): {url}")
        print("  (First access may take 20+ min for server-side caching)")
    else:
        url = nora3_wave_url(yyyymm)
        print(f"Opening NORA3 (subset): {url}")

    ds = xr.open_dataset(url)

    # NORA3 uses a rotated pole grid with 2D lat/lon arrays
    lat2d = ds["latitude"].values  # shape (rlat, rlon) or similar
    lon2d = ds["longitude"].values

    # Find nearest grid point
    dist2 = (lat2d - lat)**2 + (lon2d - lon)**2
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    nearest_lat = float(lat2d[idx])
    nearest_lon = float(lon2d[idx])
    dist_deg = float(np.sqrt(dist2[idx]))
    print(f"  Nearest grid point: {nearest_lat:.4f}N, {nearest_lon:.4f}E "
          f"(dist={dist_deg:.3f} deg, ~{dist_deg*111:.1f} km)")

    # Determine dimension ordering for the 2D array
    lat_dims = ds["latitude"].dims  # e.g. ('rlat', 'rlon')
    sel = {lat_dims[0]: idx[0], lat_dims[1]: idx[1]}

    # Variables to extract
    varnames = ["hs", "tp", "thq", "hs_sea", "hs_swell", "tp_swell",
                "thq_swell", "ff", "dd", "Pdir"]
    available = [v for v in varnames if v in ds.data_vars]
    print(f"  Extracting: {available}")

    # Extract time series at nearest point
    point = ds[available].isel(**sel)

    # Time subset
    if time_start or time_end:
        point = point.sel(time=slice(time_start, time_end))
        print(f"  Time subset: {point.time.values[0]} to {point.time.values[-1]} "
              f"({len(point.time)} records)")

    # Load into memory and convert to DataFrame
    print("  Loading data ...")
    point = point.load()
    ds.close()

    df = point.to_dataframe().reset_index()
    if "time" in df.columns:
        df = df.set_index("time")
        # Ensure timezone-aware (UTC) for compatibility with vessel data
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df.attrs["nearest_lat"] = nearest_lat
    df.attrs["nearest_lon"] = nearest_lon
    df.attrs["dist_deg"] = dist_deg
    print(f"  Got {len(df)} hourly records")
    return df


# ---------------------------------------------------------------------------
# 2. Sliding-window wave estimation
# ---------------------------------------------------------------------------

def sliding_wave_estimate(
    heave: pd.Series,
    heading: pd.Series,
    rao: dict,
    fs: float = 1.0,
    window_sec: float = 1800.0,
    step_sec: float = 600.0,
    segment_sec: float = 512.0,
    min_rao: float = 0.5,
    max_heading_std: float = 20.0,
) -> pd.DataFrame:
    """Run heave-based wave estimation on sliding windows.

    Args:
        heave: resampled heave time series [m]
        heading: resampled heading time series [deg]
        rao: dict from load_rao()
        fs: sample rate [Hz]
        window_sec: analysis window length [s]
        step_sec: step between windows [s]
        segment_sec: Welch segment length [s]
        min_rao: minimum heave RAO for valid deconvolution
        max_heading_std: reject windows with heading std above this [deg]

    Returns:
        DataFrame with columns: time, Hs, Tp, heading_mean, heading_std,
        Hs_min, Hs_max (range across directions)
    """
    t0 = heave.index[0]
    t_end = heave.index[-1]
    duration = (t_end - t0).total_seconds()
    n_windows = int((duration - window_sec) / step_sec) + 1

    results = []
    for i in range(n_windows):
        win_start = t0 + pd.Timedelta(seconds=i * step_sec)
        win_end = win_start + pd.Timedelta(seconds=window_sec)
        if win_end > t_end:
            break

        h_win = heave[win_start:win_end]
        hdg_win = heading[win_start:win_end]

        if len(h_win) < window_sec * fs * 0.8:
            continue

        # Check heading stationarity
        hdg_mean = hdg_win.median()
        # Circular std for heading
        hdg_rad = np.deg2rad(hdg_win.values)
        hdg_std = np.rad2deg(np.sqrt(-2 * np.log(
            np.sqrt(np.mean(np.cos(hdg_rad))**2 + np.mean(np.sin(hdg_rad))**2)
        )))

        if hdg_std > max_heading_std:
            continue

        # Compute heave spectrum
        f_hz, psd = compute_spectrum(h_win, fs=fs, segment_sec=segment_sec)

        # Scan all directions
        hs_all = []
        tp_best = None
        for wave_dir in np.arange(0, 360, 10):
            hr = estimate_wave_from_heave(
                f_hz, psd, rao, hdg_mean, wave_dir, min_rao=min_rao,
                quiet=True)
            if hr and 0 < hr["Hs"] < 15.0:
                hs_all.append(hr["Hs"])
                if tp_best is None:
                    tp_best = hr["Tp"]

        if not hs_all:
            continue

        # Window centre time
        t_centre = win_start + pd.Timedelta(seconds=window_sec / 2)

        results.append({
            "time": t_centre,
            "Hs": np.median(hs_all),
            "Hs_min": np.min(hs_all),
            "Hs_max": np.max(hs_all),
            "Tp": tp_best,
            "heading_mean": hdg_mean,
            "heading_std": hdg_std,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("time")
    print(f"  Sliding-window estimation: {len(df)} valid windows "
          f"(of {n_windows} total, rejected {n_windows - len(df)} for heading instability)")
    return df


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_validation(
    vessel_df: pd.DataFrame,
    nora3_df: pd.DataFrame | None,
    heading_series: pd.Series,
    out_dir: Path,
    vessel_name: str = "Vessel",
    nora3_meta: dict | None = None,
):
    """Create validation comparison plot."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # --- Panel 1: Hs comparison ---
    ax = axes[0]
    if not vessel_df.empty:
        t = vessel_df.index
        ax.fill_between(t, vessel_df["Hs_min"], vessel_df["Hs_max"],
                        alpha=0.2, color="C0", label="Heave Hs range (all dirs)")
        ax.plot(t, vessel_df["Hs"], "o-", markersize=3, linewidth=1.0,
                color="C0", label="Heave Hs (median)")

    if nora3_df is not None and "hs" in nora3_df.columns:
        ax.plot(nora3_df.index, nora3_df["hs"], "s-", markersize=5,
                linewidth=1.5, color="C3", label="NORA3 Hs")
        if "hs_swell" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["hs_swell"], "^--", markersize=4,
                    linewidth=0.8, color="C3", alpha=0.6, label="NORA3 Hs_swell")
        if "hs_sea" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["hs_sea"], "v--", markersize=4,
                    linewidth=0.8, color="C1", alpha=0.6, label="NORA3 Hs_sea")

    ax.set_ylabel("Hs [m]")
    ax.set_title(f"Hindcast Validation: {vessel_name} vs NORA3")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: Tp comparison ---
    ax = axes[1]
    if not vessel_df.empty:
        ax.plot(vessel_df.index, vessel_df["Tp"], "o-", markersize=3,
                linewidth=1.0, color="C0", label="Heave Tp")

    if nora3_df is not None and "tp" in nora3_df.columns:
        ax.plot(nora3_df.index, nora3_df["tp"], "s-", markersize=5,
                linewidth=1.5, color="C3", label="NORA3 Tp")
        if "tp_swell" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["tp_swell"], "^--", markersize=4,
                    linewidth=0.8, color="C3", alpha=0.6, label="NORA3 Tp_swell")

    ax.set_ylabel("Tp [s]")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 3: Heading ---
    ax = axes[2]
    t_min = (heading_series.index - heading_series.index[0]).total_seconds() / 60
    # Downsample heading for plotting
    hdg_plot = heading_series.resample("30s").median()
    ax.plot(hdg_plot.index, hdg_plot.values, linewidth=0.5, color="C2")
    if not vessel_df.empty:
        # Mark valid estimation windows
        ax.plot(vessel_df.index, vessel_df["heading_mean"], "o",
                markersize=4, color="C0", label="Est. window (valid)")
    ax.set_ylabel("Heading [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 360)

    # --- Panel 4: NORA3 wave direction + wind ---
    ax = axes[3]
    if nora3_df is not None:
        if "thq" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["thq"], "s-", markersize=5,
                    linewidth=1.0, color="C3", label="NORA3 mean wave dir")
        if "Pdir" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["Pdir"], "d-", markersize=4,
                    linewidth=0.8, color="C4", label="NORA3 peak dir")
        if "dd" in nora3_df.columns:
            ax.plot(nora3_df.index, nora3_df["dd"], "x-", markersize=4,
                    linewidth=0.8, color="C5", alpha=0.7, label="NORA3 wind dir")
    ax.set_ylabel("Direction [deg]")
    ax.set_xlabel("Time (UTC)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 360)

    # Format x-axis
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_locator(mdates.HourLocator())
        a.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))

    # Add info text
    info_lines = []
    if nora3_meta:
        info_lines.append(f"NORA3 grid point: {nora3_meta.get('lat', 0):.3f}N "
                          f"{nora3_meta.get('lon', 0):.3f}E "
                          f"(dist={nora3_meta.get('dist', 0):.3f} deg)")
    if not vessel_df.empty:
        info_lines.append(f"Vessel windows: {len(vessel_df)} valid "
                          f"(30-min window, 10-min step, heading std < 20 deg)")
    if info_lines:
        fig.text(0.5, 0.01, "  |  ".join(info_lines), ha="center", fontsize=8,
                 style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = out_dir / "07_hindcast_validation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return path


def plot_scatter(vessel_df: pd.DataFrame, nora3_df: pd.DataFrame,
                 out_dir: Path, vessel_name: str = "Vessel"):
    """Scatter plot: vessel Hs vs NORA3 Hs at matching times."""
    if vessel_df.empty or nora3_df is None or nora3_df.empty:
        return

    # Match vessel estimates to nearest NORA3 hourly time
    matched = []
    for t, row in vessel_df.iterrows():
        # Find nearest NORA3 time within 30 min
        dt = np.abs((nora3_df.index - t).total_seconds())
        if np.min(dt) < 1800:
            i_nearest = np.argmin(dt)
            matched.append({
                "time": t,
                "Hs_vessel": row["Hs"],
                "Hs_nora3": nora3_df.iloc[i_nearest]["hs"],
                "Tp_vessel": row["Tp"],
                "Tp_nora3": nora3_df.iloc[i_nearest]["tp"],
            })

    if len(matched) < 2:
        print("  Not enough matched points for scatter plot")
        return

    mdf = pd.DataFrame(matched)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Hs scatter
    ax = axes[0]
    ax.scatter(mdf["Hs_nora3"], mdf["Hs_vessel"], s=30, alpha=0.7, edgecolors="C0")
    lim = max(mdf["Hs_nora3"].max(), mdf["Hs_vessel"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.5, label="1:1")
    ax.set_xlabel("NORA3 Hs [m]")
    ax.set_ylabel(f"{vessel_name} Hs [m]")
    ax.set_title("Hs Comparison")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Statistics
    bias = np.mean(mdf["Hs_vessel"] - mdf["Hs_nora3"])
    rmse = np.sqrt(np.mean((mdf["Hs_vessel"] - mdf["Hs_nora3"])**2))
    si = rmse / np.mean(mdf["Hs_nora3"]) if np.mean(mdf["Hs_nora3"]) > 0 else 0
    corr = np.corrcoef(mdf["Hs_vessel"], mdf["Hs_nora3"])[0, 1] if len(mdf) > 2 else np.nan
    ax.text(0.05, 0.95, f"N={len(mdf)}\nBias={bias:+.3f} m\nRMSE={rmse:.3f} m\n"
            f"SI={si:.2f}\nR={corr:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=8)

    # Tp scatter
    ax = axes[1]
    ax.scatter(mdf["Tp_nora3"], mdf["Tp_vessel"], s=30, alpha=0.7,
               edgecolors="C1", facecolors="C1")
    lim_t = max(mdf["Tp_nora3"].max(), mdf["Tp_vessel"].max()) * 1.1
    ax.plot([0, lim_t], [0, lim_t], "k--", linewidth=0.5, label="1:1")
    ax.set_xlabel("NORA3 Tp [s]")
    ax.set_ylabel(f"{vessel_name} Tp [s]")
    ax.set_title("Tp Comparison")
    ax.set_xlim(0, lim_t)
    ax.set_ylim(0, lim_t)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    bias_t = np.mean(mdf["Tp_vessel"] - mdf["Tp_nora3"])
    rmse_t = np.sqrt(np.mean((mdf["Tp_vessel"] - mdf["Tp_nora3"])**2))
    corr_t = np.corrcoef(mdf["Tp_vessel"], mdf["Tp_nora3"])[0, 1] if len(mdf) > 2 else np.nan
    ax.text(0.05, 0.95, f"N={len(mdf)}\nBias={bias_t:+.2f} s\nRMSE={rmse_t:.2f} s\n"
            f"R={corr_t:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=8)

    fig.suptitle(f"Hindcast Validation: {vessel_name} vs NORA3", fontsize=12)
    plt.tight_layout()
    path = out_dir / "08_scatter_validation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate vessel wave-buoy estimates against NORA3 hindcast")
    parser.add_argument("csv", help="Path to exported vessel data CSV")
    parser.add_argument("--rao", required=True,
                        help="Path to PdStrip pdstrip.dat file")
    parser.add_argument("--nora3-month", type=str, default=None,
                        help="NORA3 month to fetch (YYYYMM). Auto-detected if omitted.")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Vessel speed for RAO selection (m/s)")
    parser.add_argument("--fs", type=float, default=1.0,
                        help="Resample frequency (Hz)")
    parser.add_argument("--window-sec", type=float, default=1800.0,
                        help="Sliding window length (seconds, default=1800)")
    parser.add_argument("--step-sec", type=float, default=600.0,
                        help="Sliding window step (seconds, default=600)")
    parser.add_argument("--segment-sec", type=float, default=512.0,
                        help="Welch segment length (seconds, default=512)")
    parser.add_argument("--min-rao", type=float, default=0.5,
                        help="Minimum heave RAO for valid deconvolution")
    parser.add_argument("--max-heading-std", type=float, default=20.0,
                        help="Max heading std dev to accept window (deg)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots")
    parser.add_argument("--vessel-name", default="Geir",
                        help="Vessel name for plot labels")
    parser.add_argument("--no-nora3", action="store_true",
                        help="Skip NORA3 fetch (vessel-only analysis)")
    parser.add_argument("--nora3-subset", action="store_true",
                        help="Use NORA3 monthly subset files instead of full domain")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).parent / "plots" / "wave_buoy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse vessel data ---
    print("=" * 60)
    print("Parsing vessel data ...")
    print("=" * 60)
    signals = parse_csv(str(csv_path))

    # --- Resample ---
    print(f"\nResampling to {args.fs} Hz ...")
    heave = resample_uniform(signals["VesselHeave"], fs=args.fs)
    heading = resample_uniform(signals["HeadingDeg"], fs=args.fs)
    lat_series = resample_uniform(signals["Latitude"], fs=0.1)
    lon_series = resample_uniform(signals["Longitude"], fs=0.1)

    lat_mean = lat_series.median()
    lon_mean = lon_series.median()
    t_start = heave.index[0]
    t_end = heave.index[-1]
    duration_hr = (t_end - t_start).total_seconds() / 3600

    print(f"\nVessel data:")
    print(f"  Position: {lat_mean:.4f}N, {lon_mean:.4f}E")
    print(f"  Time: {t_start} to {t_end} ({duration_hr:.1f} hours)")
    print(f"  Heave samples: {len(heave):,}")

    # --- Load RAOs ---
    print("\n" + "=" * 60)
    print("Loading RAOs ...")
    print("=" * 60)
    rao = load_rao(args.rao, speed_ms=args.speed)

    # --- Sliding-window wave estimation ---
    print("\n" + "=" * 60)
    print(f"Running sliding-window wave estimation "
          f"(window={args.window_sec:.0f}s, step={args.step_sec:.0f}s) ...")
    print("=" * 60)
    vessel_df = sliding_wave_estimate(
        heave, heading, rao,
        fs=args.fs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        segment_sec=args.segment_sec,
        min_rao=args.min_rao,
        max_heading_std=args.max_heading_std,
    )

    if not vessel_df.empty:
        print(f"\n  Vessel estimate summary:")
        print(f"    Hs: {vessel_df['Hs'].min():.2f} - {vessel_df['Hs'].max():.2f} m "
              f"(mean={vessel_df['Hs'].mean():.2f} m)")
        print(f"    Tp: {vessel_df['Tp'].min():.1f} - {vessel_df['Tp'].max():.1f} s")
        print(f"    Time span: {vessel_df.index[0]} to {vessel_df.index[-1]}")
    else:
        print("  WARNING: No valid estimation windows!")

    # --- Fetch NORA3 ---
    nora3_df = None
    nora3_meta = None
    if not args.no_nora3:
        print("\n" + "=" * 60)
        print("Fetching NORA3 hindcast ...")
        print("=" * 60)

        # Auto-detect month from vessel data
        nora3_month = args.nora3_month
        if nora3_month is None:
            nora3_month = t_start.strftime("%Y%m")
            print(f"  Auto-detected NORA3 month: {nora3_month}")

        try:
            # Add buffer around vessel time for NORA3
            t_start_str = (t_start - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            t_end_str = (t_end + pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            nora3_df = fetch_nora3_nearest(
                nora3_month, lat_mean, lon_mean,
                time_start=t_start_str, time_end=t_end_str,
                use_full_domain=not args.nora3_subset)

            nora3_meta = {
                "lat": nora3_df.attrs.get("nearest_lat", 0),
                "lon": nora3_df.attrs.get("nearest_lon", 0),
                "dist": nora3_df.attrs.get("dist_deg", 0),
            }

            if not nora3_df.empty:
                print(f"\n  NORA3 summary (nearest point):")
                if "hs" in nora3_df.columns:
                    print(f"    Hs: {nora3_df['hs'].min():.2f} - "
                          f"{nora3_df['hs'].max():.2f} m")
                if "tp" in nora3_df.columns:
                    print(f"    Tp: {nora3_df['tp'].min():.1f} - "
                          f"{nora3_df['tp'].max():.1f} s")
                if "thq" in nora3_df.columns:
                    print(f"    Mean wave dir: {nora3_df['thq'].mean():.0f} deg")
        except Exception as e:
            print(f"  WARNING: Failed to fetch NORA3: {e}")
            nora3_df = None

    # --- Plots ---
    print("\n" + "=" * 60)
    print("Creating validation plots ...")
    print("=" * 60)

    plot_validation(vessel_df, nora3_df, heading, out_dir,
                    vessel_name=args.vessel_name, nora3_meta=nora3_meta)

    if nora3_df is not None:
        plot_scatter(vessel_df, nora3_df, out_dir, vessel_name=args.vessel_name)

    # --- Print matched comparison table ---
    if nora3_df is not None and not vessel_df.empty and "hs" in nora3_df.columns:
        print("\n" + "=" * 60)
        print("Matched comparison (vessel window centre vs nearest NORA3 hour):")
        print("=" * 60)
        print(f"{'Time (UTC)':>20s}  {'Hs_vessel':>10s}  {'Hs_NORA3':>10s}  "
              f"{'Tp_vessel':>10s}  {'Tp_NORA3':>10s}  {'Heading':>8s}")
        print("-" * 75)
        for t, row in vessel_df.iterrows():
            dt = np.abs((nora3_df.index - t).total_seconds())
            if np.min(dt) < 1800:
                i_nearest = np.argmin(dt)
                nr = nora3_df.iloc[i_nearest]
                print(f"{str(t):>20s}  {row['Hs']:10.2f}  {nr['hs']:10.2f}  "
                      f"{row['Tp']:10.1f}  {nr['tp']:10.1f}  {row['heading_mean']:8.0f}")

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
