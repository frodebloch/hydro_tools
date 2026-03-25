"""
Fetch small data samples from MET Norway THREDDS and produce plots.

Generates:
  1. 2D directional wave spectrum (polar contour) from WW3 spectra
  2. Hs time series for a few stations from WW3 spectra
  3. Ocean current quiver plot from NorKyst v3
  4. Temperature cross-section from NorKyst v3

Run from the environment/ directory:
  source .venv/bin/activate
  python plot_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from common import (
    ww3_spc_url,
    NORKYST_800M_BE,
    WW3_AGG,
    latest_cycle,
)

OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)


# ---- helpers ---------------------------------------------------------------

def safe_open(url: str, timeout_msg: str = "") -> xr.Dataset | None:
    try:
        return xr.open_dataset(url)
    except Exception as e:
        print(f"  ERROR opening {url}: {e}")
        if timeout_msg:
            print(f"  {timeout_msg}")
        return None


# ---- 1. 2D wave spectrum (polar contour) -----------------------------------

def plot_spectrum():
    """Fetch a single station spectrum and plot as polar contour."""
    print("\n[1/4] Fetching WW3 point spectrum ...")
    date, cycle = latest_cycle()
    url = ww3_spc_url("poi", date, cycle)
    print(f"  URL: {url}")

    ds = safe_open(url)
    if ds is None:
        return

    # Pick first time step, first station with nonzero data
    # Dimensions: (time, y=1, x=Nstations, freq=36, direction=36)
    station_idx = 0
    time_idx = 0

    spec = ds["SPEC"].isel(time=time_idx, y=0, x=station_idx).values  # (freq, dir)
    freqs = ds["freq"].values        # Hz
    dirs_deg = ds["direction"].values # degrees
    lat = float(ds["latitude"].isel(y=0, x=station_idx).values)
    lon = float(ds["longitude"].isel(y=0, x=station_idx).values)
    hs = float(ds["hs"].isel(time=time_idx, y=0, x=station_idx).values)
    tp = float(ds["tp"].isel(time=time_idx, y=0, x=station_idx).values)
    time_val = str(ds["time"].isel(time=time_idx).values)[:16]
    ds.close()

    # Convert direction from meteorological "coming from" to radians for polar plot
    dirs_rad = np.deg2rad(dirs_deg)

    # Close the circle: append first direction at the end
    dirs_rad_closed = np.append(dirs_rad, dirs_rad[0] + 2 * np.pi)
    spec_closed = np.concatenate([spec, spec[:, :1]], axis=1)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise

    # Use log scale for energy density
    spec_plot = np.where(spec_closed > 0, spec_closed, np.nan)
    pcm = ax.pcolormesh(
        dirs_rad_closed, freqs, spec_plot,
        cmap="inferno",
        norm=mcolors.LogNorm(
            vmin=np.nanpercentile(spec_plot, 5),
            vmax=np.nanmax(spec_plot),
        ),
        shading="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Spectral density [m²/Hz/rad]")

    ax.set_title(
        f"2D Wave Spectrum — Station {station_idx}\n"
        f"({lat:.2f}°N, {lon:.2f}°E)  Hs={hs:.1f}m  Tp={tp:.1f}s\n"
        f"{time_val} UTC",
        fontsize=11, pad=20,
    )
    ax.set_ylabel("Frequency [Hz]", labelpad=30)

    out = OUT_DIR / "01_wave_spectrum_2d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- 2. Hs time series for a few stations ----------------------------------

def plot_hs_timeseries():
    """Plot Hs forecast time series for a few POI stations."""
    print("\n[2/4] Fetching WW3 Hs time series ...")
    date, cycle = latest_cycle()
    url = ww3_spc_url("poi", date, cycle)

    ds = safe_open(url)
    if ds is None:
        return

    # Pick 5 well-spaced stations
    n_stations = ds.sizes["x"]
    indices = np.linspace(0, n_stations - 1, 5, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx in indices:
        hs = ds["hs"].isel(y=0, x=idx).values
        lat = float(ds["latitude"].isel(y=0, x=idx).values)
        lon = float(ds["longitude"].isel(y=0, x=idx).values)
        times = ds["time"].values
        ax.plot(times, hs, label=f"Stn {idx} ({lat:.1f}°N, {lon:.1f}°E)", linewidth=1.2)

    ds.close()

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Significant Wave Height Hs [m]")
    ax.set_title(f"WW3 4km Wave Forecast — Hs at POI Stations\n{date} {cycle}Z run")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    out = OUT_DIR / "02_hs_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- 3. NorKyst surface current quiver plot --------------------------------

def plot_norkyst_currents():
    """Fetch a small surface current patch from NorKyst and plot quiver."""
    print("\n[3/4] Fetching NorKyst surface currents ...")
    url = NORKYST_800M_BE
    print(f"  URL: {url}")

    ds = safe_open(url, "NorKyst aggregation may be slow to open — be patient")
    if ds is None:
        return

    # Take the latest time step, surface (depth=0), and a spatial subset
    # NorKyst is on a polar stereographic grid; take a ~200x200 km patch
    # centered roughly on the Norwegian west coast
    # Use index-based subsetting since it's curvilinear
    y_slice = slice(400, 550)  # ~120 km span
    x_slice = slice(800, 1000) # ~160 km span

    print("  Subsetting and loading ...")
    u = ds["u_eastward"].isel(time=-1, depth=0, Y=y_slice, X=x_slice).values
    v = ds["v_northward"].isel(time=-1, depth=0, Y=y_slice, X=x_slice).values
    lon = ds["lon"].isel(Y=y_slice, X=x_slice).values
    lat = ds["lat"].isel(Y=y_slice, X=x_slice).values
    time_val = str(ds["time"].isel(time=-1).values)[:16]
    ds.close()

    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Pcolormesh for speed magnitude
    pcm = ax.pcolormesh(lon, lat, speed, cmap="viridis", shading="auto", vmin=0, vmax=1.0)
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.8)
    cbar.set_label("Current speed [m/s]")

    # Quiver (subsample for readability)
    skip = 5
    ax.quiver(
        lon[::skip, ::skip], lat[::skip, ::skip],
        u[::skip, ::skip], v[::skip, ::skip],
        color="white", alpha=0.7, scale=8, width=0.002,
    )

    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")
    ax.set_title(
        f"NorKyst v3 (800m) — Surface Currents\n{time_val} UTC",
        fontsize=12,
    )
    ax.set_aspect("auto")

    out = OUT_DIR / "03_norkyst_surface_currents.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- 4. NorKyst temperature cross-section ----------------------------------

def plot_norkyst_temperature():
    """Plot a depth-vs-latitude temperature cross-section from NorKyst."""
    print("\n[4/4] Fetching NorKyst temperature cross-section ...")
    url = NORKYST_800M_BE

    ds = safe_open(url)
    if ds is None:
        return

    # Take a north-south transect at a fixed X index (roughly mid-coast)
    x_idx = 900
    y_slice = slice(200, 800)

    print("  Subsetting and loading ...")
    temp = ds["temperature"].isel(time=-1, X=x_idx, Y=y_slice).values  # (depth, Y)
    depths = ds["depth"].values
    lats = ds["lat"].isel(X=x_idx, Y=y_slice).values
    time_val = str(ds["time"].isel(time=-1).values)[:16]
    ds.close()

    fig, ax = plt.subplots(figsize=(14, 5))

    # temp shape is (depth, Y), lats is (Y,)
    pcm = ax.pcolormesh(
        lats, depths, temp,
        cmap="RdYlBu_r", shading="auto",
        vmin=np.nanpercentile(temp, 2),
        vmax=np.nanpercentile(temp, 98),
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Temperature [°C]")

    ax.invert_yaxis()
    ax.set_xlabel("Latitude [°N]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(
        f"NorKyst v3 — Temperature Cross-Section (along ~constant longitude)\n"
        f"{time_val} UTC",
        fontsize=11,
    )

    out = OUT_DIR / "04_norkyst_temperature_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- main ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MET Norway Data Sample Plots")
    print("=" * 60)

    plot_spectrum()
    plot_hs_timeseries()
    plot_norkyst_currents()
    plot_norkyst_temperature()

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
