"""
Plot hourly surface and depth-resolved currents at Tampen
from NorKyst v3 800m hindcast (best-estimate aggregation).

Tampen area: ~61.0-61.5°N, 1.5-2.5°E (Snorre/Gullfaks region,
northern North Sea shelf edge).

Produces:
  1. Surface current snapshots (4 panels for 00, 06, 12, 18 UTC)
  2. Current speed vs depth vs time (time-depth heatmap at a point)
  3. Current direction vs depth vs time (stick plot or quiver)
  4. Hourly current speed timeseries at surface, 50m, 100m, 200m, 300m
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import xarray as xr

from common import NORKYST_800M_BE

OUT_DIR = Path(__file__).parent / "plots" / "tampen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tampen bounding box
LAT_MIN, LAT_MAX = 61.0, 61.5
LON_MIN, LON_MAX = 1.5, 3.0

# Hywind Tampen floating wind farm (11 x 8.6 MW, Equinor)
HYWIND_LAT, HYWIND_LON = 61.330, 2.700

# Hywind zoom: ~5 km radius around the wind farm (~0.045° lat, ~0.09° lon)
HYWIND_RADIUS_LAT = 0.045
HYWIND_RADIUS_LON = 0.09


def find_bbox_indices(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    buf: int = 5,
    label: str = "Region",
) -> dict:
    """Find Y/X index ranges that cover a geographic bounding box."""
    mask = (
        (lat2d >= lat_min) & (lat2d <= lat_max) &
        (lon2d >= lon_min) & (lon2d <= lon_max)
    )

    if not mask.any():
        print(f"ERROR: No grid points found in {label} bounding box!")
        print(f"  lat range in data: {np.nanmin(lat2d):.2f} - {np.nanmax(lat2d):.2f}")
        print(f"  lon range in data: {np.nanmin(lon2d):.2f} - {np.nanmax(lon2d):.2f}")
        sys.exit(1)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_start, y_end = np.where(rows)[0][[0, -1]]
    x_start, x_end = np.where(cols)[0][[0, -1]]

    y_start = max(0, y_start - buf)
    y_end = min(lat2d.shape[0] - 1, y_end + buf)
    x_start = max(0, x_start - buf)
    x_end = min(lat2d.shape[1] - 1, x_end + buf)

    print(f"  {label} indices: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")
    print(f"  Grid points: {y_end - y_start} x {x_end - x_start}")

    return {
        "Y": slice(y_start, y_end + 1),
        "X": slice(x_start, x_end + 1),
    }


def find_tampen_indices(ds: xr.Dataset) -> dict:
    """Find Y/X index ranges that cover the Tampen area."""
    return find_bbox_indices(
        ds["lat"].values, ds["lon"].values,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        buf=5, label="Tampen",
    )


def pick_random_day(ds: xr.Dataset) -> str:
    """Pick a date roughly in the middle of the available time range."""
    times = ds["time"].values
    mid = len(times) // 2
    date = str(times[mid])[:10]
    print(f"  Selected date: {date}")
    return date


def main():
    print("=" * 60)
    print("Tampen Area — Hourly Current Profiles (NorKyst v3 800m)")
    print("=" * 60)

    url = NORKYST_800M_BE
    print(f"\nOpening: {url}")
    ds = xr.open_dataset(url)

    print(f"  Time range: {str(ds['time'].values[0])[:10]} to "
          f"{str(ds['time'].values[-1])[:10]}")
    print(f"  Depths: {ds['depth'].values}")

    # Find spatial indices for Tampen
    idx = find_tampen_indices(ds)

    # Pick a day
    target_date = pick_random_day(ds)

    # Select 24 hours of data for Tampen region
    print(f"\nLoading 24h of data for {target_date} ...")
    time_sel = slice(f"{target_date}T00:00", f"{target_date}T23:00")

    sub = ds.sel(time=time_sel).isel(Y=idx["Y"], X=idx["X"])

    # Load into memory (this is the actual OPeNDAP fetch)
    print("  Fetching u_eastward ...")
    u = sub["u_eastward"].values      # (time, depth, Y, X)
    print("  Fetching v_northward ...")
    v = sub["v_northward"].values
    print("  Fetching temperature ...")
    temp = sub["temperature"].values
    lat = sub["lat"].values           # (Y, X)
    lon = sub["lon"].values
    times = sub["time"].values
    depths = sub["depth"].values

    ds.close()

    speed = np.sqrt(u**2 + v**2)
    n_times = len(times)
    print(f"  Loaded: {n_times} time steps, {len(depths)} depths, "
          f"{lat.shape[0]}x{lat.shape[1]} spatial points")

    # ---- Plot 1: Surface current snapshots at 4 times ----
    print("\n[1/5] Plotting surface current snapshots ...")
    plot_surface_snapshots(lat, lon, u, v, speed, times, depths, target_date)

    # ---- Plot 2: Time-depth current speed at center point ----
    print("[2/5] Plotting current speed vs depth vs time ...")
    cy, cx = lat.shape[0] // 2, lat.shape[1] // 2
    center_lat = lat[cy, cx]
    center_lon = lon[cy, cx]
    plot_time_depth_speed(speed[:, :, cy, cx], times, depths,
                          center_lat, center_lon, target_date)

    # ---- Plot 3: Current direction vs depth vs time ----
    print("[3/5] Plotting current direction vs depth vs time ...")
    plot_time_depth_direction(u[:, :, cy, cx], v[:, :, cy, cx],
                             speed[:, :, cy, cx], times, depths,
                             center_lat, center_lon, target_date)

    # ---- Plot 4: Timeseries at selected depths ----
    print("[4/5] Plotting current speed timeseries at depth levels ...")
    plot_depth_timeseries(u[:, :, cy, cx], v[:, :, cy, cx],
                          speed[:, :, cy, cx], times, depths,
                          center_lat, center_lon, target_date)

    # ---- Plot 5: Zoomed Hywind Tampen with full-resolution vectors ----
    print("[5/5] Plotting zoomed Hywind Tampen snapshots ...")
    plot_hywind_zoom_snapshots(lat, lon, u, v, speed, times, target_date)

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")


def plot_surface_snapshots(lat, lon, u, v, speed, times, depths, date):
    """4-panel surface current maps at 00, 06, 12, 18 UTC."""
    n_times = len(times)
    # Pick 4 evenly spaced time indices
    if n_times >= 4:
        indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4]
    else:
        indices = list(range(min(n_times, 4)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    vmax = np.nanpercentile(speed[:, 0, :, :], 98)

    for i, tidx in enumerate(indices):
        ax = axes[i]
        spd = speed[tidx, 0, :, :]  # surface (depth=0)
        uu = u[tidx, 0, :, :]
        vv = v[tidx, 0, :, :]
        t_str = str(times[tidx])[:16]

        pcm = ax.pcolormesh(lon, lat, spd, cmap="cividis",
                            shading="auto", vmin=0, vmax=max(vmax, 0.1))

        # Quiver (subsample)
        skip = max(1, lat.shape[0] // 15)
        ax.quiver(
            lon[::skip, ::skip], lat[::skip, ::skip],
            uu[::skip, ::skip], vv[::skip, ::skip],
            color="white", alpha=0.8, scale=5, width=0.004,
        )

        ax.set_title(f"{t_str} UTC", fontsize=10)
        ax.set_xlabel("Lon [°E]")
        ax.set_ylabel("Lat [°N]")
        ax.set_aspect("equal")

        # Hywind Tampen marker
        ax.plot(HYWIND_LON, HYWIND_LAT, marker="*", color="red",
                markersize=12, markeredgecolor="white",
                markeredgewidth=0.8, zorder=10)
        if i == 0:  # label on first panel only to avoid clutter
            ax.annotate(
                "Hywind Tampen", (HYWIND_LON, HYWIND_LAT),
                textcoords="offset points", xytext=(6, -10),
                fontsize=7, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    fig.suptitle(f"Tampen — Surface Currents (NorKyst v3 800m)\n{date}",
                 fontsize=13, fontweight="bold")
    fig.colorbar(pcm, ax=axes, shrink=0.6, label="Current speed [m/s]")

    out = OUT_DIR / "01_tampen_surface_snapshots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_time_depth_speed(speed_zt, times, depths, lat, lon, date):
    """Time-depth cross-section of current speed at center point."""
    # speed_zt shape: (time, depth)
    hours = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(14, 6))

    pcm = ax.pcolormesh(
        hours, depths, speed_zt.T,
        cmap="magma_r", shading="auto",
        vmin=0, vmax=np.nanpercentile(speed_zt, 98),
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Current speed [m/s]")

    ax.invert_yaxis()
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Depth [m]")
    ax.set_title(
        f"Tampen — Current Speed Profile\n"
        f"{date} at ({lat:.2f}°N, {lon:.2f}°E)\n"
        f"NorKyst v3 800m, hourly",
        fontsize=12,
    )
    ax.set_xticks(hours[::2])
    ax.set_xticklabels([str(t)[-8:-3] if len(str(t)) > 8 else str(i)
                        for i, t in enumerate(times) if i % 2 == 0],
                       rotation=45, fontsize=8)
    # Better: just use hour numbers
    ax.set_xticks(hours)
    ax.set_xticklabels([str(t)[11:16] for t in times], rotation=45, fontsize=7)

    out = OUT_DIR / "02_tampen_speed_depth_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_time_depth_direction(u_zt, v_zt, speed_zt, times, depths, lat, lon, date):
    """Current direction shown as quiver arrows on a time-depth grid."""
    hours = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Background: speed
    pcm = ax.pcolormesh(
        hours, depths, speed_zt.T,
        cmap="Blues", shading="auto", alpha=0.4,
        vmin=0, vmax=np.nanpercentile(speed_zt, 98),
    )

    # Quiver arrows: u/v at each (time, depth)
    hh, dd = np.meshgrid(hours, depths)
    # Normalize arrows to unit length for direction display
    spd = np.sqrt(u_zt.T**2 + v_zt.T**2)
    spd = np.where(spd > 0, spd, 1)
    ax.quiver(
        hh, dd, u_zt.T / spd, v_zt.T / spd,
        speed_zt.T, cmap="cividis",
        scale=25, width=0.003, headwidth=3,
    )

    ax.invert_yaxis()
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Depth [m]")
    ax.set_title(
        f"Tampen — Current Direction & Speed Profile\n"
        f"{date} at ({lat:.2f}°N, {lon:.2f}°E)\n"
        f"NorKyst v3 800m, hourly — arrows show direction, color shows speed",
        fontsize=11,
    )
    ax.set_xticks(hours)
    ax.set_xticklabels([str(t)[11:16] for t in times], rotation=45, fontsize=7)

    cbar = fig.colorbar(pcm, ax=ax, shrink=0.8)
    cbar.set_label("Current speed [m/s]")

    out = OUT_DIR / "03_tampen_direction_depth_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_depth_timeseries(u_zt, v_zt, speed_zt, times, depths, lat, lon, date):
    """Current speed timeseries at selected depth levels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Select depths closest to 0, 50, 100, 200, 300m
    target_depths = [0, 50, 100, 200, 300]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    hours = np.arange(len(times))

    for target_d, color in zip(target_depths, colors):
        didx = np.argmin(np.abs(depths - target_d))
        actual_d = depths[didx]
        spd = speed_zt[:, didx]
        direction = np.rad2deg(np.arctan2(u_zt[:, didx], v_zt[:, didx])) % 360

        ax1.plot(hours, spd, color=color, linewidth=1.5,
                 label=f"{actual_d:.0f}m", marker=".", markersize=4)
        ax2.plot(hours, direction, color=color, linewidth=1.0,
                 label=f"{actual_d:.0f}m", marker=".", markersize=3)

    ax1.set_ylabel("Current speed [m/s]")
    ax1.set_title(
        f"Tampen — Hourly Current at ({lat:.2f}°N, {lon:.2f}°E)\n"
        f"{date} — NorKyst v3 800m",
        fontsize=12,
    )
    ax1.legend(title="Depth", loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Current direction [°] (towards)")
    ax2.set_xlabel("Hour of day (UTC)")
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["N", "E", "S", "W", "N"])
    ax2.legend(title="Depth", loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks(hours)
    ax2.set_xticklabels([str(t)[11:16] for t in times], rotation=45, fontsize=8)

    out = OUT_DIR / "04_tampen_depth_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_hywind_zoom_snapshots(lat, lon, u, v, speed, times, date):
    """4-panel zoomed surface current maps centred on Hywind Tampen,
    with full-resolution quiver arrows (~300 m grid spacing)."""
    # Extract zoom sub-region from the already-loaded Tampen data
    hw_lat_min = HYWIND_LAT - HYWIND_RADIUS_LAT
    hw_lat_max = HYWIND_LAT + HYWIND_RADIUS_LAT
    hw_lon_min = HYWIND_LON - HYWIND_RADIUS_LON
    hw_lon_max = HYWIND_LON + HYWIND_RADIUS_LON

    mask = (
        (lat >= hw_lat_min) & (lat <= hw_lat_max) &
        (lon >= hw_lon_min) & (lon <= hw_lon_max)
    )
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum() == 0 or cols.sum() == 0:
        print("  WARNING: Hywind zoom area outside loaded Tampen data, skipping.")
        return

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = r_idx[0], r_idx[-1] + 1
    c0, c1 = c_idx[0], c_idx[-1] + 1

    z_lat = lat[r0:r1, c0:c1]
    z_lon = lon[r0:r1, c0:c1]
    z_u = u[:, 0, r0:r1, c0:c1]      # surface only
    z_v = v[:, 0, r0:r1, c0:c1]
    z_spd = speed[:, 0, r0:r1, c0:c1]

    print(f"  Hywind zoom: {z_lat.shape[0]} x {z_lat.shape[1]} grid points "
          f"(full resolution, no subsampling)")

    n_times = len(times)
    if n_times >= 4:
        indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4]
    else:
        indices = list(range(min(n_times, 4)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    vmax = float(np.nanpercentile(z_spd, 98))
    vmax = max(vmax, 0.05)

    for i, tidx in enumerate(indices):
        ax = axes[i]
        spd = z_spd[tidx]
        uu = z_u[tidx]
        vv = z_v[tidx]
        t_str = str(times[tidx])[:16]

        pcm = ax.pcolormesh(z_lon, z_lat, spd, cmap="cividis",
                            shading="auto", vmin=0, vmax=vmax)

        # Full-resolution quiver — every grid cell
        ax.quiver(
            z_lon, z_lat, uu, vv,
            color="white", alpha=0.8, scale=3, width=0.004,
            headwidth=4, headlength=5,
        )

        ax.set_title(f"{t_str} UTC", fontsize=10)
        ax.set_xlabel("Lon [°E]")
        ax.set_ylabel("Lat [°N]")
        ax.set_aspect("equal")

        # Hywind Tampen marker
        ax.plot(HYWIND_LON, HYWIND_LAT, marker="*", color="red",
                markersize=14, markeredgecolor="white",
                markeredgewidth=0.8, zorder=10)
        if i == 0:
            ax.annotate(
                "Hywind Tampen", (HYWIND_LON, HYWIND_LAT),
                textcoords="offset points", xytext=(8, -12),
                fontsize=8, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    fig.suptitle(
        f"Hywind Tampen — Surface Currents (NorKyst v3, ~300 m resolution)\n{date}",
        fontsize=13, fontweight="bold",
    )
    fig.colorbar(pcm, ax=axes, shrink=0.6, label="Current speed [m/s]")

    out = OUT_DIR / "05_hywind_zoom_snapshots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
