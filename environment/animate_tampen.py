"""
Animate hourly surface currents at Tampen over 5 days (120 frames)
from NorKyst v3 800m hindcast (best-estimate aggregation).

Tampen area: ~61.0-61.5°N, 1.5-3.0°E (Snorre/Gullfaks region,
northern North Sea shelf edge).

Produces:
  1. Wide-area Tampen animation — surface currents with subsampled quiver
     (120 frames at 4 fps = 30 s)
  2. Zoomed Hywind Tampen animation — full-resolution ~300 m quiver arrows
     centred on the wind farm (120 frames at 4 fps = 30 s)

Output:
  plots/tampen/05_tampen_surface_animation.mp4
  plots/tampen/06_hywind_zoom_animation.mp4
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

FPS = 4
DPI = 150
ANIMATION_DAYS = 5  # number of days for the animation


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


def pick_date_range(ds: xr.Dataset, n_days: int = ANIMATION_DAYS) -> tuple[str, str]:
    """Pick a start date roughly in the middle of the available time range,
    returning (start_date, end_date) strings covering *n_days* days."""
    times = ds["time"].values
    mid = len(times) // 2
    start_date = str(times[mid])[:10]
    # Compute end date
    from datetime import datetime, timedelta
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    d1 = d0 + timedelta(days=n_days - 1)
    end_date = d1.strftime("%Y-%m-%d")
    print(f"  Selected date range: {start_date} to {end_date} ({n_days} days)")
    return start_date, end_date


def _choose_writer(out_name: str) -> tuple:
    """Pick ffmpeg (MP4) or pillow (GIF) writer and return (writer, out_path)."""
    if shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=FPS, bitrate=1800)
        out_path = OUT_DIR / f"{out_name}.mp4"
    else:
        print("  WARNING: ffmpeg not found — falling back to GIF (PillowWriter)")
        writer = animation.PillowWriter(fps=FPS)
        out_path = OUT_DIR / f"{out_name}.gif"
    return writer, out_path


def _save_animation(anim_obj, out_name: str, n_times: int):
    """Render animation to disk and print summary."""
    writer, out_path = _choose_writer(out_name)
    print(f"\nRendering to: {out_path}")
    anim_obj.save(str(out_path), writer=writer, dpi=DPI)
    print(f"  Saved: {out_path.resolve()}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Duration: {n_times / FPS:.1f} s ({n_times} frames at {FPS} fps)")
    return out_path


def animate_wide_tampen(lat, lon, u_sfc, v_sfc, speed, times,
                        start_date, end_date):
    """Build and save the wide-area Tampen surface current animation."""
    n_times = len(times)

    vmax = float(np.nanpercentile(speed, 98))
    vmax = max(vmax, 0.1)
    print(f"\n[1/2] Wide Tampen animation ({n_times} frames at {FPS} fps)")
    print(f"  Color scale: 0 - {vmax:.3f} m/s (98th percentile)")

    skip = max(1, lat.shape[0] // 15)

    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = ax.pcolormesh(lon, lat, speed[0], cmap="cividis",
                        shading="auto", vmin=0, vmax=vmax)
    qv = ax.quiver(
        lon[::skip, ::skip], lat[::skip, ::skip],
        u_sfc[0, ::skip, ::skip], v_sfc[0, ::skip, ::skip],
        color="white", alpha=0.8, scale=5, width=0.004,
    )
    ax.set_xlabel("Lon [°E]")
    ax.set_ylabel("Lat [°N]")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, shrink=0.8, label="Current speed [m/s]")

    # Hywind Tampen marker
    ax.plot(HYWIND_LON, HYWIND_LAT, marker="*", color="red", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.annotate(
        "Hywind Tampen", (HYWIND_LON, HYWIND_LAT),
        textcoords="offset points", xytext=(8, -12),
        fontsize=9, color="white", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )

    t_str = str(times[0])[:16].replace("T", " ")
    title = ax.set_title(
        f"Tampen — Surface Currents ({start_date} to {end_date})\n{t_str} UTC",
        fontsize=12, fontweight="bold",
    )

    def update(frame: int):
        pcm.set_array(speed[frame].ravel())
        qv.set_UVC(u_sfc[frame, ::skip, ::skip],
                    v_sfc[frame, ::skip, ::skip])
        t_str = str(times[frame])[:16].replace("T", " ")
        title.set_text(
            f"Tampen — Surface Currents ({start_date} to {end_date})\n{t_str} UTC"
        )
        if frame % 12 == 0 or frame == n_times - 1:
            print(f"  Frame {frame + 1:3d}/{n_times}: {t_str} UTC")
        return pcm, qv, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_times, blit=False, interval=1000 // FPS,
    )
    _save_animation(anim, "05_tampen_surface_animation", n_times)
    plt.close(fig)


def animate_hywind_zoom(lat, lon, u_sfc, v_sfc, speed, times,
                        start_date, end_date):
    """Build and save the zoomed Hywind Tampen animation with full-resolution
    quiver arrows (~300 m grid spacing)."""
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
        print("\n[2/2] WARNING: Hywind zoom area outside loaded data, skipping.")
        return

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = r_idx[0], r_idx[-1] + 1
    c0, c1 = c_idx[0], c_idx[-1] + 1

    z_lat = lat[r0:r1, c0:c1]
    z_lon = lon[r0:r1, c0:c1]
    z_u = u_sfc[:, r0:r1, c0:c1]
    z_v = v_sfc[:, r0:r1, c0:c1]
    z_spd = speed[:, r0:r1, c0:c1]

    n_times = len(times)
    vmax = float(np.nanpercentile(z_spd, 98))
    vmax = max(vmax, 0.05)

    print(f"\n[2/2] Hywind zoom animation ({n_times} frames at {FPS} fps)")
    print(f"  Grid: {z_lat.shape[0]} x {z_lat.shape[1]} points "
          f"(full resolution, no subsampling)")
    print(f"  Color scale: 0 - {vmax:.3f} m/s (98th percentile)")

    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = ax.pcolormesh(z_lon, z_lat, z_spd[0], cmap="cividis",
                        shading="auto", vmin=0, vmax=vmax)
    qv = ax.quiver(
        z_lon, z_lat, z_u[0], z_v[0],
        color="white", alpha=0.8, scale=3, width=0.004,
        headwidth=4, headlength=5,
    )
    ax.set_xlabel("Lon [°E]")
    ax.set_ylabel("Lat [°N]")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, shrink=0.8, label="Current speed [m/s]")

    # Hywind Tampen marker
    ax.plot(HYWIND_LON, HYWIND_LAT, marker="*", color="red", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.annotate(
        "Hywind Tampen", (HYWIND_LON, HYWIND_LAT),
        textcoords="offset points", xytext=(8, -12),
        fontsize=9, color="white", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )

    t_str = str(times[0])[:16].replace("T", " ")
    title = ax.set_title(
        f"Hywind Tampen — Surface Currents (~300 m resolution)\n{t_str} UTC",
        fontsize=12, fontweight="bold",
    )

    def update(frame: int):
        pcm.set_array(z_spd[frame].ravel())
        qv.set_UVC(z_u[frame], z_v[frame])
        t_str = str(times[frame])[:16].replace("T", " ")
        title.set_text(
            f"Hywind Tampen — Surface Currents (~300 m resolution)\n{t_str} UTC"
        )
        if frame % 12 == 0 or frame == n_times - 1:
            print(f"  Frame {frame + 1:3d}/{n_times}: {t_str} UTC")
        return pcm, qv, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_times, blit=False, interval=1000 // FPS,
    )
    _save_animation(anim, "06_hywind_zoom_animation", n_times)
    plt.close(fig)


def main():
    print("=" * 60)
    print("Tampen Area — Surface Current Animation (NorKyst v3 800m)")
    print(f"  {ANIMATION_DAYS}-day animation, wide + Hywind zoom")
    print("=" * 60)

    url = NORKYST_800M_BE
    print(f"\nOpening: {url}")
    ds = xr.open_dataset(url)

    print(f"  Time range: {str(ds['time'].values[0])[:10]} to "
          f"{str(ds['time'].values[-1])[:10]}")

    # Find spatial indices for Tampen
    idx = find_tampen_indices(ds)

    # Pick a date range
    start_date, end_date = pick_date_range(ds)

    # Select multi-day data for Tampen region
    print(f"\nLoading {ANIMATION_DAYS} days of data ({start_date} to {end_date}) ...")
    time_sel = slice(f"{start_date}T00:00", f"{end_date}T23:00")

    sub = ds.sel(time=time_sel).isel(Y=idx["Y"], X=idx["X"])

    # Load into memory (this is the actual OPeNDAP fetch)
    print("  Fetching u_eastward ...")
    u = sub["u_eastward"].values      # (time, depth, Y, X)
    print("  Fetching v_northward ...")
    v = sub["v_northward"].values
    lat = sub["lat"].values           # (Y, X)
    lon = sub["lon"].values
    times = sub["time"].values

    ds.close()

    # Surface layer only (depth index 0)
    u_sfc = u[:, 0, :, :]  # (time, Y, X)
    v_sfc = v[:, 0, :, :]
    speed = np.sqrt(u_sfc**2 + v_sfc**2)

    n_times = len(times)
    print(f"  Loaded: {n_times} time steps, "
          f"{lat.shape[0]}x{lat.shape[1]} spatial points")

    # ---- Animation 1: Wide-area Tampen ----
    animate_wide_tampen(lat, lon, u_sfc, v_sfc, speed, times, start_date, end_date)

    # ---- Animation 2: Zoomed Hywind Tampen ----
    animate_hywind_zoom(lat, lon, u_sfc, v_sfc, speed, times, start_date, end_date)


if __name__ == "__main__":
    main()
