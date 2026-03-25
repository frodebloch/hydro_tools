"""
Animate hourly surface currents at Tampen over a full day (24 frames)
from NorKyst v3 800m hindcast (best-estimate aggregation).

Tampen area: ~61.0-61.5°N, 1.5-2.5°E (Snorre/Gullfaks region,
northern North Sea shelf edge).

Produces:
  - MP4 animation (or GIF fallback) of surface current speed + direction
    at 4 fps → 24 frames = 6 seconds.

Output: plots/tampen/05_tampen_surface_animation.mp4
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr

from common import NORKYST_800M_BE

OUT_DIR = Path(__file__).parent / "plots" / "tampen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tampen bounding box
LAT_MIN, LAT_MAX = 61.0, 61.5
LON_MIN, LON_MAX = 1.5, 2.5

FPS = 4
DPI = 150


def find_tampen_indices(ds: xr.Dataset) -> dict:
    """Find Y/X index ranges that cover the Tampen area."""
    lat2d = ds["lat"].values  # (Y, X)
    lon2d = ds["lon"].values

    mask = (
        (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
        (lon2d >= LON_MIN) & (lon2d <= LON_MAX)
    )

    if not mask.any():
        print("ERROR: No grid points found in Tampen bounding box!")
        print(f"  lat range in data: {np.nanmin(lat2d):.2f} - {np.nanmax(lat2d):.2f}")
        print(f"  lon range in data: {np.nanmin(lon2d):.2f} - {np.nanmax(lon2d):.2f}")
        sys.exit(1)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_start, y_end = np.where(rows)[0][[0, -1]]
    x_start, x_end = np.where(cols)[0][[0, -1]]

    # Add small buffer
    buf = 5
    y_start = max(0, y_start - buf)
    y_end = min(lat2d.shape[0] - 1, y_end + buf)
    x_start = max(0, x_start - buf)
    x_end = min(lat2d.shape[1] - 1, x_end + buf)

    print(f"  Tampen indices: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")
    print(f"  Grid points: {y_end - y_start} x {x_end - x_start}")

    return {
        "Y": slice(y_start, y_end + 1),
        "X": slice(x_start, x_end + 1),
    }


def pick_random_day(ds: xr.Dataset) -> str:
    """Pick a date roughly in the middle of the available time range."""
    times = ds["time"].values
    mid = len(times) // 2
    date = str(times[mid])[:10]
    print(f"  Selected date: {date}")
    return date


def main():
    print("=" * 60)
    print("Tampen Area — Surface Current Animation (NorKyst v3 800m)")
    print("=" * 60)

    url = NORKYST_800M_BE
    print(f"\nOpening: {url}")
    ds = xr.open_dataset(url)

    print(f"  Time range: {str(ds['time'].values[0])[:10]} to "
          f"{str(ds['time'].values[-1])[:10]}")

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

    # Fixed color scale from 98th percentile across all frames
    vmax = float(np.nanpercentile(speed, 98))
    vmax = max(vmax, 0.1)
    print(f"  Color scale: 0 - {vmax:.3f} m/s (98th percentile)")

    # Quiver subsampling
    skip = max(1, lat.shape[0] // 15)

    # ---- Build animation ----
    print(f"\nBuilding animation ({n_times} frames at {FPS} fps) ...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Initial frame
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

    t_str = str(times[0])[:16].replace("T", " ")
    title = ax.set_title(f"Tampen — Surface Currents\n{t_str} UTC",
                         fontsize=12, fontweight="bold")

    def update(frame: int):
        """Update pcolormesh and quiver for a single frame."""
        # pcolormesh: update the array on the underlying QuadMesh
        pcm.set_array(speed[frame].ravel())

        # quiver: update U and V components
        qv.set_UVC(u_sfc[frame, ::skip, ::skip],
                    v_sfc[frame, ::skip, ::skip])

        t_str = str(times[frame])[:16].replace("T", " ")
        title.set_text(f"Tampen — Surface Currents\n{t_str} UTC")

        print(f"  Frame {frame + 1:2d}/{n_times}: {t_str} UTC")
        return pcm, qv, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_times, blit=False, interval=1000 // FPS,
    )

    # Choose writer: prefer ffmpeg, fall back to pillow (GIF)
    if shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=FPS, bitrate=1800)
        out_path = OUT_DIR / "05_tampen_surface_animation.mp4"
    else:
        print("  WARNING: ffmpeg not found — falling back to GIF (PillowWriter)")
        writer = animation.PillowWriter(fps=FPS)
        out_path = OUT_DIR / "05_tampen_surface_animation.gif"

    print(f"\nRendering to: {out_path}")
    anim.save(str(out_path), writer=writer, dpi=DPI)
    plt.close(fig)

    print(f"\nAnimation saved: {out_path.resolve()}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Duration: {n_times / FPS:.1f} s ({n_times} frames at {FPS} fps)")


if __name__ == "__main__":
    main()
