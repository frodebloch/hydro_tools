"""
Download Topaz5 pan-Arctic ocean data from MET Norway THREDDS.

The Topaz5 product provides hourly 3D ocean fields for the entire Arctic:
  - Ocean currents (u, v) at 40 depth levels
  - Temperature, salinity
  - Sea surface height
  - Sea ice concentration, thickness, velocity

Note: The THREDDS hourly product contains only the latest forecast
(not a continuous archive). For analysis + forecast archive, use
the Copernicus Marine Service product ARCTIC_ANALYSISFORECAST_PHY_002_001.

Usage examples:

  # Explore available variables:
  python fetch_topaz5.py --info-only

  # Download with spatial and depth subset:
  python fetch_topaz5.py --output ./data \\
      --lat 65 80 --lon -10 40 --depth 0 200

Source: https://thredds.met.no/thredds/fou-hi/topaz5-arc-1hr.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from common import TOPAZ5_HOURLY_BE, make_output_dir


def fetch_topaz5(
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
    variables: list[str] | None = None,
    info_only: bool = False,
) -> Path | None:
    """Fetch Topaz5 hourly forecast data."""
    url = TOPAZ5_HOURLY_BE
    print(f"Opening: {url}")

    try:
        ds = xr.open_dataset(url)
    except OSError as e:
        print(f"ERROR: Could not open {url}\n  {e}", file=sys.stderr)
        return None

    if info_only:
        print("\n--- Topaz5 Dataset Info ---")
        print(ds)
        print("\n--- Dimensions ---")
        for name, size in ds.sizes.items():
            print(f"  {name}: {size}")
        print("\n--- Coordinates ---")
        for name, coord in ds.coords.items():
            vals = coord.values
            if np.issubdtype(vals.dtype, np.number) and vals.size > 0:
                print(f"  {name}: min={np.nanmin(vals):.4f}, "
                      f"max={np.nanmax(vals):.4f}, n={vals.size}")
            else:
                print(f"  {name}: shape={coord.shape}, dtype={coord.dtype}")
        print("\n--- Data Variables ---")
        for name, var in ds.data_vars.items():
            print(f"  {name}: dims={var.dims}, shape={var.shape}")
        ds.close()
        return None

    if variables:
        available = [v for v in variables if v in ds.data_vars]
        if available:
            ds = ds[available]

    # Topaz uses a polar stereographic grid with 2D lat/lon
    # Subsetting must be done via index masking
    ds = _subset_polar_stereo(ds, lat_range, lon_range)

    if depth_range:
        for depth_name in ["depth", "z"]:
            if depth_name in ds.dims:
                depths = ds[depth_name].values
                mask = (depths >= min(depth_range)) & (depths <= max(depth_range))
                ds = ds.isel({depth_name: mask})
                print(f"  Depth subset: {min(depth_range)}-{max(depth_range)} m "
                      f"({mask.sum()} levels)")
                break

    out = output_dir / "topaz5_arctic.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print("Done.")
    return out


def _subset_polar_stereo(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> xr.Dataset:
    """Subset a polar stereographic dataset using 2D lat/lon fields."""
    if lat_range is None and lon_range is None:
        return ds

    lat_var = None
    lon_var = None
    for name in ["latitude", "lat"]:
        if name in ds.coords:
            lat_var = name
            break
    for name in ["longitude", "lon"]:
        if name in ds.coords:
            lon_var = name
            break

    if lat_var is None or lon_var is None:
        print("  WARNING: Could not find lat/lon coords for subsetting")
        return ds

    lat2d = ds[lat_var].values
    lon2d = ds[lon_var].values

    if lat2d.ndim == 2:
        mask = np.ones_like(lat2d, dtype=bool)
        if lat_range:
            mask &= (lat2d >= min(lat_range)) & (lat2d <= max(lat_range))
        if lon_range:
            mask &= (lon2d >= min(lon_range)) & (lon2d <= max(lon_range))

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        dims = ds[lat_var].dims
        ds = ds.isel({dims[0]: rows, dims[1]: cols})
        print(f"  Spatial subset: {rows.sum()} x {cols.sum()} grid points")
    else:
        # 1D coordinates
        if lat_range:
            ds = ds.sel({lat_var: slice(min(lat_range), max(lat_range))})
        if lon_range:
            ds = ds.sel({lon_var: slice(min(lon_range), max(lon_range))})

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Download Topaz5 pan-Arctic ocean data from MET Norway"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--lat", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Latitude bounds",
    )
    parser.add_argument(
        "--lon", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Longitude bounds",
    )
    parser.add_argument(
        "--depth", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Depth bounds in meters",
    )
    parser.add_argument(
        "--variables", "-v", type=str, nargs="+", default=None,
        help="Variable names to download (default: all)",
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Just print dataset metadata",
    )

    args = parser.parse_args()
    output_dir = make_output_dir(args.output, "topaz5")

    fetch_topaz5(
        output_dir=output_dir,
        lat_range=tuple(args.lat) if args.lat else None,
        lon_range=tuple(args.lon) if args.lon else None,
        depth_range=tuple(args.depth) if args.depth else None,
        variables=args.variables,
        info_only=args.info_only,
    )


if __name__ == "__main__":
    main()
