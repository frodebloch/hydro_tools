"""
Download NorKyst v3 hindcast (free run) from MET Norway THREDDS.

This is the ROMS 800 m *hindcast* — a continuous free-running simulation
without data assimilation, as opposed to the forecast best-estimate product
served by fetch_norkyst_currents.py which stitches the first hours of
successive forecasts.  The hindcast is useful for long-term statistics,
model validation, and forcing offline particle-tracking or wave models.

Grid: curvilinear polar stereographic, 1148 x 2747 (Y x X), 2D lat/lon.
Depth: 25 z-levels (zdepth product), 0–2500 m.
Time:  hourly, 119 016 steps, 2012-01-05 to 2025-08-02.

Variables (zdepth product):
  - u_eastward, v_northward  — current components at depth (m/s)
  - temperature              — sea water potential temperature (deg C)
  - salinity                 — practical salinity (1)
  - zeta                     — sea surface height (m)
  - Uwind_eastward, Vwind_northward — 10 m wind components (m/s)
  - h                        — sea floor depth (m, static 2D)
  - sea_mask                 — land/sea mask (static 2D)
  - ln_AKs                   — log vertical salt diffusivity (m2/s)
  - projection_stere         — projection info

Usage examples:

  # Subset a week over the Lofoten area:
  python fetch_norkyst_hindcast.py -o ./data \\
      --time 2020-06-01 2020-06-07 --lat 67 70 --lon 12 16

  # Only temperature and salinity, upper 100 m:
  python fetch_norkyst_hindcast.py -o ./data \\
      --time 2019-01-01 2019-01-31 -v temperature salinity --depth 0 100

  # Sigma-depth product:
  python fetch_norkyst_hindcast.py -o ./data \\
      --product sdepth --time 2020-01-01 2020-01-02

  # Explore metadata without downloading:
  python fetch_norkyst_hindcast.py --info-only

Source: https://thredds.met.no/thredds/catalog/romshindcast/norkyst_v3/catalog.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from common import (
    NORKYST_HINDCAST_ZDEPTH,
    NORKYST_HINDCAST_SDEPTH,
    make_output_dir,
)

PRODUCT_URLS = {
    "zdepth": NORKYST_HINDCAST_ZDEPTH,
    "sdepth": NORKYST_HINDCAST_SDEPTH,
}


def fetch_hindcast(
    url: str,
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
    time_slice: tuple[str, str] | None = None,
    variables: list[str] | None = None,
    info_only: bool = False,
    label: str = "norkyst_hindcast",
) -> Path | None:
    """Fetch NorKyst v3 hindcast data.

    Parameters
    ----------
    url : str
        OPeNDAP URL for the aggregation.
    output_dir : Path
        Where to save output.
    lat_range, lon_range : tuple, optional
        Spatial bounds (decimal degrees).
    depth_range : tuple, optional
        Depth bounds in meters (positive downward, e.g. (0, 100)).
    time_slice : tuple, optional
        Time bounds as ISO strings, e.g. ("2020-01-01", "2020-01-31").
    variables : list of str, optional
        Variable names to keep. If None, keeps all.
    info_only : bool
        Just print metadata.
    label : str
        Filename label.
    """
    print(f"Opening: {url}")

    try:
        ds = xr.open_dataset(url)
    except OSError as e:
        print(f"ERROR: Could not open {url}\n  {e}", file=sys.stderr)
        return None

    if info_only:
        print("\n--- Dataset Info ---")
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

    # Subset variables
    if variables:
        available = [v for v in variables if v in ds.data_vars]
        if available:
            ds = ds[available]
        else:
            print(f"WARNING: None of {variables} found. Keeping all variables.")

    # Time subset — mandatory for the hindcast (119k+ hourly steps)
    time_dim = _find_dim(ds, ["time", "ocean_time"])
    if time_slice and time_dim:
        ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})
        n_steps = ds.sizes[time_dim]
        print(f"  Time subset: {time_slice[0]} to {time_slice[1]} ({n_steps} steps)")

    # Spatial subset — NorKyst uses curvilinear grids
    ds = _spatial_subset(ds, lat_range, lon_range)

    # Depth subset
    if depth_range:
        depth_dim = _find_dim(ds, ["depth", "z", "Zs"])
        if depth_dim:
            depths = ds[depth_dim].values
            mask = (depths >= min(depth_range)) & (depths <= max(depth_range))
            ds = ds.isel({depth_dim: mask})
            print(f"  Depth subset: {min(depth_range)}-{max(depth_range)} m "
                  f"({mask.sum()} levels)")

    out = output_dir / f"{label}.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print("Done.")
    return out


def _spatial_subset(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> xr.Dataset:
    """Handle spatial subsetting for both regular and curvilinear grids."""
    if lat_range is None and lon_range is None:
        return ds

    # Try regular grid first
    lat_name = _find_dim(ds, ["latitude", "lat"])
    lon_name = _find_dim(ds, ["longitude", "lon"])

    if lat_name and lon_name:
        if lat_range:
            ds = ds.sel({lat_name: slice(min(lat_range), max(lat_range))})
        if lon_range:
            ds = ds.sel({lon_name: slice(min(lon_range), max(lon_range))})
        return ds

    # Curvilinear grid: subset by index using 2D lat/lon arrays
    lat_var = _find_coord(ds, ["lat", "latitude", "lat_rho"])
    lon_var = _find_coord(ds, ["lon", "longitude", "lon_rho"])

    if lat_var and lon_var and ds[lat_var].ndim == 2:
        lat2d = ds[lat_var].values
        lon2d = ds[lon_var].values

        mask = np.ones_like(lat2d, dtype=bool)
        if lat_range:
            mask &= (lat2d >= min(lat_range)) & (lat2d <= max(lat_range))
        if lon_range:
            mask &= (lon2d >= min(lon_range)) & (lon2d <= max(lon_range))

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Get the dimension names for the 2D array
        dims = ds[lat_var].dims  # e.g. ('Y', 'X')
        ds = ds.isel({dims[0]: rows, dims[1]: cols})
        print(f"  Spatial subset: {rows.sum()} x {cols.sum()} grid points")

    return ds


def _find_dim(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.dims:
            return name
    return None


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download NorKyst v3 hindcast (free run) from MET Norway THREDDS"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--product", type=str, default="zdepth",
        choices=list(PRODUCT_URLS.keys()),
        help="Depth product: zdepth (z-levels, default) or sdepth (sigma-levels)",
    )
    parser.add_argument(
        "--lat", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Latitude bounds for subsetting",
    )
    parser.add_argument(
        "--lon", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Longitude bounds for subsetting",
    )
    parser.add_argument(
        "--depth", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Depth bounds in meters (e.g. 0 100)",
    )
    parser.add_argument(
        "--time", type=str, nargs=2, default=None, metavar=("START", "END"),
        help="Time bounds as ISO strings (e.g. 2020-01-01 2020-01-31). "
             "Required unless --info-only is set.",
    )
    parser.add_argument(
        "--variables", "-v", type=str, nargs="+", default=None,
        help="Variable names to download (default: all)",
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Just print dataset metadata, don't download",
    )

    args = parser.parse_args()

    if not args.info_only and args.time is None:
        parser.error(
            "--time START END is required for the hindcast dataset "
            "(119 000+ hourly steps, 2012–2025). Use --info-only to "
            "explore metadata without downloading."
        )

    output_dir = make_output_dir(args.output, "norkyst_hindcast")
    url = PRODUCT_URLS[args.product]
    label = f"norkyst_v3_hindcast_{args.product}"

    fetch_hindcast(
        url=url,
        output_dir=output_dir,
        lat_range=tuple(args.lat) if args.lat else None,
        lon_range=tuple(args.lon) if args.lon else None,
        depth_range=tuple(args.depth) if args.depth else None,
        time_slice=tuple(args.time) if args.time else None,
        variables=args.variables,
        info_only=args.info_only,
        label=label,
    )


if __name__ == "__main__":
    main()
