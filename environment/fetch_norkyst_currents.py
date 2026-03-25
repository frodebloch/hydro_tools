"""
Download depth-resolved ocean currents from MET Norway NorKyst v3 (ROMS 800m).

Variables available:
  - u_eastward, v_northward  — current components at depth (m/s)
  - temperature              — sea water temperature (deg C)
  - salinity                 — practical salinity (PSU)
  - zeta                     — sea surface height (m)

The best-estimate aggregation provides a continuous timeseries built
from the first 6h of each forecast + the most recent full forecast.

Members:
  m00 — control member (800m ensemble)
  m61 — DA-nudged member (nudged toward NorKyst-DA 2.4km analysis)
  m70/m71 — two-way nested 800m + 160m (Sulafjord / Oslofjord)

Usage examples:

  # Latest best-estimate, subset area and depth:
  python fetch_norkyst_currents.py --output ./data \\
      --lat 59 62 --lon 3 7 --depth 0 100

  # DA-nudged member:
  python fetch_norkyst_currents.py --output ./data --member m61

  # Use Norshelf 2.4km DA product instead:
  python fetch_norkyst_currents.py --output ./data --norshelf

  # Just explore metadata:
  python fetch_norkyst_currents.py --info-only

Source: https://thredds.met.no/thredds/fou-hi/norkystv3.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from common import (
    NORKYST_800M_BE,
    NORKYST_800M_DA,
    NORSHELF_ZDEPTH_AGG,
    make_output_dir,
)

MEMBER_URLS = {
    "m00": NORKYST_800M_BE,
    "m61": NORKYST_800M_DA,
}


def fetch_norkyst(
    url: str,
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
    time_slice: tuple[str, str] | None = None,
    variables: list[str] | None = None,
    info_only: bool = False,
    label: str = "norkyst",
) -> Path | None:
    """Fetch NorKyst/Norshelf ocean data.

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
        Time bounds as ISO strings, e.g. ("2026-03-20", "2026-03-25").
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

    # Spatial subset — NorKyst uses curvilinear grids so we need to identify coords
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

    # Time subset
    if time_slice:
        time_dim = _find_dim(ds, ["time", "ocean_time"])
        if time_dim:
            ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})

    out = output_dir / f"{label}_currents.nc"
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
        dims = ds[lat_var].dims  # e.g. ('eta_rho', 'xi_rho') or ('Y', 'X')
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
        description="Download NorKyst v3 / Norshelf ocean currents from MET Norway"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--member", type=str, default="m00", choices=list(MEMBER_URLS.keys()),
        help="NorKyst member: m00 (control) or m61 (DA-nudged). Default: m00",
    )
    parser.add_argument(
        "--norshelf", action="store_true",
        help="Use Norshelf 2.4km DA product instead of NorKyst 800m",
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
        help="Time bounds as ISO strings (e.g. 2026-03-20 2026-03-25)",
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
    output_dir = make_output_dir(args.output, "norkyst")

    if args.norshelf:
        url = NORSHELF_ZDEPTH_AGG
        label = "norshelf_2.4km"
    else:
        url = MEMBER_URLS[args.member]
        label = f"norkyst_800m_{args.member}"

    fetch_norkyst(
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
