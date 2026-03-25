"""
Download WW3 4 km bulk (gridded) wave fields from MET Norway THREDDS.

Products available in the gridded files:
  - Significant wave height (Hs)
  - Peak period (Tp)
  - Mean wave direction
  - Wind sea / swell partitions
  - Stokes drift components
  - Wind speed components (forcing)

Usage examples:

  # Latest forecast, full domain:
  python fetch_ww3_bulk.py --output ./data

  # Latest forecast, subset by bounding box:
  python fetch_ww3_bulk.py --output ./data --lat 58 63 --lon 3 8

  # Specific date/cycle:
  python fetch_ww3_bulk.py --output ./data --date 20260320 --cycle 12

  # Date range (first 6h of each cycle stitched as "best estimate"):
  python fetch_ww3_bulk.py --output ./data --use-aggregation

Source: https://thredds.met.no/thredds/fou-hi/ww3_4km.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from common import (
    WW3_AGG,
    latest_cycle,
    make_output_dir,
    ww3_bulk_url,
)


def fetch_aggregation(
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> Path:
    """Fetch data from the best-estimate aggregation (continuous timeseries)."""
    print(f"Opening WW3 aggregation: {WW3_AGG}")
    ds = xr.open_dataset(WW3_AGG)
    ds = _subset(ds, lat_range, lon_range)
    out = output_dir / "ww3_bulk_aggregation.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print("Done.")
    return out


def fetch_single_run(
    date: str,
    cycle: str,
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> Path:
    """Fetch a single forecast run."""
    url = ww3_bulk_url(date, cycle)
    print(f"Opening: {url}")
    ds = xr.open_dataset(url)
    ds = _subset(ds, lat_range, lon_range)
    out = output_dir / f"ww3_bulk_{date}T{cycle}Z.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print("Done.")
    return out


def _subset(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> xr.Dataset:
    """Spatially subset the dataset if bounds are given."""
    # Identify coordinate names (WW3 uses 'latitude'/'longitude' or 'rlat'/'rlon')
    lat_name = _find_coord(ds, ["latitude", "lat", "rlat"])
    lon_name = _find_coord(ds, ["longitude", "lon", "rlon"])

    if lat_range and lat_name:
        ds = ds.sel({lat_name: slice(min(lat_range), max(lat_range))})
    if lon_range and lon_name:
        ds = ds.sel({lon_name: slice(min(lon_range), max(lon_range))})
    return ds


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download WW3 4km bulk wave fields from MET Norway"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Forecast date YYYYMMDD (default: latest available)",
    )
    parser.add_argument(
        "--cycle", type=str, default=None, choices=["00", "06", "12", "18"],
        help="Forecast cycle UTC hour (default: latest available)",
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
        "--use-aggregation", action="store_true",
        help="Use the best-estimate aggregation instead of a single run",
    )

    args = parser.parse_args()
    output_dir = make_output_dir(args.output, "ww3_bulk")

    lat_range = tuple(args.lat) if args.lat else None
    lon_range = tuple(args.lon) if args.lon else None

    if args.use_aggregation:
        fetch_aggregation(output_dir, lat_range, lon_range)
    else:
        date = args.date
        cycle = args.cycle
        if date is None or cycle is None:
            d, c = latest_cycle()
            date = date or d
            cycle = cycle or c
        fetch_single_run(date, cycle, output_dir, lat_range, lon_range)


if __name__ == "__main__":
    main()
