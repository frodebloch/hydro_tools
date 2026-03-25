"""
Download WAM 800m coastal wave data from MET Norway THREDDS.

The WAM 800m model covers the Norwegian coast in 5 domains:
  finnmark, nordnorge, midtnorge, vestlandet, skagerrak

This model includes wave-current interaction (forced by NorKyst v3 surface
currents) and receives boundary spectra from the regional WW3 4km model.

Output includes bulk wave parameters at 800m resolution:
  - Significant wave height (total, wind sea, swell)
  - Peak period, mean period
  - Wave direction
  - Stokes drift
  - etc.

Usage examples:

  # Latest data for Vestlandet:
  python fetch_wam800.py --output ./data --domain vestlandet

  # Multiple domains:
  python fetch_wam800.py --output ./data --domain finnmark nordnorge

  # All domains, with spatial subset:
  python fetch_wam800.py --output ./data --domain all \\
      --lat 59 62 --lon 3 7

  # Just explore:
  python fetch_wam800.py --domain vestlandet --info-only

Source: https://thredds.met.no/thredds/fou-hi/mywavewam800current.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from common import WAM800_HOURLY_AGGS, make_output_dir


ALL_DOMAINS = list(WAM800_HOURLY_AGGS.keys())


def fetch_wam800(
    domain: str,
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    info_only: bool = False,
) -> Path | None:
    """Fetch WAM 800m hourly aggregation for a coastal domain."""
    url = WAM800_HOURLY_AGGS[domain]
    print(f"Opening: {url}")

    try:
        ds = xr.open_dataset(url)
    except OSError as e:
        print(f"ERROR: Could not open {url}\n  {e}", file=sys.stderr)
        return None

    if info_only:
        print(f"\n--- {domain.upper()} Dataset Info ---")
        print(ds)
        print("\n--- Data Variables ---")
        for name, var in ds.data_vars.items():
            print(f"  {name}: dims={var.dims}, shape={var.shape}")
        ds.close()
        return None

    ds = _subset(ds, lat_range, lon_range)

    out = output_dir / f"wam800_{domain}.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print(f"  Saved: {out}")
    return out


def _subset(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> xr.Dataset:
    if lat_range is None and lon_range is None:
        return ds

    for lat_name in ["latitude", "lat", "rlat"]:
        if lat_name in ds.coords or lat_name in ds.dims:
            if lat_range:
                ds = ds.sel({lat_name: slice(min(lat_range), max(lat_range))})
            break

    for lon_name in ["longitude", "lon", "rlon"]:
        if lon_name in ds.coords or lon_name in ds.dims:
            if lon_range:
                ds = ds.sel({lon_name: slice(min(lon_range), max(lon_range))})
            break

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Download WAM 800m coastal wave data from MET Norway"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--domain", "-d", type=str, nargs="+", default=["vestlandet"],
        help="Domain(s) to download. Use 'all' for all domains. "
             f"Choices: {ALL_DOMAINS + ['all']}",
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
        "--info-only", action="store_true",
        help="Just print dataset metadata, don't download",
    )

    args = parser.parse_args()
    output_dir = make_output_dir(args.output, "wam800")

    domains = ALL_DOMAINS if "all" in args.domain else args.domain

    for domain in domains:
        if domain not in WAM800_HOURLY_AGGS:
            print(f"WARNING: Unknown domain '{domain}', skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        fetch_wam800(
            domain, output_dir,
            lat_range=tuple(args.lat) if args.lat else None,
            lon_range=tuple(args.lon) if args.lon else None,
            info_only=args.info_only,
        )


if __name__ == "__main__":
    main()
