"""
Download hourly surface currents from Copernicus Marine (CMEMS)
North-West European Shelf multi-year reanalysis.

Dataset: cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i
         (product NWSHELF_MULTIYEAR_PHY_004_009)
  - uo, vo : eastward / northward surface current [m/s], tides included
  - grid   : regular lat/lon, ~7 km
  - time   : hourly, 1993 -> present (reanalysis + interim)

Covers the whole Rotterdam -> Gothenburg route, including the southern
North Sea, where NorKyst v3 has no coverage.

Requires a (free) Copernicus Marine account.  Log in once with

    copernicusmarine login

which stores the credentials in ~/.copernicusmarine.

Usage:

    # Full year 2024, default corridor for the Rotterdam-Gothenburg route,
    # one NetCDF file per month:
    python fetch_cmems_nws_currents.py -o ./data/cmems_nws --year 2024

    # Custom area / period:
    python fetch_cmems_nws_currents.py -o ./data/cmems_nws \\
        --time 2024-01-01 2024-01-31 --lat 51.5 58.5 --lon 3.0 12.0
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

DATASET_ID = "cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i"

# Corridor around the Rotterdam -> Gothenburg route (models/route.py),
# with margin for interpolation.
DEFAULT_LAT = (51.5, 58.5)
DEFAULT_LON = (3.0, 12.2)


def fetch(out_dir: Path, start: str, end: str, lat: tuple[float, float], lon: tuple[float, float],
          filename: str) -> Path:
    """Download one subset to out_dir/filename (skipped if it exists)."""
    import copernicusmarine

    target = out_dir / filename
    if target.exists():
        print(f"  exists, skipping: {target}")
        return target
    print(f"  downloading {start} .. {end} -> {target}")
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=lon[0],
        maximum_longitude=lon[1],
        minimum_latitude=lat[0],
        maximum_latitude=lat[1],
        start_datetime=f"{start}T00:00:00",
        end_datetime=f"{end}T23:00:00",
        output_directory=str(out_dir),
        output_filename=filename,
    )
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out-dir", type=Path, default=Path("data/cmems_nws"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", type=int, help="Download a full year, one file per month.")
    g.add_argument("--time", nargs=2, metavar=("START", "END"), help="Date range YYYY-MM-DD YYYY-MM-DD.")
    p.add_argument("--lat", nargs=2, type=float, default=DEFAULT_LAT)
    p.add_argument("--lon", nargs=2, type=float, default=DEFAULT_LON)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = tuple(args.lat), tuple(args.lon)
    try:
        if args.year:
            for m in range(1, 13):
                last = calendar.monthrange(args.year, m)[1]
                fetch(args.out_dir, f"{args.year}-{m:02d}-01", f"{args.year}-{m:02d}-{last:02d}", lat, lon,
                      f"cmems_nws_uv_{args.year}{m:02d}.nc")
        else:
            s, e = args.time
            fetch(args.out_dir, s, e, lat, lon, f"cmems_nws_uv_{s.replace('-', '')}_{e.replace('-', '')}.nc")
    except Exception as exc:  # credentials, network
        print(f"ERROR: {exc}\nHave you run 'copernicusmarine login'?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
