"""
Download WW3 4 km 2D directional-frequency wave spectra from MET Norway THREDDS.

These are point spectra (not gridded) at pre-selected locations.
Each file contains the full 2D spectrum: 36 frequencies x 36 directions.

Available domains:
  poi        — Points of Interest (selected locations in the model domain)
  finnmark   — C0 boundary spectra for WAM800 Finnmark domain
  nordnorge  — C1 boundary spectra for WAM800 Nord-Norge domain
  midtnorge  — C2 boundary spectra for WAM800 Midt-Norge domain
  vestlandet — C3 boundary spectra for WAM800 Vestlandet domain
  skagerrak  — C4 boundary spectra for WAM800 Skagerrak domain
  c5         — C5 additional boundary domain

Usage examples:

  # Latest POI spectra:
  python fetch_ww3_spectra.py --output ./data --domain poi

  # Specific run, select station(s) by index:
  python fetch_ww3_spectra.py --output ./data --domain poi \\
      --date 20260320 --cycle 12 --stations 0 5 10

  # All coastal boundary spectra for latest run:
  python fetch_ww3_spectra.py --output ./data --domain finnmark nordnorge \\
      midtnorge vestlandet skagerrak

  # Just explore what's in the file (print metadata, don't save):
  python fetch_ww3_spectra.py --domain poi --info-only

Source: https://thredds.met.no/thredds/fou-hi/ww3_4km.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from common import (
    WW3_SPC_DOMAINS,
    latest_cycle,
    make_output_dir,
    ww3_spc_url,
)


def fetch_spectra(
    domain: str,
    date: str,
    cycle: str,
    output_dir: Path,
    stations: list[int] | None = None,
    info_only: bool = False,
) -> Path | None:
    """Fetch 2D spectral data for a given domain and forecast run.

    Parameters
    ----------
    domain : str
        One of the keys in WW3_SPC_DOMAINS.
    date : str
        YYYYMMDD.
    cycle : str
        '00', '06', '12', or '18'.
    output_dir : Path
        Where to save.
    stations : list of int, optional
        Station indices to subset. If None, downloads all stations
        (can be large — 700 MB+ for POI).
    info_only : bool
        If True, just print dataset info and return.

    Returns
    -------
    Path or None
        Output file path, or None if info_only.
    """
    url = ww3_spc_url(domain, date, cycle)
    print(f"Opening: {url}")

    try:
        ds = xr.open_dataset(url)
    except OSError as e:
        print(f"ERROR: Could not open {url}\n  {e}", file=sys.stderr)
        return None

    if info_only:
        print("\n--- Dataset Info ---")
        print(ds)
        print("\n--- Coordinates ---")
        for name, coord in ds.coords.items():
            print(f"  {name}: shape={coord.shape}, dtype={coord.dtype}")
        print("\n--- Data Variables ---")
        for name, var in ds.data_vars.items():
            print(f"  {name}: dims={var.dims}, shape={var.shape}")
        ds.close()
        return None

    # Subset by station if requested
    if stations is not None:
        # Station dimension is typically 'station' or 'nstation'
        station_dim = _find_dim(ds, ["station", "nstation", "string20"])
        if station_dim:
            ds = ds.isel({station_dim: stations})
            print(f"  Selected stations: {stations}")

    out = output_dir / f"ww3_spc_{domain}_{date}T{cycle}Z.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print(f"  Saved: {out}")
    return out


def _find_dim(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.dims:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download WW3 2D point spectra from MET Norway"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--domain", "-d", type=str, nargs="+", default=["poi"],
        choices=list(WW3_SPC_DOMAINS.keys()),
        help="Spectral domain(s) to download (default: poi)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Forecast date YYYYMMDD (default: latest)",
    )
    parser.add_argument(
        "--cycle", type=str, default=None, choices=["00", "06", "12", "18"],
        help="Forecast cycle (default: latest)",
    )
    parser.add_argument(
        "--stations", type=int, nargs="+", default=None,
        help="Station indices to subset (default: all — WARNING: can be large)",
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Just print dataset metadata, don't download",
    )

    args = parser.parse_args()
    output_dir = make_output_dir(args.output, "ww3_spectra")

    date = args.date
    cycle = args.cycle
    if date is None or cycle is None:
        d, c = latest_cycle()
        date = date or d
        cycle = cycle or c

    for domain in args.domain:
        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({WW3_SPC_DOMAINS[domain]})")
        print(f"{'='*60}")
        fetch_spectra(
            domain, date, cycle, output_dir,
            stations=args.stations,
            info_only=args.info_only,
        )


if __name__ == "__main__":
    main()
