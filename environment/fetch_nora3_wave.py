"""
Download NORA3 wave hindcast data from MET Norway THREDDS.

The NORA3 wave hindcast is a WAM 3 km model covering the Norwegian Sea,
North Sea, and Barents Sea with hourly output from 1993 to present.
Data is stored as monthly files on a rotated-pole grid (rlat x rlon,
approximately 900 x 500 grid points) with 2D latitude/longitude arrays.

Variables:
  hs            — Total significant wave height [m]
  tp            — Total peak period [s]
  Pdir          — Peak direction [degree]
  thq           — Total mean wave direction [degree]
  tm1           — Total m1-period [s]
  tm2           — Total m2-period [s]
  tmp           — Total mean period [s]
  hs_sea        — Wind-sea significant wave height [m]
  hs_swell      — Swell significant wave height [m]
  tp_sea        — Wind-sea peak period [s]
  tp_swell      — Swell peak period [s]
  thq_sea       — Wind-sea mean direction [degree]
  thq_swell     — Swell mean direction [degree]
  ff            — Wind speed [m/s]
  dd            — Wind direction [degree]
  DC            — Drag coefficient [-]
  FV            — Friction velocity [m/s]
  fpI           — Interpolated peak frequency [s]
  model_depth   — Water depth [m] (2D static field)
  latitude      — 2D latitude array (dims: rlon, rlat)
  longitude     — 2D longitude array (dims: rlon, rlat)

Each monthly file contains ~744 hourly timesteps.

Usage examples:

  # Download a single month, full domain:
  python fetch_nora3_wave.py --start-month 202301 --end-month 202301

  # Download a full year with spatial subset:
  python fetch_nora3_wave.py --start-month 202301 --end-month 202312 \\
      --lat 59 62 --lon 3 7

  # Download selected variables only:
  python fetch_nora3_wave.py --start-month 202301 --end-month 202303 \\
      -v hs tp thq

  # Merge all months into a single file:
  python fetch_nora3_wave.py --start-month 202301 --end-month 202312 \\
      --lat 59 62 --lon 3 7 --merge

  # Just explore metadata from the first file:
  python fetch_nora3_wave.py --start-month 202301 --info-only

Source: https://thredds.met.no/thredds/projects/nora3.html
License: CC-BY 4.0 / NLOD — Credit "Data from MET Norway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from common import (
    NORA3_WAVE_BASE,
    NORA3_WAVE_PATTERN,
    make_output_dir,
    month_range,
    nora3_wave_url,
)


def fetch_month(
    yyyymm: str,
    output_dir: Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    variables: list[str] | None = None,
    info_only: bool = False,
) -> Path | None:
    """Fetch a single NORA3 wave monthly file.

    Parameters
    ----------
    yyyymm : str
        Month to fetch as YYYYMM (e.g. "202301").
    output_dir : Path
        Where to save output.
    lat_range, lon_range : tuple, optional
        Spatial bounds (decimal degrees). Subsetting is done via index
        masking on the 2D lat/lon arrays (rotated-pole grid).
    variables : list of str, optional
        Variable names to keep. If None, keeps all.
    info_only : bool
        Just print metadata.

    Returns
    -------
    Path or None
        Output file path, or None if info_only or on failure.
    """
    url = nora3_wave_url(yyyymm)
    print(f"Opening: {url}")

    try:
        ds = xr.open_dataset(url)
    except OSError as e:
        print(f"ERROR: Could not open {url}\n  {e}", file=sys.stderr)
        return None

    if info_only:
        print(f"\n--- NORA3 Wave {yyyymm} Dataset Info ---")
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

    # Spatial subset on the rotated-pole grid
    ds = _spatial_subset(ds, lat_range, lon_range)

    out = output_dir / f"nora3_wave_{yyyymm}.nc"
    print(f"Writing {out}  (shape: {dict(ds.sizes)})")
    ds.to_netcdf(out)
    ds.close()
    print(f"  Saved: {out}")
    return out


def _spatial_subset(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
) -> xr.Dataset:
    """Subset a rotated-pole grid using 2D latitude/longitude arrays.

    The NORA3 grid uses rlat/rlon dimensions with 2D latitude and
    longitude coordinate arrays (dims: rlon, rlat).  We build a mask
    in geographic space and then select the bounding rows/cols.
    """
    if lat_range is None and lon_range is None:
        return ds

    # Locate the 2D geographic coordinate arrays
    lat_var = _find_coord(ds, ["latitude", "lat"])
    lon_var = _find_coord(ds, ["longitude", "lon"])

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

        # Get the dimension names for the 2D array (e.g. ('rlon', 'rlat'))
        dims = ds[lat_var].dims
        ds = ds.isel({dims[0]: rows, dims[1]: cols})
        print(f"  Spatial subset: {rows.sum()} x {cols.sum()} grid points")
    else:
        # Fallback: 1D coordinates (unlikely for NORA3 but handle gracefully)
        if lat_range:
            ds = ds.sel({lat_var: slice(min(lat_range), max(lat_range))})
        if lon_range:
            ds = ds.sel({lon_var: slice(min(lon_range), max(lon_range))})

    return ds


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download NORA3 wave hindcast data from MET Norway THREDDS"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--start-month", type=str, required=True,
        help="First month to download as YYYYMM (e.g. 202301)",
    )
    parser.add_argument(
        "--end-month", type=str, default=None,
        help="Last month to download as YYYYMM (default: same as start-month)",
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
        "--variables", "-v", type=str, nargs="+", default=None,
        help="Variable names to download (default: all)",
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Just print dataset metadata from the first file, don't download",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all downloaded months into a single output file",
    )

    args = parser.parse_args()
    output_dir = make_output_dir(args.output, "nora3_wave")

    end_month = args.end_month or args.start_month
    months = month_range(args.start_month, end_month)
    lat_range = tuple(args.lat) if args.lat else None
    lon_range = tuple(args.lon) if args.lon else None

    if args.info_only:
        fetch_month(
            months[0], output_dir,
            lat_range=lat_range,
            lon_range=lon_range,
            variables=args.variables,
            info_only=True,
        )
        return

    print(f"Downloading {len(months)} month(s): {months[0]} to {months[-1]}")
    print(f"Output directory: {output_dir}")

    saved_files: list[Path] = []
    failed: list[str] = []

    for yyyymm in months:
        print(f"\n{'='*60}")
        print(f"Month: {yyyymm}")
        print(f"{'='*60}")
        try:
            path = fetch_month(
                yyyymm, output_dir,
                lat_range=lat_range,
                lon_range=lon_range,
                variables=args.variables,
            )
            if path is not None:
                saved_files.append(path)
            else:
                failed.append(yyyymm)
        except Exception as e:
            print(f"ERROR: Failed to process {yyyymm}: {e}", file=sys.stderr)
            failed.append(yyyymm)

    # Summary
    print(f"\n{'='*60}")
    print(f"Completed: {len(saved_files)}/{len(months)} months")
    if failed:
        print(f"Failed: {failed}")

    # Merge if requested
    if args.merge and len(saved_files) > 1:
        merged_path = output_dir / (
            f"nora3_wave_{months[0]}_{months[-1]}_merged.nc"
        )
        print(f"\nMerging {len(saved_files)} files into {merged_path} ...")
        ds_merged = xr.open_mfdataset(saved_files, combine="by_coords")
        ds_merged.to_netcdf(merged_path)
        ds_merged.close()
        print(f"  Saved merged file: {merged_path}")

        # Remove individual monthly files
        for f in saved_files:
            f.unlink()
            print(f"  Removed: {f}")

    print("Done.")


if __name__ == "__main__":
    main()
