"""NORA3 weather data loading and route extraction."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .route import Waypoint


class WeatherAlongRoute:
    """Load pre-downloaded NORA3 hindcast and extract weather at route points.

    Expects monthly NetCDF files in data_dir with naming convention:
        nora3_wave_YYYYMM.nc

    Variables used: hs, tp, thq (wave direction), ff (wind speed), dd (wind dir).
    """

    def __init__(self, data_dir: Path):
        import xarray as xr

        self.data_dir = data_dir
        self._datasets = {}   # yyyymm -> xr.Dataset
        self._lat2d = None
        self._lon2d = None
        self._lat_dims = None

    def _ensure_month(self, yyyymm: str):
        """Lazy-load a monthly dataset."""
        if yyyymm in self._datasets:
            return

        import xarray as xr

        path = self.data_dir / f"nora3_wave_{yyyymm}.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"NORA3 data not found: {path}\n"
                f"Run: python voyage_comparison.py --download --year {yyyymm[:4]}")

        ds = xr.open_dataset(path)
        self._datasets[yyyymm] = ds

        # Cache lat/lon grid (same for all months)
        if self._lat2d is None:
            for name in ["latitude", "lat"]:
                if name in ds.coords or name in ds.data_vars:
                    self._lat2d = ds[name].values
                    self._lon2d = ds[{
                        "latitude": "longitude",
                        "lat": "lon",
                    }[name]].values
                    self._lat_dims = ds[name].dims
                    break
            if self._lat2d is None:
                raise ValueError("Cannot find latitude/longitude in NORA3 data")

    def _nearest_idx(self, lat: float, lon: float) -> tuple:
        """Find nearest grid point indices."""
        dist2 = (self._lat2d - lat) ** 2 + (self._lon2d - lon) ** 2
        return np.unravel_index(np.argmin(dist2), dist2.shape)

    def get_weather(self, lat: float, lon: float,
                    dt: datetime) -> dict:
        """Extract weather at (lat, lon, time).

        Returns dict with keys: hs, tp, wave_dir, wind_speed, wind_dir.
        Missing values default to 0.
        """
        yyyymm = dt.strftime("%Y%m")
        self._ensure_month(yyyymm)
        ds = self._datasets[yyyymm]

        idx = self._nearest_idx(lat, lon)
        sel = {self._lat_dims[0]: idx[0], self._lat_dims[1]: idx[1]}

        # Select nearest time
        time_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            point = ds.sel(time=time_str, method="nearest").isel(**sel)
        except Exception:
            # Fallback: find nearest time manually
            times = ds["time"].values
            target = np.datetime64(dt.replace(tzinfo=None))
            ti = int(np.argmin(np.abs(times - target)))
            point = ds.isel(time=ti, **sel)

        def _val(name):
            if name in point:
                v = float(point[name].values)
                return v if np.isfinite(v) else 0.0
            return 0.0

        return {
            "hs": _val("hs"),
            "tp": _val("tp"),
            "wave_dir": _val("thq"),    # mean wave direction [deg from]
            "wind_speed": _val("ff"),    # 10m wind speed [m/s]
            "wind_dir": _val("dd"),      # 10m wind direction [deg from]
        }

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()


def download_nora3_for_route(year: int, route_waypoints: list[Waypoint],
                             output_dir: Path, margin_deg: float = 0.5):
    """Pre-download NORA3 wave data for a bounding box around the route.

    Downloads 12 monthly files using the existing fetch infrastructure.
    """
    # Add environment directory to path for imports
    env_dir = Path(__file__).parent.parent.parent / "environment"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))

    from fetch_nora3_wave import fetch_month

    # Compute bounding box
    lats = [wp.lat for wp in route_waypoints]
    lons = [wp.lon for wp in route_waypoints]
    lat_range = (min(lats) - margin_deg, max(lats) + margin_deg)
    lon_range = (min(lons) - margin_deg, max(lons) + margin_deg)

    output_dir.mkdir(parents=True, exist_ok=True)

    variables = ["hs", "tp", "thq", "ff", "dd", "latitude", "longitude"]

    print(f"\nDownloading NORA3 wave data for {year}")
    print(f"  Bounding box: lat [{lat_range[0]:.1f}, {lat_range[1]:.1f}], "
          f"lon [{lon_range[0]:.1f}, {lon_range[1]:.1f}]")
    print(f"  Variables: {variables}")
    print(f"  Output: {output_dir}")

    for month in range(1, 13):
        yyyymm = f"{year}{month:02d}"
        out_file = output_dir / f"nora3_wave_{yyyymm}.nc"
        if out_file.exists():
            print(f"\n  {yyyymm}: already exists, skipping")
            continue
        print(f"\n  Fetching {yyyymm} ...")
        fetch_month(yyyymm, output_dir, lat_range=lat_range,
                    lon_range=lon_range, variables=variables)

    print("\nDownload complete.")
