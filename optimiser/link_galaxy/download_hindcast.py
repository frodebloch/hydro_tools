"""Pre-download NORA3 wave hindcast for the Link Galaxy roundtrip.

Usage:
    python download_hindcast.py --year 2024

Downloads 12 monthly NetCDF files into
    hydro_tools/optimiser/data/nora3_link_galaxy/
covering the bounding box around every waypoint of the nominal
Sunndalsora - Karmoy - Fredrikstad - Kobenhavn - Swinoujscie -
Lysekil - Karmoy - Laerdal - Ardalstangen - Freifjorden roundtrip.

Baltic coverage
---------------
NORA3 covers the North Sea and Norwegian Sea well. The eastern
Baltic legs (Kobenhavn - Swinoujscie - Lysekil) sit near the edge of
NORA3's domain and Hs may be under-resolved by land shadowing.
If the annual-run flags gaps, the fallback is to blend in ECMWF ERA5
for those grid cells (TODO).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make the top-level `models` package importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.route import ROUTE_LINK_GALAXY_ROUNDTRIP
from models.weather import download_nora3_for_route

DEFAULT_DIR = (Path(__file__).parent.parent / "data" / "nora3_link_galaxy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--margin-deg", type=float, default=0.5)
    args = p.parse_args()

    print(f"Link Galaxy roundtrip: {len(ROUTE_LINK_GALAXY_ROUNDTRIP)} "
          f"waypoints")
    lats = [wp.lat for wp in ROUTE_LINK_GALAXY_ROUNDTRIP]
    lons = [wp.lon for wp in ROUTE_LINK_GALAXY_ROUNDTRIP]
    print(f"  lat span: {min(lats):.2f} .. {max(lats):.2f}")
    print(f"  lon span: {min(lons):.2f} .. {max(lons):.2f}")

    download_nora3_for_route(args.year, ROUTE_LINK_GALAXY_ROUNDTRIP,
                             args.data_dir, args.margin_deg)


if __name__ == "__main__":
    main()
