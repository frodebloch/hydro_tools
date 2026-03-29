"""Route map plotting with Natural Earth / GSHHG coastlines."""

from pathlib import Path

import numpy as np

from models.route import Waypoint, haversine_nm


def _load_land_polygons(bbox: tuple[float, float, float, float],
                        ) -> list[list[tuple[float, float]]]:
    """Load land polygons that intersect a bounding box.

    Searches for data sources in order of preference:
    1. Natural Earth 10m land (best detail for regional maps)
    2. Natural Earth 50m land (good compromise)
    3. GSHHG crude L1 (cartopy bundled fallback)

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    list of polygons, each a list of (lon, lat) tuples.
    """
    import shapefile

    # Data directory is at optimiser/data/, two levels up from plotting/
    data_dir = Path(__file__).parent.parent / "data"
    candidates = [
        data_dir / "ne_10m" / "ne_10m_land.shp",
        data_dir / "ne_50m" / "ne_50m_land.shp",
        Path("/usr/share/cartopy/data/shapefiles/gshhs/c/GSHHS_c_L1.shp"),
        data_dir / "gshhg" / "GSHHS_c_L1.shp",
    ]

    shp_path = None
    for c in candidates:
        if c.exists():
            shp_path = c
            break

    if shp_path is None:
        print("Warning: no coastline data found.")
        print(f"  Searched: {[str(c) for c in candidates]}")
        return []

    print(f"  Coastline data: {shp_path.name}")

    sf = shapefile.Reader(str(shp_path))
    lon_min, lat_min, lon_max, lat_max = bbox
    margin = 2.0
    polygons = []

    for shape in sf.shapes():
        sb = shape.bbox  # [min_lon, min_lat, max_lon, max_lat]
        if (sb[2] < lon_min - margin or sb[0] > lon_max + margin or
                sb[3] < lat_min - margin or sb[1] > lat_max + margin):
            continue

        parts = list(shape.parts) + [len(shape.points)]
        for k in range(len(parts) - 1):
            ring = shape.points[parts[k]:parts[k + 1]]
            # Quick filter: skip rings entirely outside the bbox
            ring_lons = [p[0] for p in ring]
            ring_lats = [p[1] for p in ring]
            if (max(ring_lons) < lon_min - margin or
                    min(ring_lons) > lon_max + margin or
                    max(ring_lats) < lat_min - margin or
                    min(ring_lats) > lat_max + margin):
                continue
            polygons.append([(p[0], p[1]) for p in ring])

    return polygons


def plot_route(waypoints: list[Waypoint]):
    """Plot the route on a map with GSHHG coastlines.

    Uses the Global Self-consistent Hierarchical High-resolution Shoreline
    Database (GSHHG) for accurate coastlines.  Falls back to a plain frame
    if the shapefiles are not available.

    The map extent is derived automatically from the waypoints, so this
    function works for any route.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        return

    # --- Compute map extent from waypoints (with margin) ---
    wlats = [wp.lat for wp in waypoints]
    wlons = [wp.lon for wp in waypoints]
    lat_span = max(wlats) - min(wlats)
    lon_span = max(wlons) - min(wlons)
    margin_lat = max(1.5, lat_span * 0.30)
    margin_lon = max(2.0, lon_span * 0.30)
    lon_min = min(wlons) - margin_lon
    lon_max = max(wlons) + margin_lon
    lat_min = min(wlats) - margin_lat
    lat_max = max(wlats) + margin_lat

    bbox = (lon_min, lat_min, lon_max, lat_max)

    # --- Load coastlines ---
    polygons = _load_land_polygons(bbox)

    # --- Set up figure ---
    mid_lat = np.mean(wlats)
    aspect = 1.0 / np.cos(np.radians(mid_lat))
    fig_width = 10
    fig_height = fig_width * (lat_max - lat_min) / (lon_max - lon_min) * aspect
    fig_height = max(6, min(14, fig_height))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Sea background
    ax.set_facecolor("#c8dce8")

    # --- Draw land polygons ---
    if polygons:
        patches = []
        for poly_pts in polygons:
            if len(poly_pts) >= 3:
                patches.append(MplPolygon(poly_pts, closed=True))
        if patches:
            pc = PatchCollection(patches,
                                 facecolor="#e8e4d8",
                                 edgecolor="#8a8578",
                                 linewidth=0.6,
                                 zorder=2)
            ax.add_collection(pc)
    else:
        # Fallback: just frame with no land
        ax.text(0.5, 0.5, "(GSHHG coastline data not available)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.5")

    # --- Draw route ---
    ax.plot(wlons, wlats, "o-", color="#c0392b", linewidth=2.5,
            markersize=6, markeredgecolor="white", markeredgewidth=0.8,
            zorder=5, label="Route")

    # Label departure and arrival
    ax.annotate(waypoints[0].name, (wlons[0], wlats[0]),
                textcoords="offset points", xytext=(-10, -15),
                fontsize=9, fontweight="bold", color="#c0392b",
                ha="right", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))
    ax.annotate(waypoints[-1].name, (wlons[-1], wlats[-1]),
                textcoords="offset points", xytext=(-10, 10),
                fontsize=9, fontweight="bold", color="#c0392b",
                ha="right", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))

    # --- Map formatting ---
    ax.set_aspect(aspect)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg E]")
    ax.set_ylabel("Latitude [deg N]")
    ax.set_title(f"{waypoints[0].name} \u2014 {waypoints[-1].name}")
    ax.grid(True, alpha=0.25, linestyle="--", color="0.5")

    # Distance annotation
    total_nm = sum(
        haversine_nm(waypoints[i].lat, waypoints[i].lon,
                     waypoints[i + 1].lat, waypoints[i + 1].lon)
        for i in range(len(waypoints) - 1)
    )
    ax.text(0.02, 0.02, f"Total distance: {total_nm:.0f} nm",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    # Output to the optimiser/ directory (two levels up from plotting/)
    out_path = Path(__file__).parent.parent / "voyage_comparison_route.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Route map saved: {out_path}")
