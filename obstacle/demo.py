"""
Demo: Visibility graph path planning through a wind farm.

Places 100 wind turbines in a variable grid (~800m apart) with 200m safety zones,
then plans several paths that require turns to avoid them.

Outputs a matplotlib figure showing the wind farm, safety zones, and planned paths.
"""

import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from navigation import latlon_to_north_east, north_east_to_latlon
from path_planner import Circle, find_path, path_total_length, PathSegment, segments_to_waypoints


# --- Configuration ---

# Wind farm origin (somewhere in the North Sea)
ORIGIN_LAT = 56.0
ORIGIN_LON = 3.0

# Turbine grid parameters
GRID_SPACING = 800.0        # meters, nominal spacing
GRID_JITTER = 150.0         # meters, random offset to make it irregular
SAFETY_RADIUS = 200.0       # meters, exclusion zone around each turbine
SUBSTATION_SAFETY_RADIUS = 500.0  # meters, exclusion zone around substations
PLANNING_MARGIN = 20.0      # meters, extra clearance outside safety zone
N_ROWS = 10
N_COLS = 10

# Corridor pre-screening width -- must accommodate the largest obstacle
CORRIDOR_WIDTH = 2.0 * (SUBSTATION_SAFETY_RADIUS + PLANNING_MARGIN) + 300.0

# Random seed for reproducibility
SEED = 42


def create_wind_farm(substations: list[dict]) -> list[dict]:
    """
    Create wind turbines in a variable grid pattern.
    Turbines that fall inside a substation safety zone are excluded.
    Returns list of dicts with 'lat', 'lon', 'north', 'east' keys.
    """
    random.seed(SEED)
    turbines = []

    for row in range(N_ROWS):
        for col in range(N_COLS):
            # Nominal position
            north = row * GRID_SPACING
            east = col * GRID_SPACING

            # Add jitter
            north += random.uniform(-GRID_JITTER, GRID_JITTER)
            east += random.uniform(-GRID_JITTER, GRID_JITTER)

            # Skip if inside any substation safety zone
            inside_substation = False
            for s in substations:
                dist = math.hypot(east - s["east"], north - s["north"])
                if dist < SUBSTATION_SAFETY_RADIUS:
                    inside_substation = True
                    break
            if inside_substation:
                continue

            lat, lon = north_east_to_latlon(north, east, ORIGIN_LAT, ORIGIN_LON)
            turbines.append({
                "lat": lat,
                "lon": lon,
                "north": north,
                "east": east,
            })

    return turbines


def create_substations() -> list[dict]:
    """
    Place a few substations within the wind farm.
    These have larger safety zones (500m) than turbines.
    """
    # Place 3 substations at strategic positions within the farm grid
    positions = [
        (2400, 2000),   # south-central area
        (4800, 5600),   # east-central area
        (6400, 1600),   # north-west area
    ]
    substations = []
    for north, east in positions:
        lat, lon = north_east_to_latlon(north, east, ORIGIN_LAT, ORIGIN_LON)
        substations.append({
            "lat": lat,
            "lon": lon,
            "north": north,
            "east": east,
        })
    return substations


def build_obstacles(turbines: list[dict], substations: list[dict]) -> list[Circle]:
    """Convert turbines and substations to Circle obstacles with planning margin."""
    obstacles = [
        Circle(x=t["east"], y=t["north"], r=SAFETY_RADIUS + PLANNING_MARGIN, label=f"T{i}")
        for i, t in enumerate(turbines)
    ]
    for i, s in enumerate(substations):
        obstacles.append(Circle(
            x=s["east"], y=s["north"],
            r=SUBSTATION_SAFETY_RADIUS + PLANNING_MARGIN,
            label=f"S{i}",
        ))
    return obstacles


def plan_path_latlon(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    obstacles: list[Circle],
) -> tuple[list[PathSegment] | None, tuple[float, float], tuple[float, float]]:
    """
    Plan a path between two lat/lon positions.
    Returns (segments, start_ne, end_ne).
    """
    sn, se = latlon_to_north_east(ORIGIN_LAT, ORIGIN_LON, start_lat, start_lon)
    en, ee = latlon_to_north_east(ORIGIN_LAT, ORIGIN_LON, end_lat, end_lon)

    segments = find_path(
        start=(se, sn),
        end=(ee, en),
        obstacles=obstacles,
        corridor_width=CORRIDOR_WIDTH,
    )
    return segments, (se, sn), (ee, en)


def plot_arc(ax, seg: PathSegment, color: str, lw: float = 2.0):
    """Plot an arc segment."""
    if seg.cw:
        # Clockwise: from angle_start going clockwise to angle_end
        a_start_deg = math.degrees(seg.angle_end)
        a_end_deg = math.degrees(seg.angle_start)
    else:
        a_start_deg = math.degrees(seg.angle_start)
        a_end_deg = math.degrees(seg.angle_end)
        if a_end_deg < a_start_deg:
            a_end_deg += 360.0

    arc = mpatches.Arc(
        (seg.cx, seg.cy),
        2.0 * seg.radius, 2.0 * seg.radius,
        angle=0,
        theta1=a_start_deg,
        theta2=a_end_deg,
        color=color,
        linewidth=lw,
    )
    ax.add_patch(arc)


def plot_results(turbines, substations, obstacles, paths_info):
    """
    Plot the wind farm and planned paths.

    paths_info: list of (segments, start_ne, end_ne, color, label) tuples
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Plot turbine safety zones and planning margins
    for obs in obstacles:
        if obs.label.startswith("T"):
            safety_r = SAFETY_RADIUS
        else:
            safety_r = SUBSTATION_SAFETY_RADIUS

        # Planning margin (outer, dashed)
        planning_circle = plt.Circle(
            (obs.x, obs.y), obs.r,
            fill=False, edgecolor="#ffaaaa",
            linewidth=0.5, linestyle="--", alpha=0.5,
        )
        ax.add_patch(planning_circle)
        # Actual safety zone (inner, solid)
        safety_circle = plt.Circle(
            (obs.x, obs.y), safety_r,
            fill=True, facecolor="#ffe0e0", edgecolor="#ff6666",
            linewidth=0.5, alpha=0.6,
        )
        ax.add_patch(safety_circle)

    # Plot turbine positions
    te = [t["east"] for t in turbines]
    tn = [t["north"] for t in turbines]
    ax.scatter(te, tn, c="red", s=15, zorder=5, label="Wind turbines")

    # Plot substations
    se = [s["east"] for s in substations]
    sn = [s["north"] for s in substations]
    ax.scatter(se, sn, c="darkred", s=60, marker="^", zorder=5,
               label="Substations")

    # Plot paths
    path_colors = ["#0066cc", "#009933", "#cc6600", "#9900cc"]
    for idx, (segments, start_ne, end_ne, label) in enumerate(paths_info):
        color = path_colors[idx % len(path_colors)]

        if segments is None:
            print(f"  Path '{label}': No path found!")
            continue

        total_len = path_total_length(segments)
        direct_dist = math.hypot(end_ne[0] - start_ne[0], end_ne[1] - start_ne[1])
        n_arcs = sum(1 for s in segments if s.segment_type == "arc")
        n_lines = sum(1 for s in segments if s.segment_type == "line")

        print(f"  Path '{label}': {total_len:.1f}m total, {direct_dist:.1f}m direct, "
              f"{n_lines} line segments, {n_arcs} arcs")

        # Convert to waypoints for tracking module
        wps = segments_to_waypoints(segments)
        print(f"    Waypoints ({len(wps)}):")
        for wi, wp in enumerate(wps):
            lat, lon = north_east_to_latlon(wp.y, wp.x, ORIGIN_LAT, ORIGIN_LON)
            r_str = f"R={wp.turn_radius:.0f}m" if wp.turn_radius > 0 else "start/end"
            print(f"      WP{wi}: ({wp.x:7.1f}, {wp.y:7.1f}) m  "
                  f"lat={lat:.6f} lon={lon:.6f}  {r_str}")

        for seg in segments:
            if seg.segment_type == "line":
                ax.plot([seg.x0, seg.x1], [seg.y0, seg.y1],
                        color=color, linewidth=2.0, zorder=10)
            else:
                plot_arc(ax, seg, color, lw=2.0)

        # Start and end markers
        ax.plot(*start_ne, "o", color=color, markersize=10, zorder=15)
        ax.plot(*end_ne, "s", color=color, markersize=10, zorder=15)

        # Plot waypoints (intermediate ones with turn radius)
        for wp in wps:
            if wp.turn_radius > 0:
                ax.plot(wp.x, wp.y, "D", color=color, markersize=6, zorder=15,
                        markeredgecolor="white", markeredgewidth=0.5)

        # Direct line (dashed)
        ax.plot([start_ne[0], end_ne[0]], [start_ne[1], end_ne[1]],
                "--", color=color, linewidth=0.8, alpha=0.4)

        # Legend entry
        ax.plot([], [], color=color, linewidth=2.0, label=f"{label} ({total_len:.0f}m)")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Wind Farm Obstacle Avoidance - Visibility Graph Path Planning")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("obstacle/wind_farm_paths.png", dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to obstacle/wind_farm_paths.png")
    plt.show()


def main():
    print("Creating wind farm with turbines and 3 substations...")
    substations = create_substations()
    turbines = create_wind_farm(substations)
    obstacles = build_obstacles(turbines, substations)

    print(f"  {len(turbines)} turbines placed in variable grid (excluded from substation zones)")
    print(f"  {len(substations)} substations")
    print(f"  Turbine safety radius: {SAFETY_RADIUS}m")
    print(f"  Substation safety radius: {SUBSTATION_SAFETY_RADIUS}m")
    print(f"  Planning margin: {PLANNING_MARGIN}m")
    print(f"  Grid spacing: ~{GRID_SPACING}m with +/-{GRID_JITTER}m jitter")

    # The farm spans roughly east: -150..7300, north: -100..7400.
    # Start/end points are distributed around the farm -- some outside, some interior
    # -- all at safe distance from obstacles.

    # Convert NE offsets to lat/lon for path endpoints
    def ne_to_ll(north, east):
        return north_east_to_latlon(north, east, ORIGIN_LAT, ORIGIN_LON)

    paths_info = []

    # Path 1: From south edge to north-west interior, crossing several turbines
    print("\nPlanning Path 1: S to NW interior...")
    s_lat, s_lon = ne_to_ll(400, 3600)
    e_lat, e_lon = ne_to_ll(7600, 1200)
    segs, s_ne, e_ne = plan_path_latlon(s_lat, s_lon, e_lat, e_lon, obstacles)
    paths_info.append((segs, s_ne, e_ne, "S to NW interior"))

    # Path 2: East edge to west interior, aimed through substation S0 (east=2000, north=2400)
    # Shows the large R=520m detour clearly
    print("Planning Path 2: E to W through substation...")
    s_lat, s_lon = ne_to_ll(1200, 6000)
    e_lat, e_lon = ne_to_ll(2400, -500)
    segs, s_ne, e_ne = plan_path_latlon(s_lat, s_lon, e_lat, e_lon, obstacles)
    paths_info.append((segs, s_ne, e_ne, "E to W through substation"))

    # Path 3: Interior point to behind substation S1 (east=5600, north=4800)
    # Forces a sharp turn around the large exclusion zone
    print("Planning Path 3: Interior to behind substation...")
    s_lat, s_lon = ne_to_ll(4400, 400)
    e_lat, e_lon = ne_to_ll(5300, 5800)
    segs, s_ne, e_ne = plan_path_latlon(s_lat, s_lon, e_lat, e_lon, obstacles)
    paths_info.append((segs, s_ne, e_ne, "Interior to behind substation"))

    # Path 4: NE corner to SW corner, long diagonal across the farm
    print("Planning Path 4: NE to SW diagonal...")
    s_lat, s_lon = ne_to_ll(7600, 7600)
    e_lat, e_lon = ne_to_ll(400, 400)
    segs, s_ne, e_ne = plan_path_latlon(s_lat, s_lon, e_lat, e_lon, obstacles)
    paths_info.append((segs, s_ne, e_ne, "NE to SW diagonal"))

    print("\nPlotting results...")
    plot_results(turbines, substations, obstacles, paths_info)


if __name__ == "__main__":
    main()
