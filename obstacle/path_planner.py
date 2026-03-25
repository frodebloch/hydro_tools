"""
Visibility graph path planner for circular obstacles.

Finds the shortest path composed of line segments and circular arcs
between a start and end point, avoiding circular obstacles.
"""

import math
import heapq
from dataclasses import dataclass, field


@dataclass
class Circle:
    """A circular obstacle in the north/east plane."""
    x: float  # east position (meters)
    y: float  # north position (meters)
    r: float  # radius (meters)
    label: str = ""


@dataclass
class Waypoint:
    """A waypoint for the tracking module.

    The waypoint is placed at the intersection of the incoming and outgoing legs.
    The turn radius defines the arc that the tracking module will inscribe at the
    waypoint. For obstacle avoidance waypoints, this equals the planning radius
    (safety zone + clearance margin).
    """
    x: float        # east position (meters)
    y: float        # north position (meters)
    turn_radius: float  # meters, 0 = straight through (start/end)


@dataclass
class PathSegment:
    """A segment of the planned path - either a line or an arc."""
    segment_type: str  # "line" or "arc"
    # For lines: start and end points
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    # For arcs: center, radius, start angle, end angle, direction
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    angle_start: float = 0.0  # radians
    angle_end: float = 0.0    # radians
    cw: bool = False  # True = clockwise

    def length(self) -> float:
        if self.segment_type == "line":
            return math.hypot(self.x1 - self.x0, self.y1 - self.y0)
        else:
            delta = self.angle_end - self.angle_start
            if self.cw:
                if delta > 0:
                    delta -= 2.0 * math.pi
                return abs(delta) * self.radius
            else:
                if delta < 0:
                    delta += 2.0 * math.pi
                return abs(delta) * self.radius


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi)."""
    while a >= math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _angle_diff_ccw(a_from: float, a_to: float) -> float:
    """Counter-clockwise angular distance from a_from to a_to, in [0, 2*pi)."""
    d = a_to - a_from
    while d < 0:
        d += 2.0 * math.pi
    while d >= 2.0 * math.pi:
        d -= 2.0 * math.pi
    return d


def _angle_diff_cw(a_from: float, a_to: float) -> float:
    """Clockwise angular distance from a_from to a_to, in [0, 2*pi)."""
    return _angle_diff_ccw(a_to, a_from)


def _external_tangents(c1: Circle, c2: Circle) -> list[tuple[float, float, float, float, float, float]]:
    """
    Compute external tangent lines between two circles.
    Returns list of (x1, y1, x2, y2, angle_on_c1, angle_on_c2) tuples,
    where (x1,y1) is tangent point on c1, (x2,y2) on c2.
    """
    dx = c2.x - c1.x
    dy = c2.y - c1.y
    d = math.hypot(dx, dy)
    if d < 1e-10:
        return []

    # Angle from c1 to c2
    theta = math.atan2(dy, dx)
    dr = c1.r - c2.r

    if d < abs(dr):
        return []  # One circle inside the other

    if abs(dr) > d:
        return []

    alpha = math.asin(max(-1.0, min(1.0, dr / d)))

    results = []
    for sign in [1.0, -1.0]:
        angle = theta + sign * (math.pi / 2.0 - alpha)
        # Tangent points
        x1 = c1.x + c1.r * math.cos(angle)
        y1 = c1.y + c1.r * math.sin(angle)
        x2 = c2.x + c2.r * math.cos(angle)
        y2 = c2.y + c2.r * math.sin(angle)
        a1 = _normalize_angle(angle)
        a2 = _normalize_angle(angle)
        results.append((x1, y1, x2, y2, a1, a2))

    return results


def _internal_tangents(c1: Circle, c2: Circle) -> list[tuple[float, float, float, float, float, float]]:
    """
    Compute internal (cross) tangent lines between two circles.
    Returns list of (x1, y1, x2, y2, angle_on_c1, angle_on_c2) tuples.
    """
    dx = c2.x - c1.x
    dy = c2.y - c1.y
    d = math.hypot(dx, dy)
    if d < 1e-10:
        return []

    theta = math.atan2(dy, dx)
    sr = c1.r + c2.r

    if d < sr:
        return []  # Circles overlap, no internal tangents

    if abs(sr) > d:
        return []

    alpha = math.asin(max(-1.0, min(1.0, sr / d)))

    results = []
    for sign in [1.0, -1.0]:
        angle1 = theta + sign * (math.pi / 2.0 - alpha)
        angle2 = angle1 + math.pi  # Opposite side for internal tangent
        # Tangent points
        x1 = c1.x + c1.r * math.cos(angle1)
        y1 = c1.y + c1.r * math.sin(angle1)
        x2 = c2.x + c2.r * math.cos(angle2)
        y2 = c2.y + c2.r * math.sin(angle2)
        a1 = _normalize_angle(angle1)
        a2 = _normalize_angle(angle2)
        results.append((x1, y1, x2, y2, a1, a2))

    return results


def _point_to_circle_tangents(px: float, py: float, c: Circle) -> list[tuple[float, float, float, float, float]]:
    """
    Compute tangent lines from a point to a circle.
    Returns list of (px, py, tx, ty, angle_on_circle) tuples.
    """
    dx = c.x - px
    dy = c.y - py
    d = math.hypot(dx, dy)
    if d < c.r + 1e-10:
        return []  # Point inside or on circle

    theta = math.atan2(dy, dx)
    alpha = math.asin(max(-1.0, min(1.0, c.r / d)))

    results = []
    for sign in [1.0, -1.0]:
        # Angle from point to tangent point, measured from center of circle
        tangent_angle = theta + sign * alpha + math.pi  # +pi because tangent is on far side
        # Actually, let's compute it properly
        # The tangent point on the circle:
        # angle from circle center to tangent point
        beta = theta + math.pi + sign * (math.pi / 2.0 - alpha)
        tx = c.x + c.r * math.cos(beta)
        ty = c.y + c.r * math.sin(beta)
        a = _normalize_angle(beta)
        results.append((px, py, tx, ty, a))

    return results


def _segment_intersects_circle(x1: float, y1: float, x2: float, y2: float,
                                c: Circle, margin: float = 1e-6) -> bool:
    """Check if a line segment intersects (enters) a circle."""
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - c.x
    fy = y1 - c.y

    a = dx * dx + dy * dy
    if a < 1e-20:
        # Degenerate segment (point)
        return math.hypot(fx, fy) < c.r - margin

    b = 2.0 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - (c.r - margin) ** 2

    discriminant = b * b - 4.0 * a * cc
    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Check if any intersection point is within the segment [0, 1]
    # We need the segment to actually enter the circle (both t values bracket some range in [0,1])
    if t1 > 1.0 + 1e-10 or t2 < -1e-10:
        return False

    return True


def _tangent_blocked(x1: float, y1: float, x2: float, y2: float,
                     obstacles: list[Circle], skip_indices: set) -> bool:
    """Check if a tangent line segment is blocked by any obstacle."""
    for i, obs in enumerate(obstacles):
        if i in skip_indices:
            continue
        if _segment_intersects_circle(x1, y1, x2, y2, obs):
            return True
    return False


@dataclass(order=True)
class _Node:
    cost: float
    index: int = field(compare=False)


def find_path(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[Circle],
    corridor_width: float | None = None,
) -> list[PathSegment] | None:
    """
    Find the shortest line-arc path from start to end avoiding circular obstacles.

    Args:
        start: (east, north) start position in meters
        end: (east, north) end position in meters
        obstacles: list of Circle obstacles
        corridor_width: if set, pre-screen obstacles to within this distance of
                        the start-end line. If None, use all obstacles.

    Returns:
        List of PathSegment objects forming the shortest path, or None if no path found.
    """
    sx, sy = start
    ex, ey = end

    # Pre-screen obstacles within corridor
    if corridor_width is not None:
        filtered = []
        # Distance from obstacle center to the start-end line segment
        seg_dx = ex - sx
        seg_dy = ey - sy
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        for obs in obstacles:
            if seg_len_sq < 1e-20:
                dist = math.hypot(obs.x - sx, obs.y - sy)
            else:
                t = max(0.0, min(1.0, ((obs.x - sx) * seg_dx + (obs.y - sy) * seg_dy) / seg_len_sq))
                proj_x = sx + t * seg_dx
                proj_y = sy + t * seg_dy
                dist = math.hypot(obs.x - proj_x, obs.y - proj_y)
            if dist - obs.r < corridor_width:
                filtered.append(obs)
        obstacles = filtered

    n_obs = len(obstacles)

    # Build graph nodes:
    # Node 0: start
    # Node 1: end
    # For each obstacle i (0..n_obs-1), tangent points are added dynamically.
    # We'll collect all tangent points and then build adjacency.

    # Each tangent point is: (x, y, obstacle_index, angle_on_obstacle)
    # obstacle_index = -1 for start, -2 for end

    tangent_points: list[tuple[float, float, int, float]] = []
    # Index 0: start point
    tangent_points.append((sx, sy, -1, 0.0))
    # Index 1: end point
    tangent_points.append((ex, ey, -2, 0.0))

    # Adjacency: list of (neighbor_index, cost)
    adjacency: dict[int, list[tuple[int, float]]] = {}

    def add_edge(i: int, j: int, cost: float):
        if i not in adjacency:
            adjacency[i] = []
        adjacency[i].append((j, cost))

    # Collect tangent points per obstacle for arc computation
    obstacle_tangent_points: dict[int, list[tuple[int, float]]] = {}  # obs_idx -> [(node_idx, angle)]

    def register_tangent_point(x: float, y: float, obs_idx: int, angle: float) -> int:
        idx = len(tangent_points)
        tangent_points.append((x, y, obs_idx, angle))
        if obs_idx >= 0:
            if obs_idx not in obstacle_tangent_points:
                obstacle_tangent_points[obs_idx] = []
            obstacle_tangent_points[obs_idx].append((idx, angle))
        return idx

    # 1. Start to end direct
    if not _tangent_blocked(sx, sy, ex, ey, obstacles, set()):
        dist = math.hypot(ex - sx, ey - sy)
        add_edge(0, 1, dist)

    # 2. Start point to each obstacle
    for i, obs in enumerate(obstacles):
        tangents = _point_to_circle_tangents(sx, sy, obs)
        for (px, py, tx, ty, angle) in tangents:
            if _tangent_blocked(px, py, tx, ty, obstacles, {i}):
                continue
            idx = register_tangent_point(tx, ty, i, angle)
            dist = math.hypot(tx - px, ty - py)
            add_edge(0, idx, dist)

    # 3. Each obstacle to end point
    for i, obs in enumerate(obstacles):
        tangents = _point_to_circle_tangents(ex, ey, obs)
        for (px, py, tx, ty, angle) in tangents:
            if _tangent_blocked(px, py, tx, ty, obstacles, {i}):
                continue
            idx = register_tangent_point(tx, ty, i, angle)
            dist = math.hypot(tx - px, ty - py)
            add_edge(idx, 1, dist)

    # 4. Obstacle to obstacle tangents
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            skip = {i, j}
            # External tangents
            for (x1, y1, x2, y2, a1, a2) in _external_tangents(obstacles[i], obstacles[j]):
                if _tangent_blocked(x1, y1, x2, y2, obstacles, skip):
                    continue
                idx1 = register_tangent_point(x1, y1, i, a1)
                idx2 = register_tangent_point(x2, y2, j, a2)
                dist = math.hypot(x2 - x1, y2 - y1)
                add_edge(idx1, idx2, dist)
                add_edge(idx2, idx1, dist)

            # Internal tangents
            for (x1, y1, x2, y2, a1, a2) in _internal_tangents(obstacles[i], obstacles[j]):
                if _tangent_blocked(x1, y1, x2, y2, obstacles, skip):
                    continue
                idx1 = register_tangent_point(x1, y1, i, a1)
                idx2 = register_tangent_point(x2, y2, j, a2)
                dist = math.hypot(x2 - x1, y2 - y1)
                add_edge(idx1, idx2, dist)
                add_edge(idx2, idx1, dist)

    # 5. Arc edges between tangent points on the same obstacle
    for obs_idx, tps in obstacle_tangent_points.items():
        obs = obstacles[obs_idx]
        # Sort by angle
        tps_sorted = sorted(tps, key=lambda t: t[1])
        n_tp = len(tps_sorted)
        for ii in range(n_tp):
            for jj in range(ii + 1, n_tp):
                idx_a, angle_a = tps_sorted[ii]
                idx_b, angle_b = tps_sorted[jj]

                # CCW arc from a to b
                arc_ccw = _angle_diff_ccw(angle_a, angle_b) * obs.r
                # CW arc from a to b
                arc_cw = _angle_diff_cw(angle_a, angle_b) * obs.r

                # Use the shorter arc
                arc_len = min(arc_ccw, arc_cw)
                add_edge(idx_a, idx_b, arc_len)
                add_edge(idx_b, idx_a, arc_len)

    # Dijkstra
    n_nodes = len(tangent_points)
    dist_to = [float("inf")] * n_nodes
    prev_node = [-1] * n_nodes
    dist_to[0] = 0.0

    pq = [_Node(0.0, 0)]
    visited = set()

    while pq:
        node = heapq.heappop(pq)
        u = node.index
        if u in visited:
            continue
        visited.add(u)

        if u == 1:
            break

        for (v, w) in adjacency.get(u, []):
            new_dist = dist_to[u] + w
            if new_dist < dist_to[v]:
                dist_to[v] = new_dist
                prev_node[v] = u
                heapq.heappush(pq, _Node(new_dist, v))

    if dist_to[1] == float("inf"):
        return None  # No path found

    # Reconstruct path
    path_indices = []
    current = 1
    while current != -1:
        path_indices.append(current)
        current = prev_node[current]
    path_indices.reverse()

    # Convert to PathSegment list
    segments = []
    for k in range(len(path_indices) - 1):
        i_from = path_indices[k]
        i_to = path_indices[k + 1]
        tp_from = tangent_points[i_from]
        tp_to = tangent_points[i_to]

        obs_from = tp_from[2]
        obs_to = tp_to[2]

        if obs_from == obs_to and obs_from >= 0:
            # Arc segment on the same obstacle
            obs = obstacles[obs_from]
            a_from = tp_from[3]
            a_to = tp_to[3]
            # Choose shorter arc
            ccw_len = _angle_diff_ccw(a_from, a_to)
            cw_len = _angle_diff_cw(a_from, a_to)
            if ccw_len <= cw_len:
                segments.append(PathSegment(
                    segment_type="arc",
                    x0=tp_from[0], y0=tp_from[1],
                    x1=tp_to[0], y1=tp_to[1],
                    cx=obs.x, cy=obs.y,
                    radius=obs.r,
                    angle_start=a_from, angle_end=a_to,
                    cw=False,
                ))
            else:
                segments.append(PathSegment(
                    segment_type="arc",
                    x0=tp_from[0], y0=tp_from[1],
                    x1=tp_to[0], y1=tp_to[1],
                    cx=obs.x, cy=obs.y,
                    radius=obs.r,
                    angle_start=a_from, angle_end=a_to,
                    cw=True,
                ))
        else:
            # Line segment
            segments.append(PathSegment(
                segment_type="line",
                x0=tp_from[0], y0=tp_from[1],
                x1=tp_to[0], y1=tp_to[1],
            ))

    return segments


def path_total_length(segments: list[PathSegment]) -> float:
    """Compute total path length in meters."""
    return sum(s.length() for s in segments)


def segments_to_waypoints(segments: list[PathSegment]) -> list[Waypoint]:
    """
    Convert a path (list of PathSegments) to a waypoint list for the tracking module.

    Each obstacle avoidance turn becomes a waypoint at the intersection of the
    incoming and outgoing line segments, with turn_radius set to the arc radius.
    Start and end points are included with turn_radius=0.
    """
    if not segments:
        return []

    waypoints = []

    # Start point
    waypoints.append(Waypoint(x=segments[0].x0, y=segments[0].y0, turn_radius=0.0))

    # Walk through segments, grouping line-arc-line sequences into waypoints
    i = 0
    while i < len(segments):
        seg = segments[i]

        if seg.segment_type == "line":
            # Check if next segment is an arc (line-arc-line pattern)
            if i + 1 < len(segments) and segments[i + 1].segment_type == "arc":
                arc = segments[i + 1]
                # Find the intersection of the incoming and outgoing legs.
                # Incoming leg: this line segment, extended forward.
                # Outgoing leg: the line segment after the arc, extended backward.
                if i + 2 < len(segments):
                    next_seg = segments[i + 2]
                    # Incoming direction
                    ix = _line_line_intersection(
                        seg.x0, seg.y0, seg.x1, seg.y1,
                        next_seg.x0, next_seg.y0, next_seg.x1, next_seg.y1,
                    )
                    if ix is not None:
                        waypoints.append(Waypoint(
                            x=ix[0], y=ix[1], turn_radius=arc.radius,
                        ))
                    else:
                        # Lines are parallel -- use the arc midpoint as waypoint
                        mid_angle = (arc.angle_start + arc.angle_end) / 2.0
                        waypoints.append(Waypoint(
                            x=arc.cx + arc.radius * math.cos(mid_angle),
                            y=arc.cy + arc.radius * math.sin(mid_angle),
                            turn_radius=arc.radius,
                        ))
                    # Skip the arc, next iteration will process the outgoing line
                    i += 2
                    continue
                else:
                    # Arc is the last segment (no outgoing line) -- rare edge case
                    i += 1
                    continue
            # Plain line segment with no arc following -- no waypoint needed mid-path
            i += 1
            continue

        elif seg.segment_type == "arc":
            # Arc without a preceding line in this walk (e.g. consecutive arcs
            # on the same obstacle). Skip -- the arc contribution is captured
            # by the surrounding line-arc-line patterns.
            i += 1
            continue

    # End point
    waypoints.append(Waypoint(
        x=segments[-1].x1, y=segments[-1].y1, turn_radius=0.0,
    ))

    return waypoints


def _line_line_intersection(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> tuple[float, float] | None:
    """Compute intersection of two lines (extended, not just segments).

    Line 1 passes through (x1,y1)-(x2,y2), line 2 through (x3,y3)-(x4,y4).
    Returns (x, y) or None if lines are parallel.
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return ix, iy
