"""Ocean surface mesh — PyVista StructuredGrid updated from wave model each frame."""

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk

from .wave_model import WaveElevation


# Grid line parameters
GRID_SPACING = 20.0          # metres between grid lines
GRID_SAMPLES_PER_LINE = 80   # number of sample points along each line
GRID_Z_OFFSET = 0.05         # metres above wave surface to avoid z-fighting
GRID_MAX_EXTENT = 400.0      # max half-size for grid lines [m] — limits line count on large oceans


class OceanSurface:
    """Manages a PyVista mesh representing the ocean surface.

    The mesh is a rectangular grid of vertices whose Z-coordinates are updated
    each frame from the wave elevation model.
    """

    def __init__(
        self,
        wave_elevation: WaveElevation,
        size: float = 300.0,
        resolution: int = 100,
        center_north: float = 0.0,
        center_east: float = 0.0,
        inner_hole: float = 0.0,
    ):
        """
        Parameters
        ----------
        wave_elevation : WaveElevation
            Wave model instance providing surface heights.
        size : float
            Side length of the square ocean patch [m].
        resolution : int
            Number of vertices per side.
        center_north, center_east : float
            NED coordinates of the patch centre [m].
        inner_hole : float
            If > 0, cells whose centre falls inside this half-width are
            removed, leaving a ring-shaped mesh.  Used for the far ocean
            so it doesn't overlap with the near ocean.
        """
        self.wave = wave_elevation
        self.size = size
        self.resolution = resolution
        self.center_north = center_north
        self.center_east = center_east
        self._inner_hole = inner_hole

        # Build the grid coordinates
        half = size / 2.0
        n_vals = np.linspace(center_north - half, center_north + half, resolution)
        e_vals = np.linspace(center_east - half, center_east + half, resolution)
        self._e_grid, self._n_grid = np.meshgrid(e_vals, n_vals)

        n_pts = resolution * resolution

        if inner_hole > 0:
            # Build an UnstructuredGrid with the inner cells removed.
            # Vertices stay on the full grid (indices unchanged) so we
            # can still update Z with a flat array indexed by grid position.
            points = np.column_stack([
                self._e_grid.ravel(),
                self._n_grid.ravel(),
                np.zeros(n_pts),
            ])
            # Cell centres for the (res-1)x(res-1) quad grid
            e_centres = (self._e_grid[:-1, :-1] + self._e_grid[:-1, 1:]) / 2.0
            n_centres = (self._n_grid[:-1, :-1] + self._n_grid[1:, :-1]) / 2.0
            # Shrink the hole by one cell width so the outermost kept cells
            # overlap the near ocean edge — no gap between the two patches.
            cell_size = size / (resolution - 1)
            effective_hole = inner_hole - cell_size
            outside = (np.abs(e_centres) >= effective_hole) | (np.abs(n_centres) >= effective_hole)
            # Build quad cells for outside region only
            cells = []
            for row in range(resolution - 1):
                for col in range(resolution - 1):
                    if not outside[row, col]:
                        continue
                    i0 = row * resolution + col
                    i1 = i0 + 1
                    i2 = (row + 1) * resolution + col + 1
                    i3 = (row + 1) * resolution + col
                    cells.append([4, i0, i1, i2, i3])
            cells_arr = np.array(cells, dtype=np.int64).ravel()
            n_cells = len(cells)
            cell_types = np.full(n_cells, 9, dtype=np.uint8)  # VTK_QUAD = 9
            self.mesh = pv.UnstructuredGrid(cells_arr, cell_types, points)
        else:
            # Standard StructuredGrid (no hole)
            z = np.zeros_like(self._n_grid)
            self.mesh = pv.StructuredGrid(
                self._e_grid, self._n_grid, z
            )
            self.mesh.dimensions = [resolution, resolution, 1]

        # Pre-allocate a contiguous points buffer for fast VTK updates.
        self._pts_buf = np.empty((n_pts, 3), dtype=np.float64)
        self._pts_buf[:, 0] = self._e_grid.ravel()
        self._pts_buf[:, 1] = self._n_grid.ravel()
        self._pts_buf[:, 2] = 0.0

        # Pre-allocate elevation scalar buffer
        self._elev_buf = np.zeros(n_pts, dtype=np.float64)
        self.mesh["elevation"] = self._elev_buf

    def update(self, time: float, center_north: float = None, center_east: float = None):
        """Recompute surface elevation for the current simulation time.

        Optionally re-centre the grid (e.g. to follow the vessel).
        """
        if center_north is not None or center_east is not None:
            if center_north is not None:
                self.center_north = center_north
            if center_east is not None:
                self.center_east = center_east
            self._rebuild_grid_coords()

        # Compute wave elevation over the grid (positive down in NED, negate for VTK Z-up)
        z = -self.wave.elevation(time, self._n_grid.ravel(), self._e_grid.ravel())
        z_flat = z.ravel()

        # Update Z in the pre-allocated buffer and push to VTK
        self._pts_buf[:, 2] = z_flat
        vtk_pts = numpy_to_vtk(self._pts_buf, deep=False)
        self.mesh.GetPoints().SetData(vtk_pts)
        self.mesh.GetPoints().Modified()

        # Update elevation scalar via direct VTK array (no PyVista overhead)
        self._elev_buf[:] = z_flat
        vtk_scalars = numpy_to_vtk(self._elev_buf, deep=False)
        vtk_scalars.SetName("elevation")
        self.mesh.GetPointData().SetScalars(vtk_scalars)
        self.mesh.GetPointData().Modified()

    def follow(self, vessel_north: float, vessel_east: float,
               snap_step: float = GRID_SPACING) -> bool:
        """Re-centre the ocean if the vessel has moved beyond one snap step.

        Returns True if the grid was shifted, False otherwise.
        The snap step defaults to GRID_SPACING (20m) so the near ocean
        jumps in grid-aligned increments — no visual pop because the
        wave model evaluates at absolute world coordinates.
        """
        target_n = round(vessel_north / snap_step) * snap_step
        target_e = round(vessel_east / snap_step) * snap_step
        if target_n == self.center_north and target_e == self.center_east:
            return False
        self.center_north = target_n
        self.center_east = target_e
        self._rebuild_grid_coords()
        self._rebuild_grid_line_coords()
        return True

    def _rebuild_grid_coords(self):
        """Recompute the E/N grid arrays and update the points buffer XY."""
        half = self.size / 2.0
        n_vals = np.linspace(
            self.center_north - half, self.center_north + half, self.resolution
        )
        e_vals = np.linspace(
            self.center_east - half, self.center_east + half, self.resolution
        )
        self._e_grid, self._n_grid = np.meshgrid(e_vals, n_vals)
        self._pts_buf[:, 0] = self._e_grid.ravel()
        self._pts_buf[:, 1] = self._n_grid.ravel()

    @property
    def z_range(self) -> tuple[float, float]:
        """Current min/max elevation — useful for colormap range."""
        if "elevation" in self.mesh.point_data:
            e = self.mesh["elevation"]
            return float(e.min()), float(e.max())
        return -1.0, 1.0

    # ── Wave-following grid lines ──────────────────────────────────────

    def build_grid_lines(self) -> list[pv.PolyData]:
        """Create polyline meshes for a North/East reference grid.

        Returns a list of PolyData polylines at fixed N and E positions,
        spanning up to GRID_MAX_EXTENT around the centre.  Their Z
        coordinates are updated each frame by update_grid_lines().
        """
        half = min(self.size / 2.0, GRID_MAX_EXTENT)
        n_min = self.center_north - half
        n_max = self.center_north + half
        e_min = self.center_east - half
        e_max = self.center_east + half

        # Snap grid origin to multiples of GRID_SPACING so lines stay at
        # round-number world coordinates regardless of patch centre.
        first_e = np.ceil(e_min / GRID_SPACING) * GRID_SPACING
        first_n = np.ceil(n_min / GRID_SPACING) * GRID_SPACING

        self._grid_lines: list[pv.PolyData] = []
        # Coordinate arrays for each line's sample points (for elevation queries)
        self._grid_north_arrs: list[np.ndarray] = []
        self._grid_east_arrs: list[np.ndarray] = []

        n_samples = GRID_SAMPLES_PER_LINE

        # East-West lines (constant North, varying East)
        n_pos = first_n
        while n_pos <= n_max:
            east_arr = np.linspace(e_min, e_max, n_samples)
            north_arr = np.full(n_samples, n_pos)
            pts = np.column_stack([east_arr, north_arr, np.zeros(n_samples)])
            line = pv.lines_from_points(pts)
            self._grid_lines.append(line)
            self._grid_north_arrs.append(north_arr)
            self._grid_east_arrs.append(east_arr)
            n_pos += GRID_SPACING

        # North-South lines (constant East, varying North)
        e_pos = first_e
        while e_pos <= e_max:
            north_arr = np.linspace(n_min, n_max, n_samples)
            east_arr = np.full(n_samples, e_pos)
            pts = np.column_stack([east_arr, north_arr, np.zeros(n_samples)])
            line = pv.lines_from_points(pts)
            self._grid_lines.append(line)
            self._grid_north_arrs.append(north_arr)
            self._grid_east_arrs.append(east_arr)
            e_pos += GRID_SPACING

        return self._grid_lines

    def update_grid_lines(self, time: float):
        """Update grid line Z coordinates to follow the wave surface."""
        if not hasattr(self, '_grid_lines'):
            return
        for i, line in enumerate(self._grid_lines):
            z = -self.wave.elevation(
                time, self._grid_north_arrs[i], self._grid_east_arrs[i]
            )
            z += GRID_Z_OFFSET
            pts = line.points
            pts[:, 2] = z
            line.points = pts

    def _rebuild_grid_line_coords(self):
        """Update grid line XY sample positions after an ocean re-centre.

        Keeps the same PolyData objects (already added to the plotter) but
        shifts their sample coordinates to the new ocean centre, snapping
        to GRID_SPACING world coordinates.
        """
        if not hasattr(self, '_grid_lines'):
            return

        half = min(self.size / 2.0, GRID_MAX_EXTENT)
        n_min = self.center_north - half
        n_max = self.center_north + half
        e_min = self.center_east - half
        e_max = self.center_east + half

        first_e = np.ceil(e_min / GRID_SPACING) * GRID_SPACING
        first_n = np.ceil(n_min / GRID_SPACING) * GRID_SPACING

        n_samples = GRID_SAMPLES_PER_LINE
        idx = 0

        # East-West lines (constant North, varying East)
        n_pos = first_n
        while n_pos <= n_max and idx < len(self._grid_lines):
            east_arr = np.linspace(e_min, e_max, n_samples)
            north_arr = np.full(n_samples, n_pos)
            self._grid_east_arrs[idx] = east_arr
            self._grid_north_arrs[idx] = north_arr
            pts = self._grid_lines[idx].points
            pts[:, 0] = east_arr
            pts[:, 1] = north_arr
            self._grid_lines[idx].points = pts
            idx += 1
            n_pos += GRID_SPACING

        # North-South lines (constant East, varying North)
        e_pos = first_e
        while e_pos <= e_max and idx < len(self._grid_lines):
            north_arr = np.linspace(n_min, n_max, n_samples)
            east_arr = np.full(n_samples, e_pos)
            self._grid_north_arrs[idx] = north_arr
            self._grid_east_arrs[idx] = east_arr
            pts = self._grid_lines[idx].points
            pts[:, 0] = east_arr
            pts[:, 1] = north_arr
            self._grid_lines[idx].points = pts
            idx += 1
            e_pos += GRID_SPACING
