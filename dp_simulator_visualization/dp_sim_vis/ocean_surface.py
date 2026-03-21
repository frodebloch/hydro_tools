"""Ocean surface mesh — PyVista StructuredGrid updated from wave model each frame."""

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk

from .wave_model import WaveElevation


# Grid line parameters
GRID_SPACING = 20.0          # metres between grid lines
GRID_SAMPLES_PER_LINE = 80   # number of sample points along each line
GRID_Z_OFFSET = 0.05         # metres above wave surface to avoid z-fighting


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
        """
        self.wave = wave_elevation
        self.size = size
        self.resolution = resolution
        self.center_north = center_north
        self.center_east = center_east

        # Build the grid coordinates
        half = size / 2.0
        n_vals = np.linspace(center_north - half, center_north + half, resolution)
        e_vals = np.linspace(center_east - half, center_east + half, resolution)
        self._e_grid, self._n_grid = np.meshgrid(e_vals, n_vals)
        z = np.zeros_like(self._n_grid)

        # Create the StructuredGrid (VTK uses X=East, Y=North, Z=Up)
        self.mesh = pv.StructuredGrid(
            self._e_grid, self._n_grid, z
        )
        self.mesh.dimensions = [resolution, resolution, 1]

        # Pre-allocate a contiguous points buffer for fast VTK updates.
        # This avoids per-frame allocation: we write into this array then push
        # it to VTK via numpy_to_vtk (zero-copy when possible).
        n_pts = resolution * resolution
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
        spanning the ocean patch.  Their Z coordinates are updated each
        frame by update_grid_lines().
        """
        half = self.size / 2.0
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
