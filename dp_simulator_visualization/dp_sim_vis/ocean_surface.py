"""Ocean surface mesh — PyVista StructuredGrid updated from wave model each frame."""

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk

from .wave_model import WaveElevation


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

        # Compute wave elevation over the grid
        z = self.wave.elevation(time, self._n_grid.ravel(), self._e_grid.ravel())
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
