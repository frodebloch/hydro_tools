"""Scene manager — sets up the PyVista plotter, camera, lighting, and
manages the real-time animation loop.
"""

import numpy as np
import pyvista as pv
import vtk

from .ocean_surface import OceanSurface
from .vessel_geometry import VesselGeometry
from .turbine_geometry import TurbineGeometry


# Colour definitions
OCEAN_CMAP = "ocean"  # blue-themed colourmap for the ocean (matplotlib built-in)
VESSEL_COLOR = "#D04030"  # dark red hull
VESSEL_SUPER_COLOR = "#E8E0D0"  # off-white superstructure
GANGWAY_COLOR = "#F0C040"  # yellow/gold for gangway equipment
TURBINE_COLOR = "#C0C0C0"  # light grey
WATERLINE_COLOR = "#1A4A6A"  # dark blue for waterline reference


class Scene:
    """Manages the 3D visualization scene.

    Creates and owns the PyVista plotter, ocean mesh, vessel mesh, and
    turbine mesh. Provides an update() method called from the animation
    timer to refresh all geometry.
    """

    def __init__(
        self,
        ocean: OceanSurface,
        vessel: VesselGeometry,
        turbine: TurbineGeometry,
        ocean_size: float = 300.0,
        update_rate_hz: float = 10.0,
    ):
        self.ocean = ocean
        self.vessel = vessel
        self.turbine = turbine
        self.ocean_size = ocean_size
        self._update_interval_ms = int(1000.0 / update_rate_hz)

        # Track whether we've been initialized
        self._initialized = False
        self._follow_vessel = False

        # Callback for each frame — set by the caller
        self.on_update = None

    def build(self):
        """Create the plotter and add all actors."""
        self.plotter = pv.Plotter(
            title="DP Simulator Visualization",
            window_size=[1400, 900],
        )

        # Background: sky-like gradient
        self.plotter.set_background("lightblue", top="white")

        # ── Add ocean surface ──────────────────────────────────────
        # Use flat shading to avoid expensive per-frame normal recomputation.
        # show_edges=False and lighting=True give acceptable visual quality.
        self._ocean_actor = self.plotter.add_mesh(
            self.ocean.mesh,
            scalars="elevation",
            cmap=OCEAN_CMAP,
            clim=[-3.0, 3.0],
            show_scalar_bar=False,
            opacity=0.85,
            smooth_shading=False,
            name="ocean",
        )

        # ── Add waterline reference plane (thin transparent disc) ──
        waterline = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=self.ocean_size * 1.5,
            j_size=self.ocean_size * 1.5,
        )
        self.plotter.add_mesh(
            waterline,
            color=WATERLINE_COLOR,
            opacity=0.1,
            name="waterline",
        )

        # ── Add vessel ─────────────────────────────────────────────
        self._vessel_actor = self.plotter.add_mesh(
            self.vessel.mesh,
            color=VESSEL_COLOR,
            smooth_shading=True,
            name="vessel",
        )
        # Apply VTK transform to the actor for fast rigid-body updates
        self._vessel_actor.SetUserTransform(self.vessel._vtk_transform)

        # ── Add gangway (articulated parts) ─────────────────────────
        self._gangway_actors = []
        for i, (mesh, transform) in enumerate(self.vessel.gangway.meshes_and_transforms):
            actor = self.plotter.add_mesh(
                mesh,
                color=GANGWAY_COLOR,
                smooth_shading=True,
                name=f"gangway_{i}",
            )
            actor.SetUserTransform(transform)
            self._gangway_actors.append(actor)

        # ── Add turbine (static: spar + tower) ──────────────────────
        self._turbine_actor = self.plotter.add_mesh(
            self.turbine.mesh,
            color=TURBINE_COLOR,
            smooth_shading=True,
            name="turbine",
        )
        self._turbine_actor.SetUserTransform(self.turbine._vtk_transform)

        # ── Add nacelle (yaws to face wind) ─────────────────────────
        self._nacelle_actor = self.plotter.add_mesh(
            self.turbine.nacelle_mesh,
            color=TURBINE_COLOR,
            smooth_shading=True,
            name="nacelle",
        )
        self._nacelle_actor.SetUserTransform(self.turbine._nacelle_vtk_transform)

        # ── Add turbine rotor (yaws + spins) ────────────────────────
        self._rotor_actor = self.plotter.add_mesh(
            self.turbine.rotor_mesh,
            color=TURBINE_COLOR,
            smooth_shading=True,
            name="rotor",
        )
        self._rotor_actor.SetUserTransform(self.turbine._rotor_vtk_transform)

        # ── Axes and labels ────────────────────────────────────────
        self.plotter.add_axes(
            xlabel="East [m]",
            ylabel="North [m]",
            zlabel="Up [m]",
            line_width=2,
        )

        # ── Camera setup ──────────────────────────────────────────
        # Isometric-ish view looking at the scene from SE, elevated
        self.plotter.camera_position = [
            (250.0, -200.0, 120.0),   # camera position
            (100.0, 100.0, 0.0),       # focal point (between vessel and platform)
            (0.0, 0.0, 1.0),           # view up
        ]

        # ── Text overlay ──────────────────────────────────────────
        # Create VTK text actor directly for fast per-frame updates
        self._vtk_text = vtk.vtkTextActor()
        self._vtk_text.SetInput("t = 0.0 s")
        self._vtk_text.SetPosition(10, 10)
        tp = self._vtk_text.GetTextProperty()
        tp.SetFontSize(16)
        tp.SetFontFamilyToCourier()
        tp.SetColor(0.0, 0.0, 0.0)
        tp.SetVerticalJustificationToTop()
        self.plotter.renderer.AddActor2D(self._vtk_text)
        # Position at upper-left after window is created
        self._vtk_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._vtk_text.GetPositionCoordinate().SetValue(0.01, 0.99)
        tp.SetVerticalJustificationToTop()

        # ── Key bindings ──────────────────────────────────────────
        self.plotter.add_key_event("f", self._toggle_follow)
        self.plotter.add_key_event("r", self._reset_camera)

        self._initialized = True

    def start(self):
        """Start the animation loop and show the window."""
        if not self._initialized:
            self.build()

        # Use add_timer_event for repeating animation.
        # max_steps=2**31 gives us essentially infinite runtime.
        self.plotter.add_timer_event(
            max_steps=2**31,
            duration=self._update_interval_ms,
            callback=self._frame_callback,
        )

        # Show (blocks until window is closed)
        self.plotter.show()

    def _frame_callback(self, step: int):
        """Called by the PyVista timer each frame."""
        if self.on_update:
            self.on_update()

    def update_info_text(self, text: str):
        """Update the on-screen information text (fast — no actor recreation)."""
        self._vtk_text.SetInput(text)

    def _toggle_follow(self):
        """Toggle camera following the vessel."""
        self._follow_vessel = not self._follow_vessel
        mode = "ON" if self._follow_vessel else "OFF"
        print(f"Follow vessel: {mode}")

    def _reset_camera(self):
        """Reset camera to default position."""
        self.plotter.camera_position = [
            (250.0, -200.0, 120.0),
            (100.0, 100.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

    def follow_vessel_camera(self, north: float, east: float, heading_deg: float):
        """Move camera to follow the vessel if follow mode is active."""
        if not self._follow_vessel:
            return
        h = np.deg2rad(heading_deg)
        # Camera behind and above the vessel
        dist = 200.0
        cam_east = east - dist * np.sin(h)
        cam_north = north - dist * np.cos(h)
        cam_z = 80.0
        self.plotter.camera_position = [
            (cam_east, cam_north, cam_z),
            (east, north, 10.0),
            (0.0, 0.0, 1.0),
        ]
