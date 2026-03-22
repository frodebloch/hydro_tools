"""Scene manager — sets up the PyVista plotter, camera, lighting, and
manages the real-time animation loop.
"""

import numpy as np
import pyvista as pv
import vtk

from .ocean_surface import OceanSurface
from .vessel_geometry import VesselGeometry
from .turbine_geometry import TurbineGeometry
from .fixed_turbine_geometry import FixedTurbineGeometry


# Colour definitions
OCEAN_CMAP = "ocean"  # blue-themed colourmap for the ocean (matplotlib built-in)
VESSEL_COLOR = "#D04030"  # dark red hull
VESSEL_SUPER_COLOR = "#E8E0D0"  # off-white superstructure
GANGWAY_COLOR = "#F0C040"  # yellow/gold for gangway equipment
INDICATOR_MAX_COLOR = "#FF2020"   # red for max/min-length limit rings
INDICATOR_WARN_COLOR = "#FFB020"  # amber for 1m-before-max warning ring
TURBINE_COLOR = "#C0C0C0"  # light grey
WATERLINE_COLOR = "#1A4A6A"  # dark blue for waterline reference
GRID_COLOR = "#2A2A2A"       # dark grey for reference grid lines

# Strip chart configuration
STRIP_BUFFER = 1800   # samples in ring buffer (~120s at 15fps)
STRIP_WINDOW = 120.0  # visible time window [s]
CHART_BG_COLOR = (0, 0, 0, 180)  # semi-transparent black
CHART_TEXT_COLOR = (1.0, 1.0, 1.0)
CHART_LABEL_COLOR = (0.8, 0.8, 0.8)
DRIFT_COLOR = "#40C0FF"  # cyan-blue for drift forces
WIND_COLOR = "#40FF90"   # green for wind forces


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
        fixed_turbines: list[FixedTurbineGeometry] | None = None,
        far_ocean: OceanSurface | None = None,
        ocean_size: float = 300.0,
        update_rate_hz: float = 10.0,
    ):
        self.ocean = ocean
        self.far_ocean = far_ocean
        self.vessel = vessel
        self.turbine = turbine
        self.fixed_turbines = fixed_turbines or []
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

        # Anti-aliasing: FXAA is a post-processing shader that works on any
        # renderer (including software/Mesa).  Smooths all polygon edges.
        self.plotter.enable_anti_aliasing('fxaa')

        # Background: sky-like gradient
        self.plotter.set_background("lightblue", top="white")

        # ── Add ocean surface ──────────────────────────────────────
        # Use flat shading to avoid expensive per-frame normal recomputation.
        # show_edges=False and lighting=True give acceptable visual quality.
        # ── Add far ocean (coarse, reduced spectrum) if present ───────
        # Rendered first (behind), with slight Z depression so the near
        # ocean always covers it in the overlap region.
        self._far_ocean_actor = None
        if self.far_ocean is not None:
            self._far_ocean_actor = self.plotter.add_mesh(
                self.far_ocean.mesh,
                scalars="elevation",
                cmap=OCEAN_CMAP,
                clim=[-3.0, 3.0],
                show_scalar_bar=False,
                opacity=0.85,
                smooth_shading=False,
                show_edges=True,
                edge_color="#0A3050",
                name="far_ocean",
            )

        # ── Add ocean surface ──────────────────────────────────────
        # Use flat shading to avoid expensive per-frame normal recomputation.
        # show_edges=False and lighting=True give acceptable visual quality.
        # Fully opaque so it cleanly covers the far ocean in the overlap
        # region — avoids dark banding from two semi-transparent layers.
        self._ocean_actor = self.plotter.add_mesh(
            self.ocean.mesh,
            scalars="elevation",
            cmap=OCEAN_CMAP,
            clim=[-3.0, 3.0],
            show_scalar_bar=False,
            opacity=1.0,
            smooth_shading=False,
            name="ocean",
        )

        # ── Add waterline reference plane (thin transparent disc) ──
        waterline_size = self.far_ocean.size if self.far_ocean else self.ocean_size
        waterline = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=waterline_size * 1.5,
            j_size=waterline_size * 1.5,
        )
        self.plotter.add_mesh(
            waterline,
            color=WATERLINE_COLOR,
            opacity=0.1,
            name="waterline",
        )

        # ── Add reference grid lines on the ocean surface ──────────
        # When a far ocean is present, align grid lines exactly to its
        # vertex positions so the polylines connect seamlessly with the
        # far ocean's show_edges wireframe at the boundary.
        line_offsets = None
        if self.far_ocean is not None:
            far_half = self.far_ocean.size / 2.0
            far_res = self.far_ocean.resolution
            # Vertex offsets from centre — identical to np.linspace formula
            line_offsets = np.linspace(-far_half, far_half, far_res)
        grid_lines = self.ocean.build_grid_lines(line_offsets=line_offsets)
        # When grid lines are aligned to far ocean edges, use matching colour
        grid_color = "#0A3050" if self.far_ocean is not None else GRID_COLOR
        self._grid_actors = []
        for i, line in enumerate(grid_lines):
            actor = self.plotter.add_mesh(
                line,
                color=grid_color,
                line_width=1,
                opacity=0.5,
                name=f"grid_{i}",
            )
            self._grid_actors.append(actor)

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
        # Part indices: 0=tower_base, 1=tower_col, 2=outer_boom, 3=inner_boom.
        # The outer boom sleeve is semi-transparent so the indicator rings
        # on the inner boom are visible sliding through it.
        self._gangway_actors = []
        for i, (mesh, transform) in enumerate(self.vessel.gangway.meshes_and_transforms):
            is_outer_boom = (i == 2)
            actor = self.plotter.add_mesh(
                mesh,
                color=GANGWAY_COLOR,
                smooth_shading=True,
                opacity=0.45 if is_outer_boom else 1.0,
                name=f"gangway_{i}",
            )
            actor.SetUserTransform(transform)
            self._gangway_actors.append(actor)

        # ── Add gangway boom limit indicator rings ────────────────────
        _indicator_colors = {
            "max": INDICATOR_MAX_COLOR,
            "warn": INDICATOR_WARN_COLOR,
            "min": INDICATOR_MAX_COLOR,
        }
        self._indicator_actors = []
        for i, (mesh, transform, key) in enumerate(
            self.vessel.gangway.indicator_meshes_and_transforms
        ):
            actor = self.plotter.add_mesh(
                mesh,
                color=_indicator_colors[key],
                smooth_shading=False,
                name=f"gangway_indicator_{i}",
            )
            actor.SetUserTransform(transform)
            self._indicator_actors.append(actor)

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

        # ── Add fixed wind turbines (bottom-mounted) ──────────────────
        self._fixed_turbine_actors = []
        for i, ft in enumerate(self.fixed_turbines):
            # Tower + monopile
            tower_actor = self.plotter.add_mesh(
                ft.mesh,
                color=TURBINE_COLOR,
                smooth_shading=True,
                name=f"fixed_tower_{i}",
            )
            tower_actor.SetUserTransform(ft._vtk_transform)

            # Nacelle
            nacelle_actor = self.plotter.add_mesh(
                ft.nacelle_mesh,
                color=TURBINE_COLOR,
                smooth_shading=True,
                name=f"fixed_nacelle_{i}",
            )
            nacelle_actor.SetUserTransform(ft._nacelle_vtk_transform)

            # Rotor
            rotor_actor = self.plotter.add_mesh(
                ft.rotor_mesh,
                color=TURBINE_COLOR,
                smooth_shading=True,
                name=f"fixed_rotor_{i}",
            )
            rotor_actor.SetUserTransform(ft._rotor_vtk_transform)

            self._fixed_turbine_actors.append((tower_actor, nacelle_actor, rotor_actor))

        # ── Axes and labels ────────────────────────────────────────
        self.plotter.add_axes(
            xlabel="East [m]",
            ylabel="North [m]",
            zlabel="Up [m]",
            line_width=2,
        )

        # ── Camera setup ──────────────────────────────────────────
        # Default view: looking at the bow from ahead, slightly to starboard
        # and elevated.  The vessel sits at the origin with bow toward +Y.
        # If fixed turbines are present, pull camera back to see the farm.
        if self.fixed_turbines:
            self.plotter.camera_position = [
                (600.0, -400.0, 300.0),
                (300.0, 300.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
        else:
            self.plotter.camera_position = [
                (80.0, 180.0, 50.0),    # ahead of bow, starboard, elevated
                (0.0, 0.0, 10.0),       # looking at midship waterline
                (0.0, 0.0, 1.0),
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
        self.plotter.add_key_event("h", self._toggle_hud)

        # ── Strip charts (drift forces HUD) ────────────────────────
        self._build_strip_charts()

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

    def _toggle_hud(self):
        """Toggle strip chart HUD visibility."""
        self._hud_visible = not self._hud_visible
        self._surge_chart.visible = self._hud_visible
        self._sway_chart.visible = self._hud_visible
        mode = "ON" if self._hud_visible else "OFF"
        print(f"Force HUD: {mode}")

    def _build_strip_charts(self):
        """Create 2D overlay strip charts for drift + wind forces."""
        # Ring buffer arrays
        self._strip_t = np.zeros(STRIP_BUFFER)
        self._strip_drift_surge = np.zeros(STRIP_BUFFER)
        self._strip_drift_sway = np.zeros(STRIP_BUFFER)
        self._strip_wind_surge = np.zeros(STRIP_BUFFER)
        self._strip_wind_sway = np.zeros(STRIP_BUFFER)
        self._strip_head = 0
        self._strip_full = False
        self._hud_visible = True

        # Chart dimensions: right side, stacked vertically
        chart_w, chart_h = 0.30, 0.20

        # ── Surge force chart (upper) — drift + wind ────────────────
        self._surge_chart = pv.Chart2D(
            size=(chart_w, chart_h),
            loc=(0.68, 0.77),
        )
        self._surge_chart.background_color = CHART_BG_COLOR
        self._surge_chart.title = "Surge Forces [kN]"
        self._surge_chart.x_label = "Time [s]"
        self._surge_chart.y_label = ""
        self._style_chart_axes(self._surge_chart)

        self._drift_surge_line = self._surge_chart.line(
            np.zeros(2), np.zeros(2),
            color=DRIFT_COLOR, width=2.0, label="Drift",
        )
        self._wind_surge_line = self._surge_chart.line(
            np.zeros(2), np.zeros(2),
            color=WIND_COLOR, width=2.0, label="Wind",
        )
        # Zero reference line
        self._surge_chart.line(
            [0, 1], [0, 0],
            color="white", width=0.5, style="--",
        )

        # ── Sway force chart (lower) — drift + wind ────────────────
        self._sway_chart = pv.Chart2D(
            size=(chart_w, chart_h),
            loc=(0.68, 0.54),
        )
        self._sway_chart.background_color = CHART_BG_COLOR
        self._sway_chart.title = "Sway Forces [kN]"
        self._sway_chart.x_label = "Time [s]"
        self._sway_chart.y_label = ""
        self._style_chart_axes(self._sway_chart)

        self._drift_sway_line = self._sway_chart.line(
            np.zeros(2), np.zeros(2),
            color=DRIFT_COLOR, width=2.0,
        )
        self._wind_sway_line = self._sway_chart.line(
            np.zeros(2), np.zeros(2),
            color=WIND_COLOR, width=2.0,
        )
        # Zero reference line
        self._sway_chart.line(
            [0, 1], [0, 0],
            color="white", width=0.5, style="--",
        )

        self.plotter.add_chart(self._surge_chart, self._sway_chart)

        # Legend only on surge chart (top-left); sway uses same colours
        surge_legend = vtk.vtkChartXY.GetLegend(self._surge_chart)
        surge_legend.SetHorizontalAlignment(0)  # LEFT
        surge_legend.SetVerticalAlignment(2)    # TOP
        sway_legend = vtk.vtkChartXY.GetLegend(self._sway_chart)
        sway_legend.SetVisible(False)

    def _style_chart_axes(self, chart):
        """Apply consistent text styling to a chart's axes."""
        chart.GetTitleProperties().SetColor(*CHART_TEXT_COLOR)
        chart.GetTitleProperties().SetFontSize(12)
        for ax in (chart.x_axis, chart.y_axis):
            ax.GetTitleProperties().SetColor(*CHART_TEXT_COLOR)
            ax.GetTitleProperties().SetFontSize(10)
            ax.GetLabelProperties().SetColor(*CHART_LABEL_COLOR)
            ax.GetLabelProperties().SetFontSize(9)

    def update_strip_charts(self, sim_time: float,
                            drift_surge_kn: float, drift_sway_kn: float,
                            wind_surge_kn: float = 0.0, wind_sway_kn: float = 0.0):
        """Push one sample into the strip charts. Called each frame."""
        if not self._hud_visible:
            return

        h = self._strip_head
        self._strip_t[h] = sim_time
        self._strip_drift_surge[h] = drift_surge_kn
        self._strip_drift_sway[h] = drift_sway_kn
        self._strip_wind_surge[h] = wind_surge_kn
        self._strip_wind_sway[h] = wind_sway_kn

        self._strip_head = (h + 1) % STRIP_BUFFER
        if self._strip_head == 0:
            self._strip_full = True

        # Extract ordered slice for display
        if self._strip_full:
            idx = np.arange(self._strip_head,
                            self._strip_head + STRIP_BUFFER) % STRIP_BUFFER
        else:
            idx = np.arange(0, self._strip_head)

        if len(idx) < 2:
            return

        t_arr = self._strip_t[idx]
        self._drift_surge_line.update(t_arr, self._strip_drift_surge[idx])
        self._drift_sway_line.update(t_arr, self._strip_drift_sway[idx])
        self._wind_surge_line.update(t_arr, self._strip_wind_surge[idx])
        self._wind_sway_line.update(t_arr, self._strip_wind_sway[idx])

        # Scrolling X window
        t_max = sim_time
        t_min = max(0.0, t_max - STRIP_WINDOW)
        self._surge_chart.x_range = [t_min, t_max]
        self._sway_chart.x_range = [t_min, t_max]

        # Auto-scale Y axis based on visible data
        # Use symmetric range with some margin, minimum ±10 kN
        if self._strip_full:
            # Find visible data within the time window
            visible = t_arr >= t_min
            if np.any(visible):
                surge_drift_vis = self._strip_drift_surge[idx][visible]
                surge_wind_vis = self._strip_wind_surge[idx][visible]
                sway_drift_vis = self._strip_drift_sway[idx][visible]
                sway_wind_vis = self._strip_wind_sway[idx][visible]
            else:
                surge_drift_vis = self._strip_drift_surge[idx]
                surge_wind_vis = self._strip_wind_surge[idx]
                sway_drift_vis = self._strip_drift_sway[idx]
                sway_wind_vis = self._strip_wind_sway[idx]
        else:
            surge_drift_vis = self._strip_drift_surge[idx]
            surge_wind_vis = self._strip_wind_surge[idx]
            sway_drift_vis = self._strip_drift_sway[idx]
            sway_wind_vis = self._strip_wind_sway[idx]

        surge_max = max(10.0, np.max(np.abs(surge_drift_vis)) * 1.2,
                        np.max(np.abs(surge_wind_vis)) * 1.2)
        sway_max = max(10.0, np.max(np.abs(sway_drift_vis)) * 1.2,
                       np.max(np.abs(sway_wind_vis)) * 1.2)
        self._surge_chart.y_range = [-surge_max, surge_max]
        self._sway_chart.y_range = [-sway_max, sway_max]

    def _reset_camera(self):
        """Reset camera to default position."""
        if self.fixed_turbines:
            self.plotter.camera_position = [
                (600.0, -400.0, 300.0),
                (300.0, 300.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
        else:
            self.plotter.camera_position = [
                (80.0, 180.0, 50.0),
                (0.0, 0.0, 10.0),
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
