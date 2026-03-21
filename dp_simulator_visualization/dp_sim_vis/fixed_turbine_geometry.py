"""Fixed (bottom-mounted) wind turbine geometry.

Reuses the NREL 5MW tower, nacelle, and rotor from turbine_geometry.py
but replaces the OC3 Hywind spar with a monopile foundation.

Each instance has its own rotor angle state so turbines spin independently.
The nacelle yaws to face the wind, same as the floating turbine.
"""

import numpy as np
import pyvista as pv
import vtk

from .turbine_geometry import (
    _build_tower,
    _build_nacelle,
    _build_rotor,
    rotor_speed,
    HUB_HEIGHT,
    SHAFT_OVERHANG,
    TOWER_BASE_DIAMETER,
    ROTOR_TIME_CONSTANT,
)

# Monopile dimensions
MONOPILE_DIAMETER = 7.1      # m (HKN tower diameter from config)
MONOPILE_DEPTH = 30.0        # m below SWL (seabed depth)
MONOPILE_TOP = 10.0          # m above SWL (transition piece top = tower base)


def _build_monopile(diameter: float = MONOPILE_DIAMETER) -> pv.PolyData:
    """Monopile cylinder from seabed to tower base."""
    monopile = pv.Cylinder(
        center=(0, 0, (-MONOPILE_DEPTH + MONOPILE_TOP) / 2.0),
        direction=(0, 0, 1),
        radius=diameter / 2.0,
        height=MONOPILE_DEPTH + MONOPILE_TOP,
        resolution=12,
        capping=True,
    )
    return monopile


class FixedTurbineGeometry:
    """Bottom-mounted wind turbine with monopile foundation.

    Three separate meshes with independent VTK transforms:
        mesh          -- monopile + tower (static, only translated)
        nacelle_mesh  -- nacelle box (yaws to face wind)
        rotor_mesh    -- 3 blades + hub (yaws + spins)
    """

    def __init__(self, name: str, north: float, east: float, azimuth_deg: float = 0.0):
        """
        Parameters
        ----------
        name : str
            Structure identifier (e.g. "SF A01").
        north, east : float
            Fixed NED position [m].
        azimuth_deg : float
            As-built azimuth [deg, 0=N, CW]. Not used for nacelle yaw
            (nacelle tracks wind independently) but stored for reference.
        """
        self.name = name
        self.north = north
        self.east = east
        self.azimuth_deg = azimuth_deg

        # Build geometry
        monopile = _build_monopile()
        tower = _build_tower()
        self.mesh = monopile.merge(tower)

        self.nacelle_mesh = _build_nacelle()
        self.rotor_mesh = _build_rotor()

        # VTK transforms
        self._vtk_transform = vtk.vtkTransform()
        self._vtk_transform.PostMultiply()

        self._nacelle_vtk_transform = vtk.vtkTransform()
        self._nacelle_vtk_transform.PostMultiply()

        self._rotor_vtk_transform = vtk.vtkTransform()
        self._rotor_vtk_transform.PostMultiply()

        # Rotor state
        self._rotor_angle_deg = 0.0
        self._omega = 0.0
        self._nacelle_yaw_deg = 0.0

        # Set initial position (static — only needs to be called once,
        # then again each frame just for rotor angle and nacelle yaw)
        self.update_transform(wind_from_deg=0.0)

    def update_transform(self, wind_from_deg: float = 0.0):
        """Update transforms — position is fixed, only nacelle yaw and rotor spin change.

        Args:
            wind_from_deg: Wind coming-from direction [deg, 0=N, CW].
        """
        self._nacelle_yaw_deg = wind_from_deg

        # Static structure (monopile + tower): just translate to fixed position
        t = self._vtk_transform
        t.Identity()
        t.Translate(self.east, self.north, 0.0)

        # Nacelle: yaw to face wind, then translate up + to position
        n = self._nacelle_vtk_transform
        n.Identity()
        n.RotateZ(-self._nacelle_yaw_deg)
        n.Translate(0, 0, HUB_HEIGHT)
        n.Translate(self.east, self.north, 0.0)

        # Rotor: spin + overhang + yaw + translate
        r = self._rotor_vtk_transform
        r.Identity()
        r.RotateY(self._rotor_angle_deg)
        r.Translate(0, SHAFT_OVERHANG, 0)
        r.RotateZ(-self._nacelle_yaw_deg)
        r.Translate(0, 0, HUB_HEIGHT)
        r.Translate(self.east, self.north, 0.0)

    def update_rotor_angle(self, dt: float, wind_speed: float, turbine_state: int = 0):
        """Advance rotor angle based on wind speed.

        Args:
            dt: Time step [s].
            wind_speed: Hub-height wind speed [m/s].
            turbine_state: 0=operating, 1=shutdown, 2=idling.
        """
        omega_target = rotor_speed(wind_speed, turbine_state)

        if dt > 0 and ROTOR_TIME_CONSTANT > 0:
            alpha = 1.0 - np.exp(-dt / ROTOR_TIME_CONSTANT)
            self._omega += alpha * (omega_target - self._omega)
        else:
            self._omega = omega_target

        self._rotor_angle_deg += np.degrees(self._omega * dt)
        self._rotor_angle_deg %= 360.0
