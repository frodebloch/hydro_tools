"""
Quasi-static catenary model for the tow wire.

Models the tow wire as a catenary curve between the vessel tow point and
the plough on the seabed. Includes:
  - Wire self-weight in water (submerged weight per unit length)
  - Hydrodynamic drag on the wire in current
  - Tension at both ends (vessel and plough)
  - Layback calculation (horizontal distance from vessel to plough)
  - Wire geometry (shape of the catenary)

The catenary equation for a cable with uniform weight w [N/m] in still water:

    z(s) = (T_h / w) * (cosh(w*s/T_h) - 1)

where:
    s = arc length along wire from touchdown
    T_h = horizontal tension component (constant along wire)
    w = submerged weight per unit length

For a cable from seabed to surface with water depth h:
    h = (T_h / w) * (cosh(w * x_h / T_h) - 1)

where x_h is the horizontal span.

With current drag on the wire, we use a modified catenary that accounts for
the distributed horizontal load from drag.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq


@dataclass
class TowWireConfig:
    """Tow wire properties."""
    diameter: float = 0.076         # Wire diameter [m] (76mm typical for ploughing)
    linear_mass: float = 25.0       # Mass per unit length in air [kg/m]
    steel_density: float = 7850.0   # Wire density [kg/m^3]
    young_modulus: float = 100e9    # Young's modulus [Pa] (wire rope ~100 GPa)
    fill_factor: float = 0.55       # Wire rope metallic area / gross area [-]
    breaking_load: float = 2000e3   # Minimum breaking load [N] (~200t)
    Cd_normal: float = 1.2          # Normal drag coefficient [-]
    total_length: float = 2500.0    # Total wire length available [m]

    @property
    def submerged_weight(self) -> float:
        """Submerged weight per unit length [N/m]."""
        rho_water = 1025.0
        area = np.pi * self.diameter**2 / 4.0
        buoyancy_per_m = rho_water * 9.81 * area
        weight_per_m = self.linear_mass * 9.81
        return weight_per_m - buoyancy_per_m

    @property
    def axial_stiffness(self) -> float:
        """EA [N] — uses fill_factor to account for wire rope construction."""
        area = np.pi * self.diameter**2 / 4.0
        return self.young_modulus * area * self.fill_factor


class CatenaryModel:
    """
    Quasi-static catenary model for the tow wire.

    Solves the catenary geometry given:
      - Water depth at vessel (tow point draft is small, ignored or add offset)
      - Horizontal tension at the plough end
      - Wire properties (submerged weight, drag)
      - Current velocity (for drag modification)

    The model works in 2D (vertical plane containing the wire).
    For 3D operations, the catenary plane rotates to align with the
    horizontal direction from vessel tow point to plough.
    """

    def __init__(self, wire_config: TowWireConfig):
        self.wire = wire_config

    def solve_catenary(self, water_depth: float, horizontal_force: float,
                       current_speed: float = 0.0, tow_point_depth: float = 5.0,
                       n_points: int = 100) -> dict:
        """
        Solve the catenary for given horizontal force at plough.

        The plough applies a horizontal force through the wire. This sets
        the horizontal tension T_h. We solve for the wire geometry that
        connects the plough (on seabed) to the vessel tow point.

        Parameters:
            water_depth: Water depth at plough location [m]
            horizontal_force: Horizontal force at plough end [N]
            current_speed: Inline current speed [m/s] (positive = same direction as tow)
            tow_point_depth: Depth of tow point below surface [m]
            n_points: Number of points for wire geometry

        Returns:
            dict with keys:
                'tension_vessel': Tension at vessel tow point [N]
                'tension_plough': Tension at plough end [N]
                'angle_vessel': Wire angle from horizontal at vessel [rad]
                'angle_plough': Wire angle from horizontal at plough [rad]
                'horizontal_force': Horizontal component at vessel [N]
                'vertical_force': Vertical component at vessel [N]
                'layback': Horizontal distance from vessel to plough [m]
                'wire_length': Suspended wire length [m]
                'wire_x': Horizontal coordinates of wire [m] (0 at plough touchdown)
                'wire_z': Vertical coordinates of wire [m] (0 at seabed)
                'grounded_length': Wire length on seabed [m]
                'safety_factor': Breaking load / max tension [-]
        """
        w = self.wire.submerged_weight  # N/m
        T_h = max(horizontal_force, 100.0)  # Minimum to avoid singularities

        # Height the wire must span
        h = water_depth - tow_point_depth
        if h < 1.0:
            h = 1.0

        # Add current drag as distributed horizontal load
        # This modifies the effective weight vector direction
        if abs(current_speed) > 0.01:
            # Drag per unit length (normal component, approximate)
            rho = 1025.0
            q_drag = 0.5 * rho * self.wire.Cd_normal * self.wire.diameter * current_speed * abs(current_speed)
            # Effective weight including drag: w_eff = sqrt(w^2 + q^2)
            w_eff = np.sqrt(w**2 + q_drag**2)
            # The catenary tilts: effective vertical = w, horizontal component from drag
            # For the catenary solution, we solve in the tilted plane
            # The horizontal tension includes the drag contribution
            T_h_eff = T_h + q_drag * h / w  # approximate correction
        else:
            w_eff = w
            q_drag = 0.0
            T_h_eff = T_h

        # --- Solve catenary geometry ---
        # Standard catenary: z(x) = (T_h/w) * (cosh(w*x/T_h) - 1)
        # At the vessel: z = h, solve for x (horizontal span)
        a = T_h_eff / w_eff  # catenary parameter

        # h = a * (cosh(x_span / a) - 1)
        # cosh(x_span / a) = 1 + h/a
        cosh_val = 1.0 + h / a
        if cosh_val < 1.0:
            cosh_val = 1.0

        x_span = a * np.arccosh(cosh_val)  # horizontal span of suspended wire

        # Suspended wire length
        # s = a * sinh(x_span / a)
        s_total = a * np.sinh(x_span / a)

        # Tension at vessel (top of catenary)
        T_vessel = T_h_eff * np.cosh(x_span / a)

        # Angle at vessel from horizontal
        angle_vessel = np.arctan2(w_eff * s_total, T_h_eff)

        # Angle at plough (bottom, where wire leaves seabed)
        # At the touchdown point, wire is tangent to horizontal
        angle_plough = 0.0  # catenary tangent at touchdown is horizontal

        # Tension at plough end
        T_plough = T_h_eff  # at touchdown, tension is purely horizontal

        # Wire on seabed (grounded)
        grounded_length = max(self.wire.total_length - s_total, 0.0)

        # Layback = horizontal span of suspended wire + grounded wire extends behind
        layback = x_span

        # Force components at vessel in wire plane
        F_horizontal = T_h_eff
        F_vertical = w_eff * s_total  # vertical component = total submerged weight of suspended wire

        # Generate wire geometry
        x_wire = np.linspace(0, x_span, n_points)
        z_wire = a * (np.cosh(x_wire / a) - 1.0)

        # Safety factor
        safety_factor = self.wire.breaking_load / T_vessel if T_vessel > 0 else float('inf')

        return {
            'tension_vessel': T_vessel,
            'tension_plough': T_plough,
            'angle_vessel': angle_vessel,
            'angle_plough': angle_plough,
            'horizontal_force': F_horizontal,
            'vertical_force': F_vertical,
            'layback': layback,
            'wire_length': s_total,
            'wire_x': x_wire,
            'wire_z': z_wire,
            'grounded_length': grounded_length,
            'safety_factor': safety_factor,
        }

    def solve_for_plough_position(self, water_depth: float,
                                   vessel_tow_point: np.ndarray,
                                   plough_position: np.ndarray,
                                   plough_depth: float = None,
                                   current_speed: float = 0.0,
                                   tow_point_depth: float = 5.0) -> dict:
        """
        Given vessel tow point and plough position, solve for wire tension.

        This is the inverse problem: given the geometry, find the tension.
        Uses the constraint that the horizontal distance must match the catenary layback.

        Parameters:
            water_depth: Water depth [m]
            vessel_tow_point: Vessel tow point [x_north, y_east] in NED [m]
            plough_position: Plough position [x_north, y_east] on seabed [m]
            plough_depth: Water depth at plough [m], defaults to water_depth
            current_speed: Inline current [m/s]
            tow_point_depth: Tow point depth below surface [m]

        Returns:
            Same dict as solve_catenary, plus:
                'wire_azimuth': Direction from vessel to plough [rad] in NED
                'force_body': Forces at vessel in body frame (needs heading)
        """
        if plough_depth is None:
            plough_depth = water_depth

        # Horizontal distance from vessel to plough
        dx = plough_position[0] - vessel_tow_point[0]
        dy = plough_position[1] - vessel_tow_point[1]
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        wire_azimuth = np.arctan2(dy, dx)

        # Height the wire spans
        h = plough_depth - tow_point_depth
        if h < 1.0:
            h = 1.0

        w = self.wire.submerged_weight

        # Solve: find T_h such that catenary layback = horizontal_distance
        # layback = (T_h/w) * arccosh(1 + w*h/T_h)
        def layback_error(T_h):
            a = T_h / w
            cosh_val = 1.0 + h / a
            if cosh_val < 1.0:
                return horizontal_distance
            x_span = a * np.arccosh(cosh_val)
            return x_span - horizontal_distance

        # Check if horizontal distance is feasible
        # Minimum layback (wire nearly vertical): approaches 0 as T_h -> 0
        # Maximum layback: limited by wire length on seabed
        # With very large T_h, wire is nearly flat: layback -> wire_length

        # For very short distances (wire nearly vertical), T_h is small
        # For large distances, T_h is large
        T_h_min = 100.0  # minimum horizontal tension
        T_h_max = self.wire.breaking_load  # can't exceed breaking load

        # Check bounds
        err_min = layback_error(T_h_min)
        err_max = layback_error(T_h_max)

        if err_min > 0:
            # Even minimum tension gives too much layback - wire is too heavy
            # or distance is too short. Use minimum tension.
            T_h = T_h_min
        elif err_max < 0:
            # Even maximum tension can't reach - wire too short or distance too far
            T_h = T_h_max
        else:
            try:
                T_h = brentq(layback_error, T_h_min, T_h_max, rtol=1e-6)
            except ValueError:
                T_h = T_h_min

        result = self.solve_catenary(plough_depth, T_h, current_speed, tow_point_depth)
        result['wire_azimuth'] = wire_azimuth
        result['horizontal_distance'] = horizontal_distance

        return result

    def vessel_force_body_frame(self, catenary_result: dict,
                                 vessel_heading: float) -> np.ndarray:
        """
        Convert catenary forces at vessel to body frame.

        Parameters:
            catenary_result: Output from solve_catenary or solve_for_plough_position
            vessel_heading: Vessel heading [rad], 0 = North

        Returns:
            [Fx_surge, Fy_sway, Mz_yaw] in body frame [N, N, Nm]
        """
        F_h = catenary_result['horizontal_force']
        F_v = catenary_result['vertical_force']
        azimuth = catenary_result.get('wire_azimuth', np.pi)  # default: wire goes aft

        # Horizontal force direction in NED
        Fx_ned = F_h * np.cos(azimuth)
        Fy_ned = F_h * np.sin(azimuth)

        # Rotate to body frame (force is pulling vessel toward plough)
        # The force on the vessel is toward the plough
        c, s = np.cos(vessel_heading), np.sin(vessel_heading)
        # NED to body rotation (inverse of body-to-NED)
        Fx_body = c * Fx_ned + s * Fy_ned
        Fy_body = -s * Fx_ned + c * Fy_ned

        # These are forces FROM the wire ON the vessel (pulling toward plough)
        # For a stern tow, Fx_body is typically negative (pulling aft)

        # Yaw moment from tow point offset
        # Tow force acts at the tow point, creating a moment about CG
        # This would need the tow point position - handled in the simulation loop

        return np.array([Fx_body, Fy_body, 0.0])

    def compute_yaw_moment(self, force_body: np.ndarray,
                           tow_point_body: np.ndarray) -> float:
        """
        Compute yaw moment from tow force acting at the tow point.

        Parameters:
            force_body: Force in body frame [Fx, Fy, Mz]
            tow_point_body: Tow point in body frame [x, y]

        Returns:
            Yaw moment [Nm]
        """
        # M = r x F (2D cross product)
        return tow_point_body[0] * force_body[1] - tow_point_body[1] * force_body[0]
