"""
Vessel hydrodynamic model.

Simplified 3DOF (surge, sway, yaw) vessel model based on the Brix/Faltinsen
strip-theory approach used in the brucon vessel simulator. Hull geometry is
generated from main particulars using a simplified section generator.

Coordinate convention:
  - NED frame for global positions
  - Body frame: x forward (surge), y starboard (sway), yaw clockwise
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class VesselConfig:
    """Main vessel parameters."""
    name: str = "Cable Layer"
    lpp: float = 130.0          # Length between perpendiculars [m]
    beam: float = 23.0          # Breadth [m]
    draft: float = 7.8          # Design draft [m]
    Cb: float = 0.72            # Block coefficient [-]
    Cm: float = 0.95            # Midship section coefficient [-]

    # Derived / optional overrides
    displacement: float = None  # [kg], computed if None
    lcg: float = 0.0            # Longitudinal CG from midship [m], positive forward

    # Radii of gyration (fractions of lpp / beam)
    kxx: float = 0.35           # Roll radius of gyration / beam
    kyy: float = 0.25           # Pitch radius of gyration / lpp
    kzz: float = 0.25           # Yaw radius of gyration / lpp

    # Wind areas (approximate for cable layer)
    lateral_wind_area: float = 2500.0   # [m^2]
    frontal_wind_area: float = 600.0    # [m^2]
    wind_area_centre: float = 0.0       # from midship [m], positive forward

    # Tow point location from midship [m], positive forward
    tow_point_x: float = -60.0   # Stern tow point
    tow_point_y: float = 0.0     # Centreline

    # Thruster capacity
    max_thrust_surge: float = 900e3   # [N] ~ 90t bollard pull
    max_thrust_sway: float = 500e3    # [N]
    max_thrust_yaw: float = 20e6     # [Nm]

    # Thrust rate limits [N/s] or [Nm/s]
    thrust_rate_surge: float = 100e3
    thrust_rate_sway: float = 80e3
    thrust_rate_yaw: float = 5e6

    def __post_init__(self):
        rho = 1025.0
        if self.displacement is None:
            self.displacement = rho * self.lpp * self.beam * self.draft * self.Cb


RHO_WATER = 1025.0
G = 9.81


class VesselModel:
    """
    3DOF vessel hydrodynamic model (surge, sway, yaw).

    Based on simplified strip-theory coefficients following the Brix unified
    model approach, similar to the brucon VesselModel implementation.
    """

    def __init__(self, config: VesselConfig):
        self.cfg = config
        self._compute_hydro_coefficients()

    def _compute_hydro_coefficients(self):
        """Compute added mass, damping, and drag coefficients from hull geometry."""
        c = self.cfg
        L = c.lpp
        B = c.beam
        T = c.draft
        M = c.displacement
        rho = RHO_WATER

        # Displacement volume
        nabla = M / rho

        # --- Added mass (Brix / Faltinsen) ---
        # Surge added mass (Brix 4.27)
        self.A11 = M / max(np.pi * np.sqrt(L**3 / nabla) - 14, 1.0)

        # 3D correction factors (Faltinsen)
        eps = 2.0 * T / L
        k1 = np.sqrt(max(1.0 - 0.245 * eps - 1.68 * eps**2, 0.1))
        k2 = np.sqrt(max(1.0 - 0.76 * eps - 4.41 * eps**2, 0.1))

        # Section added mass (Lewis form, simplified)
        # Using average section: a22_0 ~ 0.5 * pi * rho * T^2 * Cy
        # For typical ship sections, Cy ~ 0.8-1.2
        Cy = 0.9 * c.Cm
        a22_per_length = 0.5 * np.pi * rho * T**2 * Cy

        # Sway added mass: A22 = k1^2 * integral(a22, dx) over length
        self.A22 = k1**2 * a22_per_length * L

        # Sway-yaw coupling: A26 = k1*k2 * integral(x*a22, dx)
        # For symmetric hull, this is small; asymmetry from entrance/run shape
        Le_ratio = 0.4  # approximate entrance fraction
        self.A26 = k1 * k2 * a22_per_length * L**2 * (0.5 - Le_ratio) * 0.1

        # Yaw added mass: A66 = k2^2 * integral(x^2 * a22, dx)
        self.A66 = k2**2 * a22_per_length * L**3 / 12.0

        # --- Mass matrix (3DOF) ---
        self.M_surge = M + self.A11
        self.M_sway = M + self.A22
        self.M_sway_yaw = M * c.lcg + self.A26  # coupling
        Izz = M * (c.kzz * L)**2
        self.M_yaw = Izz + self.A66

        # --- Damping ---
        # Linear sway damping at low speed (Brix unified model)
        self.B22 = a22_per_length * 0.5 * L * 0.05  # small fraction of added mass
        self.B26 = 0.0  # simplified
        self.B62 = 0.0
        self.B66 = self.A66 * 0.02  # small linear yaw damping

        # Nonlinear transverse drag coefficient
        self.Cd_transverse = 0.7
        self.T_avg = T  # average section draft for drag

        # --- Surge resistance (ITTC 1957) ---
        # Wetted surface (Denny-Mumford)
        self.S_wetted = 1.7 * L * T + nabla / T

        # Form factor
        self.form_factor = 1.3

    def resistance_surge(self, u: float) -> float:
        """Hull resistance in surge [N]. u is speed through water."""
        if abs(u) < 1e-6:
            return 0.0
        Re = abs(u) * self.cfg.lpp / 1.19e-6  # kinematic viscosity seawater
        Re = max(Re, 1e3)
        Cf = 0.075 / (np.log10(Re) - 2.0)**2
        return -self.form_factor * 0.5 * RHO_WATER * u * abs(u) * Cf * self.S_wetted

    def transverse_drag(self, v: float, r: float) -> tuple:
        """
        Nonlinear cross-flow drag force and moment.
        Returns (Y_drag, N_drag) in body frame.
        """
        L = self.cfg.lpp
        T = self.T_avg
        n_sections = 21
        dx = L / (n_sections - 1)
        Y_drag = 0.0
        N_drag = 0.0
        for i in range(n_sections):
            x = -L / 2 + i * dx  # position from midship
            vx = v + r * x  # local transverse velocity
            dF = -0.5 * RHO_WATER * vx * abs(vx) * self.Cd_transverse * T * dx
            Y_drag += dF
            N_drag += dF * x
        return Y_drag, N_drag

    def forces(self, u: float, v: float, r: float,
               u_dot: float, v_dot: float, r_dot: float) -> tuple:
        """
        Compute total hydrodynamic forces in body frame.

        Parameters:
            u, v, r: surge/sway velocities [m/s], yaw rate [rad/s]
            u_dot, v_dot, r_dot: accelerations

        Returns:
            (X, Y, N): surge force, sway force, yaw moment [N, N, Nm]
        """
        # Added mass forces (reactive)
        X_am = -self.A11 * u_dot
        Y_am = -self.A22 * v_dot - self.A26 * r_dot
        N_am = -self.A26 * v_dot - self.A66 * r_dot

        # Coriolis / Munk moment
        X_cor = self.A22 * v * r  # Munk: A22*v*r
        Y_cor = -self.A11 * u * r
        N_cor = (self.A11 - self.A22) * u * v  # Munk moment

        # Speed-dependent linear damping
        Y_lin = self.B22 * v + self.B26 * r
        N_lin = self.B62 * v + self.B66 * r

        # Nonlinear transverse drag
        Y_drag, N_drag = self.transverse_drag(v, r)

        # Surge resistance
        X_res = self.resistance_surge(u)

        X = X_am + X_cor + X_res
        Y = Y_am + Y_cor + Y_lin + Y_drag
        N = N_am + N_cor + N_lin + N_drag

        return X, Y, N

    def tow_point_body(self) -> np.ndarray:
        """Tow point position in body frame [x, y]."""
        return np.array([self.cfg.tow_point_x, self.cfg.tow_point_y])


class VesselSimulator:
    """
    3DOF vessel dynamics integrator.

    State: [x_north, y_east, psi, u, v, r]
    Uses RK4 integration in the NED frame.
    """

    def __init__(self, model: VesselModel, dt: float = 0.1):
        self.model = model
        self.dt = dt

        # State: position in NED + body velocities
        self.x = 0.0       # North [m]
        self.y = 0.0       # East [m]
        self.psi = 0.0     # Heading [rad], 0 = North, positive clockwise
        self.u = 0.0       # Surge velocity [m/s]
        self.v = 0.0       # Sway velocity [m/s]
        self.r = 0.0       # Yaw rate [rad/s]

        # Accelerations (for output)
        self.u_dot = 0.0
        self.v_dot = 0.0
        self.r_dot = 0.0

        # External forces in body frame [N, N, Nm]
        self.tau_external = np.array([0.0, 0.0, 0.0])

        # Thruster forces in body frame (from DP controller)
        self.tau_thruster = np.array([0.0, 0.0, 0.0])

    def set_position(self, x: float, y: float, psi: float):
        self.x = x
        self.y = y
        self.psi = psi

    def set_velocity(self, u: float, v: float, r: float):
        self.u = u
        self.v = v
        self.r = r

    def rotation_matrix(self, psi: float) -> np.ndarray:
        """2D rotation from body to NED frame."""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s],
                         [s,  c]])

    def tow_point_ned(self) -> np.ndarray:
        """Tow point position in NED coordinates."""
        tp_body = self.model.tow_point_body()
        R = self.rotation_matrix(self.psi)
        tp_ned = R @ tp_body
        return np.array([self.x + tp_ned[0], self.y + tp_ned[1]])

    def tow_point_velocity_ned(self) -> np.ndarray:
        """Tow point velocity in NED frame."""
        tp_body = self.model.tow_point_body()
        # Velocity at tow point in body frame
        u_tp = self.u - self.r * tp_body[1]
        v_tp = self.v + self.r * tp_body[0]
        R = self.rotation_matrix(self.psi)
        return R @ np.array([u_tp, v_tp])

    def step(self, tau_external: np.ndarray, tau_thruster: np.ndarray):
        """
        Advance one time step.

        Parameters:
            tau_external: External forces in body frame [Fx, Fy, Mz] [N, N, Nm]
            tau_thruster: Thruster forces in body frame [Fx, Fy, Mz] [N, N, Nm]
        """
        self.tau_external = tau_external.copy()
        self.tau_thruster = tau_thruster.copy()

        dt = self.dt
        m = self.model

        # Total external force (excluding hydrodynamic reaction forces)
        tau_total = tau_external + tau_thruster

        def compute_accelerations(u, v, r, tau):
            """Solve M * acc = tau + hydro_forces for accelerations."""
            # Hydro forces excluding added-mass terms (those go to LHS)
            _, Y_cor, N_cor = 0.0, -m.A11 * u * r, (m.A11 - m.A22) * u * v
            X_cor = m.A22 * v * r
            Y_lin = m.B22 * v + m.B26 * r
            N_lin = m.B62 * v + m.B66 * r
            Y_drag, N_drag = m.transverse_drag(v, r)
            X_res = m.resistance_surge(u)

            # RHS: external + hydrodynamic (excluding added mass)
            rhs_x = tau[0] + X_cor + X_res
            rhs_y = tau[1] + Y_cor + Y_lin + Y_drag
            rhs_n = tau[2] + N_cor + N_lin + N_drag

            # Solve mass matrix [M_surge, 0, 0; 0, M_sway, M_sy; 0, M_sy, M_yaw]
            u_dot = rhs_x / m.M_surge

            # 2x2 sway-yaw system
            det = m.M_sway * m.M_yaw - m.M_sway_yaw**2
            if abs(det) < 1e-10:
                det = 1e-10
            v_dot = (m.M_yaw * rhs_y - m.M_sway_yaw * rhs_n) / det
            r_dot = (-m.M_sway_yaw * rhs_y + m.M_sway * rhs_n) / det

            return u_dot, v_dot, r_dot

        # --- RK4 integration ---
        # State: [x, y, psi, u, v, r]
        def state_derivatives(x, y, psi, u, v, r):
            u_d, v_d, r_d = compute_accelerations(u, v, r, tau_total)
            R = self.rotation_matrix(psi)
            vel_ned = R @ np.array([u, v])
            return vel_ned[0], vel_ned[1], r, u_d, v_d, r_d

        s = (self.x, self.y, self.psi, self.u, self.v, self.r)
        k1 = state_derivatives(*s)
        s2 = tuple(si + 0.5 * dt * ki for si, ki in zip(s, k1))
        k2 = state_derivatives(*s2)
        s3 = tuple(si + 0.5 * dt * ki for si, ki in zip(s, k2))
        k3 = state_derivatives(*s3)
        s4 = tuple(si + dt * ki for si, ki in zip(s, k3))
        k4 = state_derivatives(*s4)

        for i, attr in enumerate(['x', 'y', 'psi', 'u', 'v', 'r']):
            val = getattr(self, attr) + dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            setattr(self, attr, val)

        # Wrap heading to [-pi, pi]
        self.psi = np.arctan2(np.sin(self.psi), np.cos(self.psi))

        # Store accelerations
        self.u_dot, self.v_dot, self.r_dot = compute_accelerations(
            self.u, self.v, self.r, tau_total)
