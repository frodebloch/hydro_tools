#!/usr/bin/env python3
"""
Compare waterline pressure/elevation between pdstrip and Capytaine
for beam seas at lambda=3m to understand the WL integral discrepancy.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential
import logging

cpt.set_logging(logging.WARNING)

R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

# Target: beam seas, lambda=3m
lam = 3.0
k = 2 * np.pi / lam
omega = np.sqrt(k * g)
beta = np.pi / 2  # beam seas from starboard

print(f"omega = {omega:.4f}, k = {k:.4f}, lambda = {lam:.1f}")

# --- Capytaine mesh and solve ---
mesh_res = (10, 40, 50)
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=mesh_res, name="hull"
)
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")

volume = np.pi * R**2 / 2 * L
mass_val = rho * volume
zcg = -4 * R / (3 * np.pi)
body.center_of_mass = np.array([0.0, 0.0, zcg])
body.mass = mass_val
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()
dof_names = list(body.dofs.keys())

# Radiation
rad_results = {}
for dof in dof_names:
    prob = cpt.RadiationProblem(body=body, radiating_dof=dof, omega=omega,
                                 water_depth=np.inf, rho=rho, g=g)
    rad_results[dof] = solver.solve(prob)

# Added mass, damping, stiffness
n_dof = len(dof_names)
A = np.zeros((n_dof, n_dof))
B_mat = np.zeros((n_dof, n_dof))
for i, rdof in enumerate(dof_names):
    for j, idof in enumerate(dof_names):
        A[i, j] = rad_results[rdof].added_masses[idof]
        B_mat[i, j] = rad_results[rdof].radiation_dampings[idof]

M = np.zeros((n_dof, n_dof))
for i, dof in enumerate(dof_names):
    if dof in ('Surge', 'Sway', 'Heave'):
        M[i, i] = mass_val
    elif dof == 'Roll':
        M[i, i] = mass_val * R**2 / 4
    elif dof == 'Pitch':
        M[i, i] = mass_val * (L**2/12 + R**2/4)
    elif dof == 'Yaw':
        M[i, i] = mass_val * (L**2/12 + R**2/4)

stiffness_xr = body.compute_hydrostatic_stiffness()
C = np.zeros((n_dof, n_dof))
for i, idof in enumerate(dof_names):
    for j, jdof in enumerate(dof_names):
        try:
            C[i, j] = float(stiffness_xr.sel(influenced_dof=idof, radiating_dof=jdof))
        except:
            C[i, j] = 0.0

# Diffraction
diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta, omega=omega,
                                    water_depth=np.inf, rho=rho, g=g)
diff_result = solver.solve(diff_prob)

# Excitation and RAOs
F_exc = np.array([diff_result.forces[dof] for dof in dof_names])
Z = -omega**2 * (M + A) + 1j * omega * B_mat + C
xi = np.linalg.solve(Z, F_exc)

print(f"\nRAOs:")
for i, dof in enumerate(dof_names):
    print(f"  {dof}: |xi|={np.abs(xi[i]):.6f}, phase={np.degrees(np.angle(xi[i])):.1f}°")

xi_surge, xi_sway, xi_heave = xi[0], xi[1], xi[2]
xi_roll, xi_pitch, xi_yaw = xi[3], xi[4], xi[5]

# Evaluate potential at waterline points along the port and starboard rails
# At x=0 (midship), y=±1, z slightly below 0
n_x_pts = 50
x_pts = np.linspace(-L/2, L/2, n_x_pts)

# Port rail (y = -1.0 in Capytaine convention = port)
port_pts = np.column_stack([x_pts, -np.ones(n_x_pts), -0.001 * np.ones(n_x_pts)])
# Starboard rail (y = +1.0 in Capytaine convention = starboard)
stbd_pts = np.column_stack([x_pts, np.ones(n_x_pts), -0.001 * np.ones(n_x_pts)])

# Compute total potential at these points
# Incident
inc_pot_port = airy_waves_potential(port_pts, diff_prob)
inc_pot_stbd = airy_waves_potential(stbd_pts, diff_prob)

# Diffraction
diff_pot_port = solver.compute_potential(port_pts, diff_result)
diff_pot_stbd = solver.compute_potential(stbd_pts, diff_result)

# Radiation
rad_pot_port = np.zeros(n_x_pts, dtype=complex)
rad_pot_stbd = np.zeros(n_x_pts, dtype=complex)
for i, dof in enumerate(dof_names):
    rp_port = solver.compute_potential(port_pts, rad_results[dof])
    rp_stbd = solver.compute_potential(stbd_pts, rad_results[dof])
    rad_pot_port += xi[i] * rp_port
    rad_pot_stbd += xi[i] * rp_stbd

# Total potential
tot_pot_port = inc_pot_port + diff_pot_port + rad_pot_port
tot_pot_stbd = inc_pot_stbd + diff_pot_stbd + rad_pot_stbd

# Wave elevation: eta = (iw/g) * phi  (Capytaine convention)
eta_port = (1j * omega / g) * tot_pot_port
eta_stbd = (1j * omega / g) * tot_pot_stbd

# Relative wave elevation: eta_rel = eta - z_body_at_wl
z_body_port = xi_heave + xi_roll * (-1.0) - xi_pitch * x_pts
z_body_stbd = xi_heave + xi_roll * (1.0) - xi_pitch * x_pts
eta_rel_port = eta_port - z_body_port
eta_rel_stbd = eta_stbd - z_body_stbd

# Capytaine convention pressure: p = iωρφ
p_total_port = 1j * omega * rho * tot_pot_port
p_total_stbd = 1j * omega * rho * tot_pot_stbd

print(f"\n--- Waterline elevations at x=0 (midship) ---")
mid = n_x_pts // 2
print(f"Port (y=-1):  |eta|={np.abs(eta_port[mid]):.4f}, |eta_rel|={np.abs(eta_rel_port[mid]):.4f}, |p|={np.abs(p_total_port[mid]):.1f}")
print(f"Stbd (y=+1):  |eta|={np.abs(eta_stbd[mid]):.4f}, |eta_rel|={np.abs(eta_rel_stbd[mid]):.4f}, |p|={np.abs(p_total_stbd[mid]):.1f}")

# Compare with pdstrip
# pdstrip: pres at index 1 (y=-1, PORT in pdstrip z-down coords) = 1744.79
# pdstrip: pres at index npres (y=+1, STBD) = 20557.4
# Note: pdstrip uses exp(+iωt), p = -iωρφ = ρg*η
# So |p_pd| = ρg*|η|. But it includes hydrostatic correction, so |p_pd| = ρg*|η_rel|
print(f"\npdstrip waterline pressures (from debug.out):")
print(f"  |p_stb| (index 1, y=-1 PORT) = 1744.79   -> |eta_rel| = {1744.79/(rho*g):.4f}")
print(f"  |p_port| (index npres, y=+1 STBD) = 20557.4  -> |eta_rel| = {20557.4/(rho*g):.4f}")

print(f"\nCapytaine waterline (at x=0):")
print(f"  Port (y=-1): |eta_rel| = {np.abs(eta_rel_port[mid]):.4f}")
print(f"  Stbd (y=+1): |eta_rel| = {np.abs(eta_rel_stbd[mid]):.4f}")
print(f"  Port (y=-1): |eta|     = {np.abs(eta_port[mid]):.4f}")
print(f"  Stbd (y=+1): |eta|     = {np.abs(eta_stbd[mid]):.4f}")

# Now compute the WL integral along the ship length
# Using capytaine convention: n_y at port (y=-1) is negative (outward=away from body)
# and n_y at starboard (y=+1) is positive (outward=away from body)
# F_WL_y = 0.25 * rho * g * integral of |eta_rel|^2 * n_y * dl

# For a straight waterline edge: n_y = -1 (port) and +1 (stbd), dl = dx
dx = x_pts[1] - x_pts[0]

F_wl_y_port = 0.25 * rho * g * np.sum(np.abs(eta_rel_port)**2 * (-1.0)) * dx
F_wl_y_stbd = 0.25 * rho * g * np.sum(np.abs(eta_rel_stbd)**2 * (+1.0)) * dx
F_wl_y_total = F_wl_y_port + F_wl_y_stbd

# Same using absolute elevation (not relative)
F_wl_y_port_abs = 0.25 * rho * g * np.sum(np.abs(eta_port)**2 * (-1.0)) * dx
F_wl_y_stbd_abs = 0.25 * rho * g * np.sum(np.abs(eta_stbd)**2 * (+1.0)) * dx
F_wl_y_total_abs = F_wl_y_port_abs + F_wl_y_stbd_abs

print(f"\n--- WL integral comparison ---")
print(f"Capytaine WL Fy (from main script):    {-288572.7:.1f}")
print(f"Capytaine WL Fy (reconstructed, rel):  {F_wl_y_total:.1f}")
print(f"Capytaine WL Fy (reconstructed, abs):  {F_wl_y_total_abs:.1f}")
print(f"  Port contrib (rel):  {F_wl_y_port:.1f}")
print(f"  Stbd contrib (rel):  {F_wl_y_stbd:.1f}")
print(f"  Port contrib (abs):  {F_wl_y_port_abs:.1f}")
print(f"  Stbd contrib (abs):  {F_wl_y_stbd_abs:.1f}")

# pdstrip WL integral for comparison
# pdstrip: dfeta_per_section = 0.25*dx*(|p_stb|^2 - |p_port|^2)/(rho*g)
# In Capytaine Fy convention: pd_WL_Fy = -sum(dfeta)
# For 5 sections: 2*half + 3*full = 2*0.5*5 + 3*5 = 5 + 15 = 20m total
# Pressure is constant along length, so:
p_stb_pd = 1744.79  # at index 1 (y=-1, PORT in geometry)
p_port_pd = 20557.4  # at index npres (y=+1, STBD in geometry)
# dfeta uses (|p_stb|^2 - |p_port|^2) which is (|p(y=-1)|^2 - |p(y=+1)|^2)
# Since in pdstrip coords, y=-1 is port waterline, y=+1 is stbd waterline
# And the sway force convention: feta positive = starboard
# feta_WL = sum_sections of 0.25*dx*(|p(y=-1)|^2 - |p(y=+1)|^2)/(rho*g)
# Fy_capytaine = -feta_pdstrip

pd_WL_feta = 0.25 * L * (p_stb_pd**2 - p_port_pd**2) / (rho * g)
pd_WL_Fy = -pd_WL_feta

print(f"\npdstrip WL Fy = -feta_WL = {pd_WL_Fy:.1f}")
print(f"  |p(y=-1)|/rho/g = |eta_rel(port)|_pd = {p_stb_pd/(rho*g):.4f}")
print(f"  |p(y=+1)|/rho/g = |eta_rel(stbd)|_pd = {p_port_pd/(rho*g):.4f}")

# Show the variation along the ship length
print(f"\n--- |eta_rel| along ship length (Capytaine 3D) ---")
for i in range(0, n_x_pts, 5):
    print(f"  x={x_pts[i]:6.1f}: port |eta_rel|={np.abs(eta_rel_port[i]):.4f}  stbd |eta_rel|={np.abs(eta_rel_stbd[i]):.4f}")
