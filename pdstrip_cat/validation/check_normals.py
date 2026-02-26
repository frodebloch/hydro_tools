#!/usr/bin/env python3
"""
Verify Capytaine normal direction convention by checking panel normals
on a simple body.
"""
import numpy as np
import capytaine as cpt
import logging

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0
mesh_res = (10, 40, 50)

mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0), resolution=mesh_res, name="hull")
hull_mesh = mesh_full.immersed_part()

centers = hull_mesh.faces_centers
normals = hull_mesh.faces_normals

# Find panels near the bottom (z ≈ -1, y ≈ 0)
bottom_mask = (centers[:, 2] < -0.9) & (np.abs(centers[:, 1]) < 0.2) & (np.abs(centers[:, 0]) < 1)
print("Bottom panels (z < -0.9, |y| < 0.2):")
for i in np.where(bottom_mask)[0][:5]:
    print(f"  center=({centers[i,0]:6.3f}, {centers[i,1]:6.3f}, {centers[i,2]:6.3f}), "
          f"normal=({normals[i,0]:6.3f}, {normals[i,1]:6.3f}, {normals[i,2]:6.3f})")

# Find panels on the curved side (at y > 0, z ≈ -0.5)
side_mask = (centers[:, 1] > 0.5) & (np.abs(centers[:, 2] + 0.5) < 0.3) & (np.abs(centers[:, 0]) < 1)
print("\nSide panels (y > 0.5, z ≈ -0.5):")
for i in np.where(side_mask)[0][:5]:
    # Expected: normal should point in +y direction (outward into fluid)
    r_vec = np.array([0, centers[i,1], centers[i,2]])  # from center axis
    r_norm = r_vec / np.linalg.norm(r_vec)
    dot = np.dot(normals[i], r_norm)
    print(f"  center=({centers[i,0]:6.3f}, {centers[i,1]:6.3f}, {centers[i,2]:6.3f}), "
          f"normal=({normals[i,0]:6.3f}, {normals[i,1]:6.3f}, {normals[i,2]:6.3f}), "
          f"dot(n, r̂)={dot:6.3f}")

# Find panels near the waterline (z ≈ 0)
wl_mask = (np.abs(centers[:, 2]) < 0.1) & (np.abs(centers[:, 0]) < 1) & (centers[:, 1] > 0.5)
print("\nWaterline panels (z ≈ 0, y > 0.5):")
for i in np.where(wl_mask)[0][:5]:
    print(f"  center=({centers[i,0]:6.3f}, {centers[i,1]:6.3f}, {centers[i,2]:6.3f}), "
          f"normal=({normals[i,0]:6.3f}, {normals[i,1]:6.3f}, {normals[i,2]:6.3f})")

# Find a panel on the flat keel (z bottom, y near center)
keel_mask = (centers[:, 2] < -0.95) & (np.abs(centers[:, 1]) < 0.1) & (np.abs(centers[:, 0]) < 1)
print("\nKeel panels (z < -0.95, |y| < 0.1):")
for i in np.where(keel_mask)[0][:5]:
    print(f"  center=({centers[i,0]:6.3f}, {centers[i,1]:6.3f}, {centers[i,2]:6.3f}), "
          f"normal=({normals[i,0]:6.3f}, {normals[i,1]:6.3f}, {normals[i,2]:6.3f})")

# Now check the lid panels
lid = hull_mesh.generate_lid(z=-0.01)
lid_centers = lid.faces_centers
lid_normals = lid.faces_normals
print("\nLid panels (z = -0.01):")
for i in range(min(5, lid.nb_faces)):
    print(f"  center=({lid_centers[i,0]:6.3f}, {lid_centers[i,1]:6.3f}, {lid_centers[i,2]:6.3f}), "
          f"normal=({lid_normals[i,0]:6.3f}, {lid_normals[i,1]:6.3f}, {lid_normals[i,2]:6.3f})")

# Check: for a body submerged in fluid, the convention should be:
# Integral of n dS over a closed surface = 0
# For our hull + lid, check ∫ n dS
hull_areas = hull_mesh.faces_areas
integral = np.sum(normals * hull_areas[:, None], axis=0)
print(f"\n∫ n dS (hull only): ({integral[0]:.4f}, {integral[1]:.4f}, {integral[2]:.4f})")

lid_areas = lid.faces_areas
integral_lid = np.sum(lid_normals * lid_areas[:, None], axis=0)
print(f"∫ n dS (lid only):  ({integral_lid[0]:.4f}, {integral_lid[1]:.4f}, {integral_lid[2]:.4f})")

integral_total = integral + integral_lid
print(f"∫ n dS (hull+lid):  ({integral_total[0]:.4f}, {integral_total[1]:.4f}, {integral_total[2]:.4f})")

# For a closed surface with normals pointing outward: ∫ n dS = 0
# But our surface is NOT closed — it's missing the waterplane area (the deck)
# The waterplane area has normal in +z, area = 2R × L = 40 m²
# If hull+lid normals point INTO fluid (outward from body), then
# ∫ n dS (hull+lid) + (0, 0, 1) × 2R×L should ≈ 0
waterplane_integral = np.array([0, 0, 2*R*L])
closed_check = integral_total + waterplane_integral
print(f"∫ n dS (hull+lid+waterplane): ({closed_check[0]:.4f}, {closed_check[1]:.4f}, {closed_check[2]:.4f})")
print("(Should be ≈ 0 if normals point INTO fluid)")

minus_check = integral_total - waterplane_integral
print(f"∫ n dS (hull+lid-waterplane): ({minus_check[0]:.4f}, {minus_check[1]:.4f}, {minus_check[2]:.4f})")
print("(Should be ≈ 0 if normals point INTO body)")

print("\nDone.")
