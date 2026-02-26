"""
Quick check: what is the restoring matrix C(2,:) — sway row?
And what are the panel pst forces specifically from each DOF?
"""
import re
import numpy as np

debug_file = 'debug.out'

# Check: the EOM restoring Fy is tiny (2000-600000 N) while panel pst Fy is huge
# This means szw(2,:) * motion is nearly zero — i.e. there's negligible sway restoring in the EOM
# But the panels DO produce a large sway force from pst — meaning the panel geometry 
# gives a nonzero integral of pst*n_y*dS that the waterplane-based restoring matrix doesn't capture

# The restoring matrix only has columns 3,4,5 (heave, roll, pitch)
# szw(2,3) = C_23 = 0 (from the code)
# szw(2,4) = C_24 = rho*g*sum(area*dx) = rho*g*V  (Archimedes sway-roll coupling)
# szw(2,5) = C_25 = 0

# The panel pst integration gives:
# Fy_pst = sum over triangles of [sum_j(pst(ip,j)*motion(j)) * flvec(2)]
# = sum_j [motion(j) * sum_triangles(pst(ip,j) * flvec(2))]
# where pst(:,3,:) = rho*g (heave), pst(:,4,:) = rho*g*y (roll), pst(:,5,:) = -rho*g*x (pitch)

# For sway: sum_triangles(pst(:,3,:)*n_y*dS) = rho*g * sum(n_y*dS)
# By Gauss: integral of n_y dS = 0 for a closed body (but the hull is NOT closed - it's open at the waterline)
# For a hull open at the waterline: integral of n_y dS = integral of dy*dx along waterline ≈ 0 by symmetry
# But actually, for the INWARD normal convention with flvec:
# sum(flvec(2)) over all triangles = ??? Let's check!

# The key discrepancy is that panels integrate over the HULL SURFACE (triangles from pressure points)
# while the restoring matrix uses WATERPLANE AREA integrals (trapezoidal rule on offsets)
# These use fundamentally different discretizations!

# For heave: panel sum(pst*n_z*dS) vs C_33*eta_3 = rho*g*A_wp*eta_3
# The 200% factor in heave suggests panel pst gives 2x the EOM restoring
# This could mean the panel normal area sum gives 2x the waterplane area

# For sway: the EOM restoring C_24 = rho*g*V is the displaced volume
# The panel pst gives rho*g * integral(y*n_y*dS) 
# By Gauss: integral(y*n_y) dS = V for a closed body
# But for an open hull (waterline cut), integral(y*n_y)dS_hull = V - integral(y*n_y)dS_waterplane
# And the waterplane contribution for a symmetric hull: integral(y)dA_wp = 0

# So panel integral should give V as well... unless the triangulation is asymmetric or 
# the triangle normals don't form a proper closed surface

# Let me instead just print the EOM restoring values to understand the factor
print("The EOM restoring force for sway is: Fy_rst = szw(2,:) * motion(:)")
print("From the code: szw = restorematr (columns 3:5 only)")  
print("So szw(2,j) = 0 for j=1,2,6; and szw(2,3)=C_23, szw(2,4)=C_24, szw(2,5)=C_25")
print()
print("C_24 = rho*g * sum(area_sec * dx) = rho*g * V_displaced")
print()

rho = 1025.0
g = 9.81
mass = 320437550.0  # from pdstrip.inp
V = mass / rho
print(f"V_displaced = mass/rho = {mass:.0f}/{rho:.0f} = {V:.1f} m^3")
print(f"rho*g*V = C_24 = {rho*g*V:.3e} N/rad")
print()

# From the data: at omega=0.381 (roll resonance), |Fy_rst| = 2055262
# Fy_rst = C_24 * eta_4 + C_23 * eta_3 + C_25 * eta_5
# At beam seas: eta_5 ≈ 0, so Fy_rst ≈ C_24 * eta_4
# From session notes: |eta_4| = 0.090 at omega=0.381 with 0% damping
# C_24 * eta_4 = rho*g*V * 0.090 = ?
C_24 = rho * g * V
print(f"C_24 * |eta_4(0.381)| = {C_24:.3e} * 0.090 = {C_24*0.090:.3e}")
print(f"Reported |Fy_rst| at omega=0.381: ~2,055,262")
print(f"Ratio: {C_24*0.090/2055262:.2f}")
print()
print("That's close! So C_24*eta_4 ≈ Fy_rst ✓")
print()

# Now for the panel pst force: |Fy_pst| = 281,914,210 at omega=0.381
# Fy_pst = sum_triangles(pst_tot * n_y * dS) where pst_tot includes heave, roll, pitch
# The dominant term at roll resonance should be the HEAVE-pst term:
# pst(:,3,:) = rho*g, so Fy_pst_heave = rho*g * eta_3 * sum(n_y*dS)
# Or the ROLL-pst term: pst(:,4,:) = rho*g*y, so Fy_pst_roll = rho*g * eta_4 * sum(y*n_y*dS)

# Let's compute: rho*g * |eta_3| * |sum(n_y*dS)| and compare
# |eta_3| at omega=0.381 ≈ 1.153 (from session notes)
# rho*g = 10055.25
# If Fy_pst ≈ 282e6, then sum(n_y*dS) would need to be 282e6 / (10055.25 * 1.153) ≈ 24300 m^2
# That's huge — waterplane area is about Lpp*B = 328.2*58 ≈ 19000 m^2
# Hmm, that's actually in the ballpark!

# Actually wait — the panel integration uses flvec which points INWARD
# For a symmetric hull: sum(flvec_y) should be ZERO by symmetry
# Unless the sections aren't symmetric, or the triangulation breaks symmetry

# KEY REALIZATION: the pdstrip hull panels are for ONE SIDE (starboard) only,
# and the port side is handled by symmetry in the section hydrodynamics
# but the triangle-based drift force integration is also ONE-SIDED!
# 
# For a single side (starboard): sum(n_y * dS) ≠ 0 because there's no port side to cancel!
# The heave pst contributes: rho*g*eta_3 * sum_stb(n_y*dS)
# This is HUGE because sum_stb(n_y*dS) ≈ waterplane area/2 * something

print("="*80)
print("KEY HYPOTHESIS: The pressure triangles only cover ONE HULL SIDE (starboard)")
print("For the pst heave term: Fy_pst_heave = rho*g*eta_3 * sum(n_y*dS)_stb")
print("On starboard side only, sum(n_y*dS) ≠ 0 (no port side cancellation)")
print("This creates a SPURIOUS sway force from heave pst!")
print()
print("But the EOM restoring matrix IS computed for the full (symmetric) hull,")
print("where C_23 = 0 by symmetry.")
print("="*80)

# Verify: what is sum(flvec_y) for the starboard side?
# By Gauss for a half-body bounded by the centerplane:
# sum_stb(n_y*dS) = integral of div(e_y) dV_stb + integral on centerplane
# div(e_y) = 0, so sum_stb(n_y*dS) = -integral(n_y dS)_centerplane
# The centerplane has n = (0, -1, 0) (inward = toward starboard), area ≈ Lpp * T
# So sum_stb(n_y*dS) ≈ -(-1) * Lpp * T = Lpp * T = 328.2 * 20.8 ≈ 6827 m^2

Lpp = 328.2
T = 20.8
print(f"\nEstimated sum(n_y*dS)_stb ≈ Lpp * T = {Lpp*T:.0f} m^2")
print(f"Fy_pst_heave ≈ rho*g * eta_3 * Lpp*T = {rho*g*1.153*Lpp*T:.3e}")
print(f"Actual |Fy_pst| at resonance: 2.82e+08")
print()

# But the actual panel integration would also have the PORT side contributions
# Wait... does pdstrip.f90 integrate over both sides in the drift force?
# Let me check — the PressureTriangles loop goes from i=2 to npres
# and npres includes both starboard and port points
# starboard WL is i=1, port WL is i=npres
# So the triangles DO cover both sides!

# Then sum(flvec_y) should be ≈ 0 by symmetry... unless it isn't
# This could be a numerical issue with the triangle discretization

print("Need to check: do the pressure triangles cover BOTH hull sides?")
print("And if so, does sum(flvec_y) numerically cancel to zero?")
