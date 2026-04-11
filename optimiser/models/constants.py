"""Constants, file paths, and vessel data for the voyage comparison tool.

All module-level constants that were previously in voyage_comparison.py
are collected here to avoid circular imports and provide a single source
of truth.
"""

from pathlib import Path

import numpy as np

# ============================================================
# File paths
# ============================================================

DATA_PATH_C440 = "/home/blofro/src/prop_model/c4_40.dat"
DATA_PATH_C455 = "/home/blofro/src/prop_model/C4_55.dat"
DATA_PATH_C470 = "/home/blofro/src/prop_model/c4_70.dat"

PDSTRIP_DAT = ("/home/blofro/src/brucon/build/bin/"
               "vessel_simulator_config/propulsion_optimiser_pdstrip.dat")

NORA3_DATA_DIR = Path(__file__).parent.parent / "data" / "nora3_route"


# ============================================================
# Vessel 206 propeller/engine parameters
# ============================================================

PROP_DIAMETER = 4.80        # m
PROP_BAR = 0.432            # blade area ratio
PROP_DESIGN_PITCH = 0.771   # P/D

GEAR_RATIO = 800.0 / 117.6  # engine rpm / shaft rpm = 6.803
SHAFT_EFF = 0.97             # shaft line efficiency

KN_TO_MS = 0.5144           # knots to m/s
G = 9.81                    # m/s^2
RHO_WATER = 1025.0          # kg/m^3
RHO_AIR = 1.225             # kg/m^3
NU_AIR = 1.5e-5             # kinematic viscosity of air [m^2/s]
GENSET_SFOC = 215.0         # auxiliary genset SFOC [g/kWh] for rotor motor power


# ============================================================
# Hull data from model test (vessel 206, test 25-0461/25-0288)
# ============================================================
# Source: Trial prediction and hull efficiency elements from model test
# report dated 02.06.2025.  Draught 7.600 m, displacement 11165 m3.
#
# HULL_RESISTANCE_KN is the calm-water total resistance RT [kN] from the
# trial prediction (clean smooth hull, deep calm water, no current).
# HULL_WAKE and HULL_T_DEDUCTION are the wake fraction (WFT) and thrust
# deduction factor (THDF) from the self-propulsion test.
#
# The equilibrium equation is:
#   T_prop * (1 - t) = R_calm + R_aw + R_wind + R_roughness - F_flettner
# so:
#   T_prop = (R_calm + R_added) / (1 - t)

HULL_SPEEDS_KN = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
                           12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])

# Calm-water resistance [kN] from trial prediction (RT column)
HULL_RESISTANCE_KN = np.array([70, 79, 89, 99, 110, 121, 133, 146,
                               160, 175, 192, 208, 224, 243, 265])

# Wake fraction from self-propulsion test (WFT column)
HULL_WAKE = np.array([.223, .224, .226, .227, .227, .228, .229, .229,
                      .229, .229, .229, .231, .233, .234, .234])

# Thrust deduction factor from self-propulsion test (THDF column)
HULL_T_DEDUCTION = np.array([.174, .173, .173, .172, .171, .170, .169, .169,
                             .168, .171, .175, .171, .168, .168, .171])

# Backward-compatible alias: calm-water propeller thrust [kN].
# T_calm = R_calm / (1 - t)
HULL_THRUST_CALM_KN = HULL_RESISTANCE_KN / (1.0 - HULL_T_DEDUCTION)


# ============================================================
# Hull geometry for roughness penalty calculation
# ============================================================

HULL_S_WET = 3153.0         # wetted surface area with appendages [m^2]
HULL_LWL = 96.0             # waterline length [m] (approx from Loa 100m)
NU_WATER = 1.19e-6          # kinematic viscosity of seawater at 10°C [m^2/s]
PROP_N_BLADES = 4           # number of propeller blades


# ============================================================
# Blendermann wind resistance data
# ============================================================

# Vessel wind area parameters.
# These are for a general cargo / multi-purpose vessel of ~5000 DWT class,
# approximately 100m Loa.  Values based on typical general arrangement:
#   Frontal (transverse) projected wind area: superstructure + hull freeboard
#   Lateral projected wind area: full profile view above waterline
WIND_AREA_FRONTAL_M2 = 280.0   # A_F [m^2]
WIND_AREA_LATERAL_M2 = 1100.0  # A_L [m^2]
VESSEL_LOA_M = 100.0           # Overall length [m] (for yaw moment, not used here)

# Blendermann (1994) wind force coefficients for a general cargo vessel.
# Tabulated CX(alpha) and CY(alpha) where alpha = apparent wind angle
# measured from the bow (0 = head wind, 90 = beam, 180 = following).
#
# CX is the surge force coefficient (positive = driving force aft->fwd)
#   - Referenced to FRONTAL area A_F
#   - F_x = CX * 0.5 * rho_air * V_app^2 * A_F
#
# CY is the sway force coefficient (positive = force to starboard)
#   - Referenced to LATERAL area A_L
#   - F_y = CY * 0.5 * rho_air * V_app^2 * A_L
#
# We only need CX (surge) for propulsion resistance.  CY included for
# completeness but not used in thrust demand.
#
# Source: Blendermann, W. (1994). "Parameter identification of wind loads
# on ships." J. Wind Eng. Ind. Aerodyn., 51(3), 339-351.
# Vessel type 7: "General cargo / multi-purpose" (Table 3)
#
# The coefficients are symmetric about 0/180 (port/starboard symmetry).
# Tabulated at 10-degree intervals, 0-180.

BLEND_ANGLES = np.array([
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 110, 120, 130, 140, 150, 160, 170, 180
])

# CX: negative = resistance (opposing forward motion), positive = drive
# Head wind CX ~ -0.70, beam CX ~ 0, following CX ~ +0.40
BLEND_CX = np.array([
    -0.70, -0.72, -0.72, -0.68, -0.55, -0.38, -0.18, -0.02, 0.10, 0.15,
     0.18,  0.22,  0.25,  0.30,  0.32,  0.34,  0.38,  0.40, 0.38
])

# CY: zero at 0/180, peak ~+0.70 at beam (referenced to A_L)
BLEND_CY = np.array([
    0.00, 0.12, 0.25, 0.40, 0.54, 0.65, 0.72, 0.74, 0.72, 0.66,
    0.58, 0.48, 0.37, 0.26, 0.16, 0.08, 0.03, 0.00, 0.00
])
