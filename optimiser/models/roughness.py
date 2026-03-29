"""Hull and propeller roughness models.

Includes:
- FoulingScenario: piecewise-linear fouling roughness growth
- hull_roughness_delta_R_kN: Townsin-Dey (2003) hull roughness penalty
- propeller_roughness_fuel_factor: blade roughness fuel multiplier
- Pre-defined fouling scenarios (FOULING_LOW, FOULING_CENTRAL, etc.)
"""

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    HULL_LWL,
    HULL_S_WET,
    KN_TO_MS,
    NU_WATER,
    PROP_BAR,
    PROP_DIAMETER,
    PROP_N_BLADES,
    RHO_WATER,
)


@dataclass
class FoulingScenario:
    """Piecewise-linear fouling roughness growth trajectory.

    Defined by discrete (year, ks) data points from literature,
    with linear interpolation between them.
    """
    name: str
    years: list      # [year, ...] must start at 0
    ks_values: list  # [ks in um, ...]

    def ks_um_at(self, t: float) -> float:
        """Return ks [um] at time t [years] by linear interpolation."""
        return float(np.interp(t, self.years, self.ks_values))

    def ks_m_at(self, t: float) -> float:
        """Return ks [m] at time t [years]."""
        return self.ks_um_at(t) * 1e-6


# ============================================================
# Pre-defined fouling scenarios
# ============================================================

# Hull fouling scenarios
FOULING_LOW = FoulingScenario(
    name="Low (Nordic, good AF)",
    years=     [0,  1,   2,   3,   4,   5],
    ks_values= [30, 50,  80,  150, 250, 500],   # um
)
FOULING_CENTRAL = FoulingScenario(
    name="Central (temperate, std SPC)",
    years=     [0,  1,   2,   3,    4,    5],
    ks_values= [50, 100, 300, 700,  1500, 3000],
)
FOULING_HIGH = FoulingScenario(
    name="High (tropical, poor AF)",
    years=     [0,   1,   2,    3,    4,    5],
    ks_values= [100, 500, 2000, 5000, 8000, 10000],
)

# Propeller blade fouling scenario.
# Blades are partially self-cleaning due to hydrodynamic shear at RPM,
# so roughness grows much slower than the hull.  Typical values for a
# coated CPP blade in temperate waters (Nordic/North Sea).
BLADE_FOULING = FoulingScenario(
    name="Blade (coated CPP, Nordic)",
    years=     [0,  1,   2,   3,   4,    5],
    ks_values= [10, 20,  40,  70,  100,  150],  # um
)


# ============================================================
# Hull roughness resistance model
# ============================================================

def hull_roughness_delta_R_kN(
    speed_kn: float,
    ks_hull_m: float,
    ks_clean_m: float = 30e-6,
    S_wet: float = HULL_S_WET,
    L_wl: float = HULL_LWL,
    nu: float = NU_WATER,
) -> float:
    """Compute hull frictional resistance increase due to roughness [kN].

    Uses the Townsin-Dey roughness penalty formula (Townsin 2003):

        10^3 x dCF = 44 x [(ks/L)^(1/3) - 10*Re_L^(-1/3)] + 0.125

    i.e. dCF = {44 x [(ks/L)^(1/3) - 10*Re^(-1/3)] + 0.125} x 10^-3

    applied as the difference between the current roughness state and the
    clean-hull baseline (ks_clean, default 30 um for new AF coating).
    The clean-hull resistance is already in our calm-water data, so we
    only add the INCREMENT above the clean baseline.

    Parameters
    ----------
    speed_kn : float
        Vessel speed [knots].
    ks_hull_m : float
        Current hull equivalent sand-grain roughness [m].
    ks_clean_m : float
        Clean-hull roughness [m]. Default 30e-6 (new AF coating).
    S_wet : float
        Wetted surface area [m^2].
    L_wl : float
        Waterline length [m].
    nu : float
        Kinematic viscosity [m^2/s].

    Returns
    -------
    delta_R_kN : float
        Extra resistance due to roughness above clean baseline [kN].
        Always >= 0.
    """
    if ks_hull_m <= ks_clean_m:
        return 0.0

    V = speed_kn * KN_TO_MS
    Re_L = V * L_wl / nu

    def _townsin_delta_cf(ks_m):
        """Townsin-Dey dCF for a given ks."""
        if ks_m <= 0 or Re_L < 1:
            return 0.0
        return max(0.0, (44.0 * ((ks_m / L_wl) ** (1.0 / 3.0)
                                  - 10.0 * Re_L ** (-1.0 / 3.0))
                         + 0.125) * 1e-3)

    dcf_current = _townsin_delta_cf(ks_hull_m)
    dcf_clean = _townsin_delta_cf(ks_clean_m)
    delta_cf = max(0.0, dcf_current - dcf_clean)

    # dR = dCF x 0.5 x rho x V^2 x S_wet
    delta_R_N = delta_cf * 0.5 * RHO_WATER * V ** 2 * S_wet
    return delta_R_N / 1000.0


# ============================================================
# Propeller blade roughness model
# ============================================================

def propeller_roughness_fuel_factor(
    speed_kn: float,
    shaft_rpm: float,
    ks_blade_m: float,
    ks_clean_m: float = 10e-6,
    D: float = PROP_DIAMETER,
    BAR: float = PROP_BAR,
    n_blades: int = PROP_N_BLADES,
    nu: float = NU_WATER,
) -> float:
    """Estimate propeller fuel rate multiplier due to blade surface roughness.

    Uses a simplified strip-theory approach at the 0.75R reference station
    (standard ITTC practice).  The extra blade drag is converted to a power
    increase fraction, which is then applied as a fuel rate multiplier.

    Method:
        1. Compute resultant velocity W at 0.75R from RPM and advance speed
        2. Compute chord at 0.75R from Wageningen B-series chord distribution
        3. Compute dCF = CF_rough - CF_smooth using ITTC 1957 + Townsin
        4. dCD = 2 x dCF (both blade surfaces)
        5. Estimate dP/P from the drag-to-power ratio at 0.75R

    Parameters
    ----------
    speed_kn : float
        Vessel speed [knots].
    shaft_rpm : float
        Shaft (propeller) RPM.
    ks_blade_m : float
        Blade surface equivalent sand-grain roughness [m].
    ks_clean_m : float
        Clean blade roughness [m]. Default 10e-6 (polished).
    D : float
        Propeller diameter [m].
    BAR : float
        Blade area ratio.
    n_blades : int
        Number of blades.
    nu : float
        Kinematic viscosity [m^2/s].

    Returns
    -------
    fuel_factor : float
        Multiplier on fuel rate (>= 1.0).  E.g. 1.03 = 3% increase.
    """
    if ks_blade_m <= ks_clean_m or shaft_rpm < 1.0:
        return 1.0

    R = D / 2.0
    r_075 = 0.75 * R
    omega = shaft_rpm * 2.0 * math.pi / 60.0

    # Advance speed (use wake-corrected Va for consistency, but we
    # approximate with vessel speed here -- the wake correction is small
    # and the same for smooth and rough)
    Va = speed_kn * KN_TO_MS * (1.0 - 0.24)  # approximate w ~ 0.24

    # Resultant velocity at 0.75R
    V_tan = omega * r_075
    W = math.sqrt(Va ** 2 + V_tan ** 2)

    # Chord at 0.75R -- Wageningen C4 chord distribution:
    # c/D = BAR x k_c(r/R) / n_blades, where k_c is the expanded chord factor.
    # For Wageningen B/C series at r/R=0.75, k_c ~ 1.54 (from tabulated data).
    K_C_075 = 1.54
    c_075 = D * BAR * K_C_075 / n_blades

    # Chord Reynolds number
    Re_c = W * c_075 / nu

    # ITTC 1957 friction line for smooth blade:
    CF_smooth = 0.075 / (math.log10(Re_c) - 2.0) ** 2

    # Townsin roughness penalty at chord scale:
    # 10^3 x dCF = 44 x [(ks/c)^(1/3) - 10*Re_c^(-1/3)] + 0.125
    def _townsin_cf(ks_m, Re):
        if ks_m <= 0 or Re < 1:
            return 0.0
        L_c = c_075  # use chord as length scale
        return max(0.0, (44.0 * ((ks_m / L_c) ** (1.0 / 3.0)
                                  - 10.0 * Re ** (-1.0 / 3.0))
                         + 0.125) * 1e-3)

    dcf_rough = _townsin_cf(ks_blade_m, Re_c)
    dcf_clean = _townsin_cf(ks_clean_m, Re_c)
    delta_CF = max(0.0, dcf_rough - dcf_clean)

    # dCD for both blade surfaces, with form factor for airfoil sections
    FORM_FACTOR = 1.3  # typical for marine propeller sections
    delta_CD = FORM_FACTOR * 2.0 * delta_CF

    # Power increase fraction:
    # At 0.75R the power-absorbing (tangential) fraction of drag is cos(phi)
    # where phi = atan2(Va, V_tan) is the inflow angle.
    # dP/P ~ (dCD / CD_total) x (viscous drag fraction of total power)
    #
    # More directly: the smooth-blade CD ~ 2 x CF_smooth x FORM_FACTOR
    # The viscous drag fraction of propeller power is approximately
    # CD / (CD + CL x tan(phi_i)) where phi_i is the induced inflow angle.
    # For a lightly loaded prop this is roughly 10-20% of total power.
    #
    # Simplification: dP/P ~ dCD / (2 x CF_smooth x FORM_FACTOR) x f_drag
    # where f_drag ~ 0.15 is the viscous drag power fraction.
    CD_smooth = FORM_FACTOR * 2.0 * CF_smooth
    if CD_smooth < 1e-10:
        return 1.0

    # The relative drag increase applies to the viscous drag component only.
    # Viscous losses are typically 12-18% of delivered power for a CPP at
    # moderate loading.  Use 15% as representative.
    DRAG_POWER_FRACTION = 0.15
    delta_P_frac = (delta_CD / CD_smooth) * DRAG_POWER_FRACTION

    return 1.0 + max(0.0, delta_P_frac)
