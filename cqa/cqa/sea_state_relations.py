"""Wind-wave relations for fully-developed seas (DNV-ST-0111 / DNV-RP-C205).

Used by the operability polar (`cqa.operability_polar`) to derive a
single-parameter sea state from a wind speed Vw at 10 m AMSL. This
gives a "Level-3-light" capability sweep where the radial axis is one
operator-familiar number (wind speed) instead of the full (Vw, Hs, Tp)
triple, mirroring the standard DP capability plot convention.

Pierson-Moskowitz fully-developed sea
-------------------------------------
For a fully-developed wind sea (infinite fetch, infinite duration) at
mean wind speed U_10 [m/s] at 10 m above sea level:

    H_s = 0.21 * U_10^2 / g                       [m]
    omega_p = 0.877 * g / U_10                    [rad/s]
    T_p = 2 * pi / omega_p
        = 2 * pi * U_10 / (0.877 * g)             [s]
        ~= 0.71 * U_10                            (with g = 9.81 m/s^2)

The H_s coefficient 0.21 comes from the PM spectrum integrated to
infinity (m_0 = (alpha * g^2) / (4 * beta * omega_p^4) with alpha =
0.0081, beta = 0.74, U_19.5 reference; converted to U_10 via the
1/7 power-law profile this collapses to H_s = 0.21 U_10^2 / g).
The omega_p coefficient 0.877 is the PM peak frequency at U_10
(DNV-RP-C205 sec 3.5.5.4, eq. 3.5.36).

These match
* DNV-RP-C205 (2021), sec. 3.5.5.4 ("Pierson-Moskowitz spectrum"),
  eqs. (3.5.35)-(3.5.36).
* DNV-ST-0111 (2021), App. F.1 ("Wind-wave correlations for
  station-keeping analyses").

Limitations
-----------
* Fully-developed = no fetch limitation. Real North-Sea SOV operations
  are usually fetch-limited; PM overestimates Hs at low Vw and
  underestimates Tp. JONSWAP fetch-limited refinement is left for a
  follow-up turn.
* Single sea state: combined wind-sea + swell collapses into one (Hs,
  Tp). Adequate for a feasibility-grade capability sweep; refine with
  Torsethaugen for production.
"""

from __future__ import annotations

from dataclasses import dataclass


G_STD: float = 9.81  # gravitational acceleration [m/s^2]

# Pierson-Moskowitz coefficients (DNV-RP-C205 sec 3.5.5.4).
PM_HS_COEFF: float = 0.21      # H_s = PM_HS_COEFF * U_10^2 / g
PM_OMEGA_P_COEFF: float = 0.877  # omega_p = PM_OMEGA_P_COEFF * g / U_10


@dataclass(frozen=True)
class WindSeaState:
    """Sea state derived from wind speed via the wind-wave relation.

    Fields
    ------
    Vw_m_s : wind speed at 10 m AMSL [m/s].
    Hs_m   : significant wave height [m].
    Tp_s   : peak period [s].
    omega_p : peak angular frequency [rad/s].
    """

    Vw_m_s: float
    Hs_m: float
    Tp_s: float
    omega_p: float


def pm_hs_from_vw(Vw: float, g: float = G_STD) -> float:
    """Pierson-Moskowitz fully-developed H_s [m] from wind U_10 [m/s].

    H_s = 0.21 * Vw^2 / g.
    """
    if Vw < 0.0:
        raise ValueError(f"Vw must be >= 0, got {Vw}")
    return float(PM_HS_COEFF * Vw * Vw / g)


def pm_tp_from_vw(Vw: float, g: float = G_STD) -> float:
    """Pierson-Moskowitz fully-developed T_p [s] from wind U_10 [m/s].

    T_p = 2 * pi * Vw / (0.877 * g). Returns +inf for Vw == 0.
    """
    if Vw < 0.0:
        raise ValueError(f"Vw must be >= 0, got {Vw}")
    if Vw == 0.0:
        return float("inf")
    import math
    return float(2.0 * math.pi * Vw / (PM_OMEGA_P_COEFF * g))


def pm_sea_state(Vw: float, g: float = G_STD) -> WindSeaState:
    """Bundle (H_s, T_p, omega_p) for a given wind speed.

    Parameters
    ----------
    Vw : float. Mean wind speed at 10 m AMSL [m/s].
    g : float, default 9.81. Gravitational acceleration [m/s^2].

    Returns
    -------
    WindSeaState.
    """
    if Vw < 0.0:
        raise ValueError(f"Vw must be >= 0, got {Vw}")
    Hs = pm_hs_from_vw(Vw, g=g)
    if Vw == 0.0:
        return WindSeaState(Vw_m_s=0.0, Hs_m=0.0, Tp_s=float("inf"),
                            omega_p=0.0)
    omega_p = float(PM_OMEGA_P_COEFF * g / Vw)
    Tp = pm_tp_from_vw(Vw, g=g)
    return WindSeaState(Vw_m_s=float(Vw), Hs_m=float(Hs), Tp_s=float(Tp),
                        omega_p=float(omega_p))


__all__ = [
    "G_STD",
    "PM_HS_COEFF",
    "PM_OMEGA_P_COEFF",
    "WindSeaState",
    "pm_hs_from_vw",
    "pm_tp_from_vw",
    "pm_sea_state",
]
