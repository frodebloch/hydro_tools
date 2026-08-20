"""Link Galaxy vessel-parameter overlay.

Loads the physical constants, Blendermann tables and Aas206 defaults
from ``models.constants`` and then overrides every vessel-specific
value with the Link Galaxy numbers.

Usage
-----
The downstream simulation code (``simulation.orchestrator``,
``simulation.engine``, ``models.combinator`` ...) does
``from models.constants import <NAME>`` at import time, so a
per-vessel switch has to happen *before* those modules load.

The Link Galaxy driver ``link_galaxy/annual_run.py`` does::

    import sys
    import models.vessel_link_galaxy as lg
    sys.modules["models.constants"] = lg
    # ... only now import simulation.orchestrator etc.

That gives every downstream ``from models.constants import X`` the
overridden LG value while non-vessel symbols (physical constants,
Blendermann coefficients, propeller open-water file paths) fall
through to their originals.

Data provenance
---------------
- Propeller / gear / wake / t / eta_R:
    brucon/modules/config_link_galaxy/propulsion_optimiser_config_4.prototxt.in
    brucon/modules/config_link_galaxy/gear_control_parameters_4.prototxt.in
- Calm-water resistance:
    V2597 DWL table (Vs 0..19.5 kn -> RTS 0..608.5 kN) at Tm ~4.5 m.
- PDStrip:
    brucon/build/bin/vessel_simulator_config/link_galaxy_pdstrip.dat
- Hull geometry:
    modules/config_link_galaxy/hull_geometry/ (R sections at T=5.0 m).
"""

from pathlib import Path

import numpy as np

# Physical constants, Blendermann tables and Aas206 defaults.
from .constants import *  # noqa: F401,F403


# ============================================================
# File paths
# ============================================================

PDSTRIP_DAT = ("/home/blofro/src/brucon/build/bin/"
               "vessel_simulator_config/link_galaxy_pdstrip.dat")

NORA3_DATA_DIR = Path(__file__).parent.parent / "data" / "nora3_link_galaxy"


# ============================================================
# Propeller and drivetrain
# ============================================================

PROP_DIAMETER = 4.30            # m
PROP_BAR = 0.435                # blade area ratio (proxy: C4_40 series)
PROP_DESIGN_PITCH = 1.05        # P/D at design
PROP_N_BLADES = 4

GEAR_RATIO = 6.26174            # engine rpm / shaft rpm
SHAFT_EFF = 0.97                # shaft line efficiency
GENSET_SFOC = 215.0             # g/kWh (aux/harbour genset placeholder)


# ============================================================
# Calm-water resistance and hull efficiency (V2597 DWL, Tm ~4.5 m)
# ============================================================
# TODO(operator): also load the loaded-condition table (Tm 5.5-6.1 m).

HULL_SPEEDS_KN = np.array([
     0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
    10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5,
    16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
])
# Holtrop-Mennen (ITTC 1957 viscous + wave-making) evaluated from the
# real R-section geometry.  See link_galaxy/holtrop_mennen.py; the port
# reproduces brucon's ``VesselResistance::HoltropCalmWaterResistance``
# so the Python voyage-optimiser sees the same calm-water curve that
# the C++ propulsion optimiser would compute internally.  Endpoint
# 570 kN @ 19.5 kn is ~94% of the V2597 towing-tank value (608.5 kN);
# replace with the tabulated V2597 curve once it lands.
from link_galaxy.holtrop_mennen import resistance_table as _hm_table
HULL_RESISTANCE_KN = _hm_table(HULL_SPEEDS_KN, 137.2, 19.0, 5.0)

HULL_WAKE = np.full_like(HULL_SPEEDS_KN, 0.262)
HULL_T_DEDUCTION = np.full_like(HULL_SPEEDS_KN, 0.172)
HULL_ETA_R = np.full_like(HULL_SPEEDS_KN, 1.025)
HULL_THRUST_CALM_KN = HULL_RESISTANCE_KN / (1.0 - HULL_T_DEDUCTION)


# ============================================================
# Hull geometry
# ============================================================

HULL_LWL = 137.2                # Lpp [m]
HULL_LOA = 147.2                # LOA [m]
HULL_B = 19.0                   # beam [m]
HULL_T_MEAN = 5.00              # mean draft used for pdstrip / R sections
HULL_DISPLACEMENT_M3 = 8973.0   # from R section integration at T=5.0 m
HULL_S_WET = 3900.0             # TODO: integrate from geomet.out


# ============================================================
# Wind (Blendermann A_F, A_L placeholders)
# ============================================================
# TODO: extract A_F and A_L from the Link Galaxy GA drawing / AIS
# silhouette; below are conservative placeholders.

WIND_AREA_FRONTAL_M2 = 460.0
WIND_AREA_LATERAL_M2 = 2100.0
VESSEL_LOA_M = HULL_LOA


# ============================================================
# Engine factory override
# ============================================================
# simulation.orchestrator uses ENGINE_FACTORY to instantiate the engine.
# For Link Galaxy this is the Wartsila Vasa 32D 16V (derated 4000 kW),
# anchored to the Marine Project Guide 2/1997 three-point SFOC data.
from optimiser import make_wartsila_vasa32_16v as ENGINE_FACTORY  # noqa: E402
