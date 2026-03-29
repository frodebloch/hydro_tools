"""Physical models and data for voyage comparison."""

from .constants import (
    DATA_PATH_C440,
    DATA_PATH_C455,
    DATA_PATH_C470,
    GENSET_SFOC,
    HULL_SPEEDS_KN,
    HULL_THRUST_CALM_KN,
    HULL_THRUST_KN,
    HULL_T_DEDUCTION,
    HULL_WAKE,
    KN_TO_MS,
    NORA3_DATA_DIR,
    PDSTRIP_DAT,
    PROP_BAR,
    PROP_DESIGN_PITCH,
    PROP_DIAMETER,
    RHO_AIR,
    RHO_WATER,
)
from .roughness import (
    BLADE_FOULING,
    FOULING_CENTRAL,
    FOULING_HIGH,
    FOULING_LOW,
    FoulingScenario,
)
from .route import (
    ROUTE_GOTHENBURG_ROTTERDAM,
    ROUTE_ROTTERDAM_GOTHENBURG,
    Route,
    Waypoint,
)
