"""Result dataclasses for voyage simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class HourlyResult:
    """Result for a single hourly evaluation."""
    time_hours: float
    lat: float
    lon: float
    heading_deg: float
    hs: float
    tp: float
    wind_speed: float
    R_calm_kN: float
    R_aw_kN: float
    R_wind_kN: float                                     # Blendermann wind resistance
    F_flettner_kN: float
    T_required_kN: float
    T_required_no_flettner_kN: float     # thrust without Flettner reduction
    factory_fuel_rate: Optional[float]   # g/h (with Flettner thrust reduction)
    factory_no_flettner_fuel_rate: Optional[float] = None  # g/h (without Flettner)
    optimised_fuel_rate: Optional[float] = None  # g/h (with Flettner)
    optimised_no_flettner_fuel_rate: Optional[float] = None  # g/h (without Flettner)
    rotor_power_kW: float = 0.0          # Flettner rotor electrical power [kW]
    rotor_fuel_rate: float = 0.0         # rotor fuel cost [g/h] via genset SFOC
    factory_pitch: Optional[float] = None
    factory_rpm: Optional[float] = None
    optimised_pitch: Optional[float] = None
    optimised_rpm: Optional[float] = None


@dataclass
class VoyageResult:
    """Result for a complete voyage."""
    departure: datetime
    total_hours: float
    # All four on the SAME feasible-hours basis (hours where factory+Fl feasible
    # AND optimiser feasible AND factory_NF feasible AND opt_NF feasible):
    total_fuel_factory_kg: float        # factory at Flettner-reduced thrust
    total_fuel_factory_no_flettner_kg: float  # factory at no-Flettner thrust
    total_fuel_opt_no_flettner_kg: float  # optimiser at no-Flettner thrust
    total_fuel_optimised_kg: float      # optimiser at Flettner-reduced thrust
    saving_pct: float           # (factory_fl - opt_fl) / factory_fl
    saving_kg: float            # factory_fl - opt_fl
    # Split savings (same feasible-hours basis):
    # Baseline = factory no-Flettner
    saving_pitch_rpm_kg: float  # factory_nf - opt_nf
    saving_flettner_kg: float   # opt_nf - opt_fl
    mean_hs: float
    mean_wind: float
    mean_R_aw_kN: float
    mean_R_wind_kN: float
    mean_F_flettner_kN: float
    mean_rotor_power_kW: float = 0.0     # mean Flettner rotor electrical power
    total_rotor_fuel_kg: float = 0.0     # total fuel for rotor drive (all hours)
    hull_ks_um: float = 0.0              # hull roughness [µm]
    blade_ks_um: float = 0.0             # blade roughness [µm]
    R_roughness_kN: float = 0.0          # hull roughness resistance increment [kN]
    blade_fuel_factor: float = 1.0       # propeller roughness fuel multiplier
    n_hours_both_feasible: int = 0
    n_hours_factory_infeasible: int = 0  # factory can't deliver, optimiser can
    n_hours_total: int = 0
    hourly: list[HourlyResult] = field(default_factory=list)


@dataclass
class SpeedSweepResult:
    """Aggregated results for one speed in a speed-sensitivity sweep."""
    speed_kn: float
    transit_hours: float
    voyages_per_year: float
    n_voyages: int
    # Per-voyage means
    mean_fuel_factory_nf_kg: float
    mean_fuel_factory_fl_kg: float
    mean_fuel_opt_nf_kg: float
    mean_fuel_opt_fl_kg: float
    mean_saving_pitch_rpm_kg: float
    mean_saving_flettner_kg: float
    mean_saving_total_kg: float
    mean_saving_pct: float
    pct_pitch_rpm: float            # % of factory_nf baseline
    pct_flettner: float             # % of factory_nf baseline
    # Annualized (tonnes/year)
    ann_fuel_factory_nf_t: float
    ann_fuel_opt_fl_t: float
    ann_saving_pitch_rpm_t: float
    ann_saving_flettner_t: float
    ann_saving_total_t: float
    # Weather
    mean_hs: float
    mean_wind: float
    mean_R_aw_kN: float
    mean_R_wind_kN: float
    mean_F_flettner_kN: float
    mean_rotor_power_kW: float
    # Feasibility
    pct_factory_infeasible: float
    # Roughness
    hull_ks_um: float = 0.0
    blade_ks_um: float = 0.0
    R_roughness_kN: float = 0.0
    blade_fuel_factor: float = 1.0
