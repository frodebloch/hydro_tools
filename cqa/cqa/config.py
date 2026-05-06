"""Configuration container for the cqa prototype.

Holds the few high-level numbers needed to drive the P1 prototype. The
production system will load these from brucon prototxt configs; for the
study we hard-code a CSOV-flavoured defaults helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VesselParticulars:
    """Principal dimensions and mass-related quantities for the linearised model."""

    name: str
    lpp: float  # m, length between perpendiculars
    loa: float  # m, length overall
    beam: float  # m
    draft: float  # m
    block_coefficient: float
    rho_water: float = 1025.0  # kg/m^3

    # Linearised hydrodynamic parameters at zero forward speed.
    # Added mass / inertia are stored as fractions of rigid-body values for
    # convenience; production code will pull these from the brucon section
    # integration. Defaults are sourced from porting brucon's section
    # integration + Lewis-form added-mass algorithm
    # (libs/dp/vessel_model/vessel_coefficients.cpp::AddedMass::A11/A22/A66
    # with section_cy from SectionAddedMassCoefficient) and applying it to
    # the CSOV `vessel_data.prototxt` section table at design draft 6.50 m.
    # The resulting M11/M22/M66 totals match brucon to within ~1 % for the
    # CSOV. See analysis.md sec.12.17.
    surge_added_mass_frac: float = 0.060
    sway_added_mass_frac: float = 0.671
    yaw_added_inertia_frac: float = 0.620

    # Linear damping coefficients in body frame (zero-forward-speed approximation).
    # Units: surge/sway in N s/m, yaw in N m s / rad.
    # These feed the linearised vessel model. For the study we set them so
    # the resulting closed-loop bandwidth and damping are realistic.
    linear_damping_surge: float = 0.0
    linear_damping_sway: float = 0.0
    linear_damping_yaw: float = 0.0

    @property
    def displacement_mass(self) -> float:
        return self.rho_water * self.lpp * self.beam * self.draft * self.block_coefficient

    @property
    def yaw_inertia(self) -> float:
        # Rigid-body yaw inertia using empirical r66 = 0.25 * Lpp.
        r66 = 0.25 * self.lpp
        return self.displacement_mass * r66 ** 2


@dataclass
class WindParticulars:
    """Wind force model inputs (DNV-style table or simple drag model)."""

    lateral_area: float  # m^2
    frontal_area: float  # m^2
    lateral_area_centre: float  # m, longitudinal coordinate of lateral area centroid
    height_of_wind_area_centre: float  # m
    rho_air: float = 1.225  # kg/m^3

    # Drag coefficient amplitudes used by the simple cosine model in WindForceModel
    # if no coefficient table is supplied.
    cx_amp: float = 0.7
    cy_amp: float = 0.9
    cn_amp: float = 0.05  # yaw-moment coefficient amplitude (dimensionless, scaled by Loa)


@dataclass
class CurrentParticulars:
    """Current force model inputs (DNV ST-0111 style)."""

    cx_amp: float = 0.07  # surge force coefficient
    cy_amp: float = 0.6  # sway force coefficient
    cn_amp: float = 0.05  # yaw moment coefficient
    rho_water: float = 1025.0


@dataclass
class WaveDriftParticulars:
    """Mean wave drift coefficients (very simplified placeholder for P1).

    Production: replaced by RAO-based drift forces from `pdstrip.dat`.
    """

    drift_x_amp: float = 8000.0  # N per (Hs^2 [m^2]) at head sea
    drift_y_amp: float = 25000.0  # N per (Hs^2 [m^2]) at beam sea
    drift_n_amp: float = 0.0  # N m per (Hs^2 [m^2])


@dataclass
class ThrustCapability:
    """Per-DOF intact thrust/moment capability ceiling.

    These are the maximum sustained body-frame forces the DP system can
    deliver in each DOF when *all* thruster groups are healthy. Used as the
    saturation cap for the linearised transient analysis.

    CSOV-class defaults (Norwind SOV, ~8000 t displacement, 4 azimuths +
    2 tunnels): roughly ~500 kN surge, ~700 kN sway, ~30 MN m yaw moment.
    These are coarse first-cut numbers; production should derive them from
    brucon `BasicAllocator` envelopes per actual thruster layout.
    """

    surge: float = 5.0e5  # N
    sway: float = 7.0e5  # N
    yaw: float = 3.0e7  # N m

    def as_array(self):
        import numpy as np
        return np.array([self.surge, self.sway, self.yaw])


@dataclass
class GangwayConfig:
    """Gangway geometry (CSOV defaults). See §4.4 of analysis.md."""

    base_position_body: tuple[float, float, float] = (5.0, -9.0, -8.0)
    rotation_centre_height_above_base: float = 12.5  # m, operator-set; default mid-stroke
    rotation_centre_height_max: float = 25.0  # m
    telescope_min: float = 18.0  # m
    telescope_max: float = 32.0  # m
    # Display-only thresholds (no pass/fail in P1 except telescope end-stops).
    telescope_velocity_default_threshold: float = 0.5  # m/s


@dataclass
class OperationalLimits:
    """Operator-set position-keeping deviation limits.

    These are compared against the predicted post-WCFDI vessel position
    excursion (relative to the gangway base point) to produce the
    operator-facing P_exceed numbers (warning, alarm).

    Defaults below are the gangway-base-relative warning/alarm radii used
    on a typical CSOV W2W operation. Production should source these from
    the operator's PosRef alert settings or from the gangway-vendor
    operating envelope.
    """

    position_warning_radius_m: float = 2.0
    position_alarm_radius_m: float = 4.0


@dataclass
class ControllerParams:
    """Linearised DP controller / observer tuning.

    Single source of truth for the closed-loop bandwidth, damping and the
    bias-estimator / thruster lag time constants used by both the WCFDI
    Monte Carlo (`cqa.wcfdi_mc`) and the intact-prior Rice analysis
    (`cqa.operator_view.summarise_intact_prior`).

    Defaults match the values previously hard-coded as `wcfdi_mc`
    arguments and reflect a typical CSOV-class DP-2 vessel:

      omega_n_surge / sway = 0.06 rad/s  (closed-loop natural frequency
                                          ~95 s period; well below the
                                          wave band, well above the
                                          slow-drift band).
      omega_n_yaw          = 0.05 rad/s.
      zeta                 = 0.9 (over-damped DP, standard).
      T_b                  = 1000 s (bias-estimator time constant; matches
                                     brucon's `passive_observer.cpp`
                                     `T_b_default = 1000 s`. Earlier 100 s
                                     default was a misreading; verified
                                     against config_csov/observer.prototxt
                                     and against §12.20 sandbox closure
                                     of the σ_y_LF gap to brucon).
      T_thr                = 5 s   (1st-order thruster lag).

    Setting `omega_n` and `zeta` here ensures that the WCFDI MC, the
    intact-prior Rice analysis and any future operating-point sweep all
    integrate the *same* closed-loop transfer function, eliminating
    silent drift between the two pipelines.
    """

    omega_n_surge: float = 0.06
    omega_n_sway: float = 0.06
    omega_n_yaw: float = 0.05
    zeta_surge: float = 0.9
    zeta_sway: float = 0.9
    zeta_yaw: float = 0.9
    bias_time_constant_s: float = 1000.0   # T_b (brucon passive_observer default)
    thruster_time_constant_s: float = 5.0  # T_thr

    @property
    def omega_n(self) -> tuple[float, float, float]:
        return (self.omega_n_surge, self.omega_n_sway, self.omega_n_yaw)

    @property
    def zeta(self) -> tuple[float, float, float]:
        return (self.zeta_surge, self.zeta_sway, self.zeta_yaw)


@dataclass
class ObserverParams:
    """Brucon-style Fossen passive observer + Saelid/Jensen wave filter gains.

    Per-DOF gains read from `config_csov/observer.prototxt`. CSOV defaults:
    surge & sway are identical (K_a1 = 0.12, K_b1 = 0.0012); heading uses
    higher gains (K_a1 = 0.2, K_b1 = 0.002) reflecting tighter heading
    control. T_b = 1000 s and ω_c = 1.04 rad/s are common to all DOFs and
    in fact identical to `ControllerParams.bias_time_constant_s` (kept
    duplicated here so the observer module can be invoked standalone).

    The wave-filter peak frequency ω_w = 2π / Tp is *not* a tuning gain
    but a sea-state-dependent input. Brucon estimates it on-line from
    the pitch motion (`EstWavePeriodPitch`) and applies the same value
    to all 4 controlled DOFs. cqa takes Tp from the operating point and
    uses it identically across surge/sway/yaw observer blocks.

    The wave-filter notch damping ζ_n is gain-scheduled on ω_w via the
    brucon `ScaleGainLinear` rule:
        Tp ≤ 18 s : ζ_n = 0.25
        Tp ≥ 10 s : ζ_n = 0.10
        else      : linear interpolation
    (note `ScaleGainLinear` lo/hi convention: lo gain at low ω_w, hi at
    high ω_w; for the period grid this inverts to lo gain at long Tp).

    Notation matches brucon `passive_observer.cpp`:
      e             = y_meas − ŷ_LF − η̂_w        (wave-corrected innovation)
      ŷ_LF_dot      = ν̂ + ω_c · e
      ν̂_dot         = (1/M)·(b̂ + u + D·ν̂) + K_a1·e
      b̂_dot         = -(1/T_b)·b̂ + K_b1·e
      ξ_w_dot       = η̂_w + k1_f · e
      η̂_w_dot       = -ω_w² · ξ_w − 2 ζ_n ω_w · η̂_w + k2_f · e
    with
      k1_f = -2 (1 − ζ_n) ω_c / ω_w
      k2_f =  2 (1 − ζ_n) ω_w
    """

    K_a1_surge: float = 0.12
    K_a1_sway: float = 0.12
    K_a1_yaw: float = 0.20
    K_b1_surge: float = 0.0012
    K_b1_sway: float = 0.0012
    K_b1_yaw: float = 0.0020
    omega_c: float = 1.04           # position wave-filter cutoff [rad/s]
    bias_time_constant_s: float = 1000.0   # T_b, same as ControllerParams

    @property
    def K_a1(self) -> tuple[float, float, float]:
        return (self.K_a1_surge, self.K_a1_sway, self.K_a1_yaw)

    @property
    def K_b1(self) -> tuple[float, float, float]:
        return (self.K_b1_surge, self.K_b1_sway, self.K_b1_yaw)


def wave_filter_zeta_n(Tp: float) -> float:
    """Brucon ScaleGainLinear schedule for the wave-filter notch damping.

    Mirrors `dp_estimator_wrapper`-side construction: ζ_n falls from 0.25
    at long Tp (= low ω_w) to 0.10 at short Tp (= high ω_w), linearly in
    ω_w over [2π/18, 2π/10] rad/s.
    """
    import math
    omega_w = 2.0 * math.pi / Tp
    omega_lo, omega_hi = 2.0 * math.pi / 18.0, 2.0 * math.pi / 10.0
    zeta_lo, zeta_hi = 0.25, 0.10
    if omega_w <= omega_lo:
        return zeta_lo
    if omega_w >= omega_hi:
        return zeta_hi
    f = (omega_w - omega_lo) / (omega_hi - omega_lo)
    return zeta_lo + f * (zeta_hi - zeta_lo)


@dataclass
class CqaConfig:
    vessel: VesselParticulars
    wind: WindParticulars
    current: CurrentParticulars
    wave_drift: WaveDriftParticulars
    gangway: GangwayConfig = field(default_factory=GangwayConfig)
    thrust_capability: ThrustCapability = field(default_factory=ThrustCapability)
    operational_limits: OperationalLimits = field(default_factory=OperationalLimits)
    controller: ControllerParams = field(default_factory=ControllerParams)
    observer: ObserverParams = field(default_factory=ObserverParams)


def csov_default_config() -> CqaConfig:
    """CSOV-flavoured default configuration, sourced from brucon `config_csov`.

    Numbers are read from:
      ~/src/brucon/modules/config_csov/vessel_data.prototxt.in
      ~/src/brucon/modules/config_csov/vessel_wind_data.prototxt.in
      ~/src/brucon/modules/config_csov/posrefs.prototxt.in
    """
    vessel = VesselParticulars(
        name="Norwind SOV",
        lpp=101.1,
        loa=111.5,
        beam=22.4,
        draft=6.50,
        block_coefficient=0.7369815,
    )
    # Damping: pick linear coefficients that give realistic open-loop time
    # constants for an 8000 t CSOV. We target ~60 s surge open-loop time
    # constant and ~40 s sway, ~80 s yaw (driven by wind/wave-drift damping).
    m11 = vessel.displacement_mass * (1.0 + vessel.surge_added_mass_frac)
    m22 = vessel.displacement_mass * (1.0 + vessel.sway_added_mass_frac)
    m66 = vessel.yaw_inertia * (1.0 + vessel.yaw_added_inertia_frac)
    vessel.linear_damping_surge = m11 / 60.0
    vessel.linear_damping_sway = m22 / 40.0
    vessel.linear_damping_yaw = m66 / 80.0

    wind = WindParticulars(
        lateral_area=1755.0,
        frontal_area=700.0,
        lateral_area_centre=20.24,
        height_of_wind_area_centre=14.0,
    )
    current = CurrentParticulars()
    wave_drift = WaveDriftParticulars()
    gangway = GangwayConfig()
    thrust_cap = ThrustCapability()
    op_limits = OperationalLimits()
    controller = ControllerParams()
    observer = ObserverParams()
    return CqaConfig(
        vessel=vessel,
        wind=wind,
        current=current,
        wave_drift=wave_drift,
        gangway=gangway,
        thrust_capability=thrust_cap,
        operational_limits=op_limits,
        controller=controller,
        observer=observer,
    )
