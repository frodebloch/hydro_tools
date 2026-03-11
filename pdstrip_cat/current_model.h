// current_model.h — Shared ocean current environment + per-body drag forces
//
// Architecture: two-layer design for the DP alongside scenario.
//
//   CurrentEnvironment  (shared, one instance)
//     Owns the environmental state: mean current, variability spectrum,
//     depth profile, and the pre-synthesized time-domain velocity signal.
//     Queried by ALL bodies in the simulation (FWT platform, DP vessel,
//     mooring lines, etc.) to get consistent current velocity at any
//     position and time.
//
//   CurrentForceModel  (per-body, one per simulator)
//     Computes the Morison drag on a specific hull, given current
//     velocity from the shared environment and the body's own velocity.
//     The FloatingPlatformSimulator owns one for the spar; the
//     VesselSimulatorWrapper uses its existing current-through-water
//     damping formulation but queries the same environment.
//
// This ensures the DP vessel and FWT see the SAME current realization,
// so correlated surge motions are correctly captured.
//
// Reference: pdstrip_cat/slow_drift_swell.py (Python prototype)
//            pdstrip_cat/SESSION_NOTES.md, Sessions 59-60

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "current_drag.h"
#include "current_spectrum.h"

namespace brucon {
namespace simulator {

// ============================================================
// Enums (in the simulator namespace — shared across vessel/platform)
// ============================================================

/// Spectrum model selection for current variability.
enum class CurrentSpectrumType {
  kNone,           ///< No variability — constant current
  kGeneric,        ///< Gaussian-in-log-f parametric spectrum
  kInternalWave,   ///< Garrett-Munk f^-2 in the internal wave band
  kBimodal,        ///< Two peaks: tidal/inertial + internal wave
  kTimeSeries,     ///< User-supplied time series (from file or external)
};

/// Current profile type (depth dependence).
enum class CurrentProfileType {
  kUniform,        ///< Same speed at all depths (simplest)
  kPowerLaw,       ///< U(z) = U_surface * (1 + z/d)^alpha  (z<0, d=depth)
  kBilinear,       ///< Two-layer: upper speed + lower speed with linear taper
  kCustom,         ///< User-supplied depth-speed table
};

/// Direction convention for current heading.
enum class CurrentDirectionConvention {
  kGoingTo,        ///< Oceanographic: direction current flows TOWARDS
  kComingFrom,     ///< Meteorological: direction current comes FROM (like wind)
};

// ============================================================
// Settings — Current Environment (shared)
// ============================================================

/// Current variability spectrum settings.
struct CurrentSpectrumSettings {
  CurrentSpectrumType type = CurrentSpectrumType::kGeneric;

  // --- Generic spectrum ---
  /// RMS current speed fluctuation [m/s].
  /// Typical North Sea values: 0.05–0.20 m/s.
  double sigma_uc = 0.10;

  /// Peak period of current variability [s].
  /// Typical values: 600–7200 s (10 min – 2 hr).
  /// Internal waves at Tampen: ~1800 s (30 min).
  double t_peak = 1800.0;

  /// Bandwidth parameter for the Gaussian-in-log-f spectrum [decades].
  /// Controls how broad/narrow the spectral peak is.
  /// Narrower (0.2) → nearly tonal; wider (0.6) → broad-banded.
  double sigma_log_f = 0.4;

  // --- Internal wave spectrum (Garrett-Munk) ---
  /// Latitude for inertial frequency computation [degrees N].
  double latitude = 61.2;  // Tampen, North Sea

  /// Buoyancy frequency [Hz].  If <= 0, uses default 0.005 Hz.
  double buoyancy_freq_hz = -1.0;

  // --- Bimodal spectrum ---
  /// Low-frequency (tidal/inertial) component.
  double sigma_tidal = 0.07;     ///< [m/s]
  double t_peak_tidal = 21600.0; ///< [s] (6 hours)

  /// High-frequency (internal wave / submesoscale) component.
  double sigma_iw = 0.07;      ///< [m/s]
  double t_peak_iw = 600.0;    ///< [s] (10 min)

  // --- Time series input ---
  /// Path to external current time series file (CSV: time_s, speed_m_s).
  /// Used only when type == kTimeSeries.
  std::string time_series_file;

  // --- Spectral synthesis ---
  /// Number of spectral components for time-domain synthesis.
  int n_components = 200;

  /// Minimum frequency for synthesis [Hz].
  /// Must be >= 1/simulation_duration to get at least one full cycle.
  /// If <= 0, auto-computed from simulation duration.
  double f_min_hz = -1.0;

  /// Maximum frequency for synthesis [Hz].
  /// Should be well above the surge natural frequency.
  double f_max_hz = 0.05;

  /// Use geometric (log-spaced) frequency grid for synthesis.
  /// Gives better resolution near the peak period.
  bool geometric_spacing = true;
};

/// Depth profile settings.
struct CurrentProfileSettings {
  CurrentProfileType type = CurrentProfileType::kUniform;

  // --- Power law ---
  /// Exponent for power-law profile: U(z) = U_s * ((d+z)/d)^alpha.
  /// Typical values: 1/7 (wind-driven), 1/10 (tidal).
  double power_law_exponent = 1.0 / 7.0;

  /// Reference depth for power-law profile [m] (positive downward).
  /// Velocity is zero at this depth.
  double power_law_reference_depth = 200.0;

  // --- Bilinear ---
  /// Upper layer speed fraction (1.0 = full surface speed).
  double upper_layer_fraction = 1.0;

  /// Lower layer speed fraction.
  double lower_layer_fraction = 0.3;

  /// Depth of transition between layers [m] (positive downward).
  double transition_depth = 50.0;

  /// Thickness of transition zone [m].
  double transition_thickness = 10.0;

  // --- Custom profile ---
  /// Depth-speed pairs: (depth_m, speed_fraction).
  /// Depth is positive downward; speed_fraction is relative to surface.
  /// Linearly interpolated between points.
  std::vector<std::pair<double, double>> custom_profile;
};

/// Settings for the shared current environment.
struct CurrentEnvironmentSettings {
  /// Enable/disable the entire current model.
  bool enabled = true;

  // --- Mean current ---
  /// Mean current speed at surface [m/s].
  double mean_speed = 0.50;

  /// Mean current direction [degrees].
  /// Interpretation depends on direction_convention.
  double mean_direction_deg = 0.0;

  /// Direction convention.
  CurrentDirectionConvention direction_convention =
      CurrentDirectionConvention::kGoingTo;

  /// Allow current direction to vary over time (not just speed).
  /// If false, only the speed magnitude varies; direction is constant.
  bool variable_direction = false;

  // --- Variability ---
  CurrentSpectrumSettings spectrum;

  // --- Depth profile ---
  CurrentProfileSettings profile;

  // --- Numerics ---
  /// Random seed for spectral synthesis.
  /// If 0, a random seed is generated from std::random_device.
  uint64_t seed = 0;

  /// Low-pass filter cutoff period [s] for the synthesized current.
  /// Removes energy above the surge bandwidth to avoid aliasing.
  /// If <= 0, no filtering (let the spectral synthesis handle it).
  double lowpass_cutoff_period = 0.0;
};

// ============================================================
// CurrentEnvironment — shared environmental current state
// ============================================================

/// Provides time-varying ocean current velocity for all bodies in
/// a simulation.  Owns the spectral synthesis and time-domain
/// realization.  Both the DP vessel and the floating platform
/// query this object for current velocity.
///
/// Usage (at simulation setup):
///   auto env = std::make_shared<CurrentEnvironment>(settings);
///   env->Initialize(simulation_duration);
///   vessel_wrapper.SetCurrentEnvironment(env);
///   platform_sim.SetCurrentEnvironment(env);
///
/// Usage (each timestep, called by consumers):
///   auto [Ux, Uy] = env->Velocity(time);
///   double U_at_depth = env->SpeedAtDepth(time, 50.0);
///
/// Thread safety: Initialize() must complete before any calls to
/// Velocity/Speed.  After initialization, all query methods are
/// const and safe for concurrent reads.
class CurrentEnvironment {
 public:
  explicit CurrentEnvironment(const CurrentEnvironmentSettings& settings);

  /// Pre-compute the current velocity time series for the full
  /// simulation duration.  Must be called once before any queries.
  ///
  /// @param duration  Total simulation duration [s].
  void Initialize(double duration);

  // --- Velocity queries (const after Initialize) ---

  /// Surface current speed at time t [m/s] (mean + fluctuation).
  double Speed(double time) const;

  /// Current speed at a given depth [m/s].
  /// @param time   Simulation time [s].
  /// @param depth  Depth below SWL [m] (positive downward).
  double SpeedAtDepth(double time, double depth) const;

  /// Current direction at time t [degrees, going-to convention].
  double Direction(double time) const;

  /// Current velocity components at time t [m/s] at the surface.
  /// @return {Ux, Uy} in the global NED frame.
  ///         Ux = speed × cos(dir), Uy = speed × sin(dir).
  std::pair<double, double> Velocity(double time) const;

  /// Current velocity components at a given depth [m/s].
  /// @return {Ux, Uy} in the global NED frame.
  std::pair<double, double> VelocityAtDepth(double time,
                                             double depth) const;

  /// Mean current speed [m/s] (time-invariant component).
  double MeanSpeed() const { return settings_.mean_speed; }

  /// Mean current direction [degrees, going-to].
  double MeanDirection() const;

  /// Whether the environment is enabled and initialized.
  bool IsActive() const { return settings_.enabled && initialized_; }

  /// Get the variability spectrum (for diagnostics/logging).
  /// @return Pair of vectors: {frequencies_hz, spectral_density}.
  std::pair<std::vector<double>, std::vector<double>>
  GetSpectrum() const;

  /// Access settings (read-only).
  const CurrentEnvironmentSettings& Settings() const { return settings_; }

  /// Apply depth profile factor at a given depth.
  /// @param depth  Depth below SWL [m] (positive downward).
  /// @return Fraction of surface speed at the given depth (0–1).
  double DepthFactor(double depth) const;

 private:
  CurrentEnvironmentSettings settings_;
  bool initialized_ = false;

  // Pre-computed time series
  std::vector<double> time_grid_;     ///< Time points [s]
  std::vector<double> speed_series_;  ///< Surface current speed(t) [m/s]
  std::vector<double> dir_series_;    ///< Current direction(t) [deg]

  // Pre-computed spectrum (for diagnostics)
  std::vector<double> freq_hz_;
  std::vector<double> spectrum_;

  // Random state
  std::mt19937_64 rng_;

  /// Synthesize the current speed time series from the spectrum.
  void SynthesizeTimeSeries(double duration);

  /// Load time series from external file.
  void LoadTimeSeries();

  /// Interpolate the pre-computed speed at an arbitrary time.
  double InterpolateSpeed(double time) const;

  /// Interpolate the pre-computed direction at an arbitrary time.
  double InterpolateDirection(double time) const;
};

// ============================================================
// Settings — Current Force Model (per-body)
// ============================================================

/// Per-body settings for how a specific hull interacts with current.
struct CurrentForceSettings {
  /// Global drag coefficient multiplier for this body.
  /// Applied on top of the per-section Cd values in the hull geometry.
  double cd_multiplier = 1.0;

  /// Use relative-velocity Morison formulation.
  /// If true: F = CdA × (U_current - U_body) × |U_current - U_body|.
  /// If false: F = CdA × U_current × |U_current| (ignores body velocity).
  /// Relative velocity provides additional hydrodynamic damping.
  bool relative_velocity = true;

  /// Use fully nonlinear drag (|U+u'|×(U+u')) each timestep.
  /// If false, linearize about mean: F ≈ F_mean + dF/dU × u'.
  /// Nonlinear automatically captures rectification; linearized needs
  /// an explicit correction.
  bool nonlinear_drag = true;

  /// Include rectification effect (only used when nonlinear_drag=false).
  /// Adds 0.5 × d²F/dU² × σ_Uc² / K to the mean offset.
  bool include_rectification = true;
};

// ============================================================
// CurrentForceModel — per-body drag force from shared environment
// ============================================================

namespace floating_platform {

/// Computes Morison drag force on a specific submerged hull from
/// the shared current environment.
///
/// Each body (FWT spar, DP vessel hull, etc.) has its own instance
/// with its own hull geometry and drag settings, but they all query
/// the same CurrentEnvironment for the current velocity field.
///
/// Usage:
///   // Setup
///   auto env = std::make_shared<CurrentEnvironment>(env_settings);
///   CurrentForceModel force(force_settings, hull_sections, env);
///
///   // Each timestep
///   auto [Fx, Fy] = force.ComputeForce(time,
///       platform_surge_vel, platform_sway_vel);
///
/// For the DP vessel, the existing VesselSimulatorWrapper can query
/// CurrentEnvironment directly and feed it into its speed-through-water
/// computation — no need for a CurrentForceModel wrapper.
class CurrentForceModel {
 public:
  /// Construct with hull geometry and a reference to the shared
  /// current environment.
  ///
  /// @param settings    Per-body drag settings.
  /// @param hull        Hull sections for Morison integration
  ///                    (from PlatformModel::hull_sections).
  /// @param environment Shared current environment (must outlive this).
  CurrentForceModel(
      const CurrentForceSettings& settings,
      const std::vector<HullSection>& hull,
      std::shared_ptr<const CurrentEnvironment> environment);

  /// Compute drag force at the current timestep.
  ///
  /// @param time            Simulation time [s].
  /// @param body_surge_vel  Body surge velocity in NED [m/s].
  /// @param body_sway_vel   Body sway velocity in NED [m/s].
  /// @param body_heading    Body heading [rad] (for resolving current
  ///                        into body-frame surge/sway components).
  /// @return {Fx, Fy} drag forces in the global NED frame [N].
  ///         Positive Fx = force in the current direction.
  std::pair<double, double> ComputeForce(
      double time, double body_surge_vel = 0.0,
      double body_sway_vel = 0.0, double body_heading = 0.0) const;

  /// Compute drag force with depth integration (using the environment's
  /// depth profile).  More accurate than uniform-speed approximation
  /// for deep-draft structures with non-uniform current profiles.
  ///
  /// @param time            Simulation time [s].
  /// @param body_surge_vel  Body surge velocity [m/s].
  /// @param body_sway_vel   Body sway velocity [m/s].
  /// @param body_heading    Body heading [rad].
  /// @return {Fx, Fy} drag forces in NED [N].
  std::pair<double, double> ComputeForceDepthIntegrated(
      double time, double body_surge_vel = 0.0,
      double body_sway_vel = 0.0, double body_heading = 0.0) const;

  // --- Linearized quantities (for spectral analysis / initialization) ---

  /// Mean drag force [N] from mean current on this hull.
  double MeanDragForce() const;

  /// Linearized drag sensitivity dF/dU [N/(m/s)] at mean current.
  double DragSensitivity() const;

  /// Second derivative d²F/dU² [N/(m/s)²] at mean current.
  double DragSecondDerivative() const;

  /// Rectification mean offset increase [m].
  /// @param k_tangent  Tangent mooring stiffness at mean offset [N/m].
  double RectificationOffset(double k_tangent) const;

  /// Access the underlying drag calculator.
  const CurrentDrag& Drag() const { return drag_; }

  /// Access the shared environment.
  const CurrentEnvironment& Environment() const { return *env_; }

 private:
  CurrentForceSettings settings_;
  CurrentDrag drag_;
  std::shared_ptr<const CurrentEnvironment> env_;
};

}  // namespace floating_platform
}  // namespace simulator
}  // namespace brucon
