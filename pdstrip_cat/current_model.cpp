// current_model.cpp — Implementation of CurrentEnvironment and CurrentForceModel
//
// See current_model.h for interface documentation.
// Python prototype: pdstrip_cat/slow_drift_swell.py, lines 314–690

#include "current_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brucon {
namespace simulator {

// ============================================================
// CurrentEnvironment
// ============================================================

CurrentEnvironment::CurrentEnvironment(
    const CurrentEnvironmentSettings& settings)
    : settings_(settings) {
  // Initialize RNG
  if (settings_.seed != 0) {
    rng_.seed(settings_.seed);
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }
}

void CurrentEnvironment::Initialize(double duration) {
  if (!settings_.enabled) {
    initialized_ = true;
    return;
  }
  if (duration <= 0.0) {
    throw std::invalid_argument(
        "CurrentEnvironment::Initialize: duration must be > 0");
  }

  if (settings_.spectrum.type == CurrentSpectrumType::kTimeSeries) {
    LoadTimeSeries();
  } else if (settings_.spectrum.type != CurrentSpectrumType::kNone) {
    SynthesizeTimeSeries(duration);
  }
  // kNone: no time series needed, Speed() returns mean_speed directly.

  initialized_ = true;
}

// --- Velocity queries ---

double CurrentEnvironment::Speed(double time) const {
  assert(initialized_);
  if (!settings_.enabled) return 0.0;
  if (settings_.spectrum.type == CurrentSpectrumType::kNone ||
      speed_series_.empty()) {
    return settings_.mean_speed;
  }
  return InterpolateSpeed(time);
}

double CurrentEnvironment::SpeedAtDepth(double time, double depth) const {
  return Speed(time) * DepthFactor(depth);
}

double CurrentEnvironment::Direction(double time) const {
  assert(initialized_);
  if (!settings_.enabled) return 0.0;
  if (!settings_.variable_direction || dir_series_.empty()) {
    // Convert from configured convention to going-to
    if (settings_.direction_convention ==
        CurrentDirectionConvention::kComingFrom) {
      double dir = settings_.mean_direction_deg + 180.0;
      if (dir >= 360.0) dir -= 360.0;
      return dir;
    }
    return settings_.mean_direction_deg;
  }
  return InterpolateDirection(time);
}

std::pair<double, double> CurrentEnvironment::Velocity(double time) const {
  double spd = Speed(time);
  double dir_rad = Direction(time) * M_PI / 180.0;
  return {spd * std::cos(dir_rad), spd * std::sin(dir_rad)};
}

std::pair<double, double> CurrentEnvironment::VelocityAtDepth(
    double time, double depth) const {
  double spd = SpeedAtDepth(time, depth);
  double dir_rad = Direction(time) * M_PI / 180.0;
  return {spd * std::cos(dir_rad), spd * std::sin(dir_rad)};
}

double CurrentEnvironment::MeanDirection() const {
  if (settings_.direction_convention ==
      CurrentDirectionConvention::kComingFrom) {
    double dir = settings_.mean_direction_deg + 180.0;
    if (dir >= 360.0) dir -= 360.0;
    return dir;
  }
  return settings_.mean_direction_deg;
}

std::pair<std::vector<double>, std::vector<double>>
CurrentEnvironment::GetSpectrum() const {
  return {freq_hz_, spectrum_};
}

// --- Private methods ---

void CurrentEnvironment::SynthesizeTimeSeries(double duration) {
  const auto& sp = settings_.spectrum;

  // Frequency grid
  double f_min = sp.f_min_hz;
  if (f_min <= 0.0) {
    // Auto: lowest frequency = 1 full cycle in the simulation duration
    f_min = 1.0 / duration;
  }
  double f_max = sp.f_max_hz;
  int n_comp = sp.n_components;

  if (sp.geometric_spacing) {
    freq_hz_ = MakeLogFrequencies(f_min, f_max, n_comp);
  } else {
    freq_hz_ = MakeLinearFrequencies(f_min, f_max, n_comp);
  }

  // Compute the spectrum
  switch (sp.type) {
    case CurrentSpectrumType::kGeneric:
      spectrum_ = GenericCurrentSpectrum(freq_hz_, sp.sigma_uc, sp.t_peak,
                                         sp.sigma_log_f);
      break;
    case CurrentSpectrumType::kInternalWave:
      spectrum_ = InternalWaveCurrentSpectrum(freq_hz_, sp.sigma_uc,
                                               sp.latitude,
                                               sp.buoyancy_freq_hz);
      break;
    case CurrentSpectrumType::kBimodal:
      spectrum_ = BimodalCurrentSpectrum(freq_hz_, sp.sigma_tidal,
                                          sp.t_peak_tidal, sp.sigma_iw,
                                          sp.t_peak_iw, sp.sigma_log_f);
      break;
    default:
      // kNone or kTimeSeries should not reach here
      return;
  }

  // Time grid: dt chosen so the highest frequency is well resolved.
  // Nyquist: dt <= 1/(2*f_max).  Use dt = 1/(4*f_max) for safety.
  double dt = 1.0 / (4.0 * f_max);
  // But also cap at a reasonable step size for long simulations.
  // For 3-hour sim with f_max=0.05 Hz, dt=5s → 2160 points.  Fine.
  int n_time = static_cast<int>(std::ceil(duration / dt)) + 1;
  time_grid_.resize(n_time);
  for (int i = 0; i < n_time; ++i) {
    time_grid_[i] = i * dt;
  }

  // Synthesize zero-mean fluctuation
  std::vector<double> u_fluct =
      SynthesizeFromSpectrum(freq_hz_, spectrum_, time_grid_, rng_);

  // Build speed series: mean + fluctuation
  // Ensure speed stays non-negative (physical constraint).
  speed_series_.resize(n_time);
  for (int i = 0; i < n_time; ++i) {
    speed_series_[i] = std::max(0.0, settings_.mean_speed + u_fluct[i]);
  }

  // Direction: constant unless variable_direction is enabled.
  // For variable direction, we'd synthesize a second independent series
  // for directional fluctuation.  Not implemented yet — direction is
  // constant at the mean value.
  // (Future: could add a second spectrum for directional spreading.)
}

void CurrentEnvironment::LoadTimeSeries() {
  const std::string& path = settings_.spectrum.time_series_file;
  if (path.empty()) {
    throw std::runtime_error(
        "CurrentEnvironment: time_series_file path is empty");
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "CurrentEnvironment: cannot open time series file: " + path);
  }

  time_grid_.clear();
  speed_series_.clear();

  std::string line;
  while (std::getline(file, line)) {
    // Skip comments and empty lines
    if (line.empty() || line[0] == '#' || line[0] == '%') continue;

    std::istringstream iss(line);
    double t, u;
    char comma;
    // Try CSV format first (time, speed)
    if (line.find(',') != std::string::npos) {
      iss >> t >> comma >> u;
    } else {
      // Whitespace-separated
      iss >> t >> u;
    }
    if (!iss.fail()) {
      time_grid_.push_back(t);
      speed_series_.push_back(u);
    }
  }

  if (time_grid_.size() < 2) {
    throw std::runtime_error(
        "CurrentEnvironment: time series file has fewer than 2 points: " +
        path);
  }
}

double CurrentEnvironment::InterpolateSpeed(double time) const {
  // Linear interpolation on pre-computed time series.
  // Clamp to boundaries.
  if (time_grid_.empty()) return settings_.mean_speed;
  if (time <= time_grid_.front()) return speed_series_.front();
  if (time >= time_grid_.back()) return speed_series_.back();

  // Binary search for the interval containing 'time'
  auto it = std::lower_bound(time_grid_.begin(), time_grid_.end(), time);
  size_t i = static_cast<size_t>(it - time_grid_.begin());
  if (i == 0) i = 1;

  double t0 = time_grid_[i - 1];
  double t1 = time_grid_[i];
  double alpha = (time - t0) / (t1 - t0);
  return speed_series_[i - 1] + alpha * (speed_series_[i] - speed_series_[i - 1]);
}

double CurrentEnvironment::InterpolateDirection(double time) const {
  if (dir_series_.empty()) return MeanDirection();
  if (time <= time_grid_.front()) return dir_series_.front();
  if (time >= time_grid_.back()) return dir_series_.back();

  auto it = std::lower_bound(time_grid_.begin(), time_grid_.end(), time);
  size_t i = static_cast<size_t>(it - time_grid_.begin());
  if (i == 0) i = 1;

  double t0 = time_grid_[i - 1];
  double t1 = time_grid_[i];
  double alpha = (time - t0) / (t1 - t0);

  // Interpolate direction with wrapping (handle 350° → 10° correctly)
  double d0 = dir_series_[i - 1];
  double d1 = dir_series_[i];
  double diff = d1 - d0;
  if (diff > 180.0) diff -= 360.0;
  if (diff < -180.0) diff += 360.0;
  double dir = d0 + alpha * diff;
  if (dir < 0.0) dir += 360.0;
  if (dir >= 360.0) dir -= 360.0;
  return dir;
}

double CurrentEnvironment::DepthFactor(double depth) const {
  // depth is positive downward (convention from the API).
  // Internally, the profile functions work with positive depth.
  double z = depth;  // positive downward

  switch (settings_.profile.type) {
    case CurrentProfileType::kUniform:
      return 1.0;

    case CurrentProfileType::kPowerLaw: {
      double d = settings_.profile.power_law_reference_depth;
      double alpha = settings_.profile.power_law_exponent;
      if (z >= d) return 0.0;  // below reference depth
      return std::pow((d - z) / d, alpha);
    }

    case CurrentProfileType::kBilinear: {
      double z_trans = settings_.profile.transition_depth;
      double dz = settings_.profile.transition_thickness;
      double f_upper = settings_.profile.upper_layer_fraction;
      double f_lower = settings_.profile.lower_layer_fraction;
      if (z <= z_trans - dz / 2.0) return f_upper;
      if (z >= z_trans + dz / 2.0) return f_lower;
      // Linear taper in transition zone
      double frac = (z - (z_trans - dz / 2.0)) / dz;
      return f_upper + frac * (f_lower - f_upper);
    }

    case CurrentProfileType::kCustom: {
      const auto& prof = settings_.profile.custom_profile;
      if (prof.empty()) return 1.0;
      if (z <= prof.front().first) return prof.front().second;
      if (z >= prof.back().first) return prof.back().second;
      // Linear interpolation
      for (size_t i = 1; i < prof.size(); ++i) {
        if (z <= prof[i].first) {
          double frac = (z - prof[i - 1].first) /
                        (prof[i].first - prof[i - 1].first);
          return prof[i - 1].second + frac * (prof[i].second - prof[i - 1].second);
        }
      }
      return prof.back().second;
    }
  }
  return 1.0;  // fallback
}

// ============================================================
// CurrentForceModel
// ============================================================

namespace floating_platform {

CurrentForceModel::CurrentForceModel(
    const CurrentForceSettings& settings,
    const std::vector<HullSection>& hull,
    std::shared_ptr<const CurrentEnvironment> environment)
    : settings_(settings),
      drag_(hull, 1025.0, settings.cd_multiplier),
      env_(std::move(environment)) {}

std::pair<double, double> CurrentForceModel::ComputeForce(
    double time, double body_surge_vel, double body_sway_vel,
    double body_heading) const {
  if (!env_ || !env_->IsActive()) return {0.0, 0.0};

  auto [Ux, Uy] = env_->Velocity(time);

  // Resolve current into body-frame surge/sway
  double cos_h = std::cos(body_heading);
  double sin_h = std::sin(body_heading);
  double U_surge = Ux * cos_h + Uy * sin_h;   // current in body surge direction
  double U_sway = -Ux * sin_h + Uy * cos_h;   // current in body sway direction

  double Fx_body, Fy_body;
  if (settings_.relative_velocity) {
    Fx_body = drag_.ForceRelative(U_surge, body_surge_vel);
    Fy_body = drag_.ForceRelative(U_sway, body_sway_vel);
  } else {
    Fx_body = drag_.Force(U_surge);
    Fy_body = drag_.Force(U_sway);
  }

  // Rotate back to NED frame
  double Fx_ned = Fx_body * cos_h - Fy_body * sin_h;
  double Fy_ned = Fx_body * sin_h + Fy_body * cos_h;

  return {Fx_ned, Fy_ned};
}

std::pair<double, double> CurrentForceModel::ComputeForceDepthIntegrated(
    double time, double body_surge_vel, double body_sway_vel,
    double body_heading) const {
  if (!env_ || !env_->IsActive()) return {0.0, 0.0};

  double dir_rad = env_->Direction(time) * M_PI / 180.0;
  double cos_h = std::cos(body_heading);
  double sin_h = std::sin(body_heading);

  // Depth-integrated force using ForceWithProfile
  // The depth_factor lambda queries the environment's depth profile
  // for the speed ratio at each depth.
  double U_surface = env_->Speed(time);

  // Surge component: project current direction into body surge
  double cos_delta = std::cos(dir_rad - body_heading);
  double sin_delta = std::sin(dir_rad - body_heading);
  double U_surge_surface = U_surface * cos_delta;
  double U_sway_surface = U_surface * sin_delta;

  // For proper depth integration, pass the environment's depth profile
  // through ForceWithProfile.
  auto profile = [this](double z_neg) -> double {
    // z_neg is negative (below SWL from HullSection convention).
    // DepthFactor expects positive-downward depth.
    double depth_pos = -z_neg;
    return env_->DepthFactor(depth_pos);
  };

  double Fx_body, Fy_body;
  if (settings_.relative_velocity) {
    Fx_body = drag_.ForceWithProfile(
        U_surge_surface, profile, body_surge_vel);
    Fy_body = drag_.ForceWithProfile(
        U_sway_surface, profile, body_sway_vel);
  } else {
    Fx_body = drag_.ForceWithProfile(U_surge_surface, profile);
    Fy_body = drag_.ForceWithProfile(U_sway_surface, profile);
  }

  // Rotate back to NED
  double Fx_ned = Fx_body * cos_h - Fy_body * sin_h;
  double Fy_ned = Fx_body * sin_h + Fy_body * cos_h;

  return {Fx_ned, Fy_ned};
}

double CurrentForceModel::MeanDragForce() const {
  if (!env_) return 0.0;
  return drag_.MeanForce(env_->MeanSpeed());
}

double CurrentForceModel::DragSensitivity() const {
  if (!env_) return 0.0;
  return drag_.Sensitivity(env_->MeanSpeed());
}

double CurrentForceModel::DragSecondDerivative() const {
  return drag_.SecondDerivative();
}

double CurrentForceModel::RectificationOffset(double k_tangent) const {
  if (!env_) return 0.0;
  double sigma_uc = env_->Settings().spectrum.sigma_uc;
  return drag_.RectificationOffset(sigma_uc, k_tangent);
}

}  // namespace floating_platform
}  // namespace simulator
}  // namespace brucon
