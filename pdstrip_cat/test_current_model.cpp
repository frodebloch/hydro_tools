// test_current_model.cpp — Validation tests for current model
//
// Compares C++ outputs against Python prototype (slow_drift_swell.py).
// Compile: g++ -std=c++17 -O2 -o test_current_model test_current_model.cpp current_model.cpp -lm
// Run: ./test_current_model

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "current_drag.h"
#include "current_model.h"
#include "current_spectrum.h"

using namespace brucon::simulator;

static int n_tests = 0;
static int n_pass = 0;
static int n_fail = 0;

static void check(const char* name, double actual, double expected,
                   double tol_pct) {
  ++n_tests;
  double err_pct;
  bool pass;
  if (std::abs(expected) < 1e-15) {
    // For zero expected, check absolute difference
    err_pct = std::abs(actual) * 100.0;  // treat as percentage of 1
    pass = std::abs(actual) < 1e-10;
    err_pct = pass ? 0.0 : err_pct;
  } else {
    err_pct = std::abs(actual - expected) / std::abs(expected) * 100.0;
    pass = err_pct <= tol_pct;
  }
  if (pass) {
    ++n_pass;
    std::printf("  PASS  %-45s %12.2f  (expected %12.2f, err %.2f%%)\n",
                name, actual, expected, err_pct);
  } else {
    ++n_fail;
    std::printf("  FAIL  %-45s %12.2f  (expected %12.2f, err %.2f%% > %.1f%%)\n",
                name, actual, expected, err_pct, tol_pct);
  }
}

// ============================================================
// Test 1: Static drag force on OC3 Hywind hull
// ============================================================
void test_static_drag() {
  std::printf("\n--- Test 1: Static drag force ---\n");
  auto hull = OC3HywindHull(1.05, 8);
  CurrentDrag drag(hull, 1025.0, 1.0);

  // Python reference: F(0.5) = 148095.4 N
  double F05 = drag.Force(0.5);
  check("Force(0.5) [N]", F05, 148095.4, 1.0);

  // Check CdA_eff = F / U^2
  double CdA = drag.EffectiveDragArea();
  check("CdA_eff [N/(m/s)^2]", CdA, 592381.5, 1.0);

  // Check F = CdA * U|U| at various speeds
  double F10 = drag.Force(1.0);
  check("Force(1.0) [N]", F10, CdA * 1.0, 0.01);

  double F03 = drag.Force(0.3);
  check("Force(0.3) [N]", F03, CdA * 0.09, 0.01);
}

// ============================================================
// Test 2: Linearized drag sensitivity
// ============================================================
void test_linearized_sensitivity() {
  std::printf("\n--- Test 2: Linearized drag sensitivity ---\n");
  auto hull = OC3HywindHull(1.05, 8);
  CurrentDrag drag(hull, 1025.0, 1.0);

  // Python reference: dF/dU(0.5) = 592381.5 N/(m/s)
  double dFdU = drag.Sensitivity(0.5);
  check("Sensitivity(0.5) [N/(m/s)]", dFdU, 592381.5, 1.0);

  // Python reference: d2F/dU2 = 1184762.9 N/(m/s)^2
  double d2F = drag.SecondDerivative();
  check("SecondDerivative [N/(m/s)^2]", d2F, 1184762.9, 1.0);

  // Analytical: dF/dU = 2*CdA*U, d2F/dU2 = 2*CdA
  double CdA = drag.EffectiveDragArea();
  check("dF/dU analytical", dFdU, 2.0 * CdA * 0.5, 0.01);
  check("d2F/dU2 analytical", d2F, 2.0 * CdA, 0.01);

  // Numerical sensitivity should agree closely
  auto [dFdU_num, d2F_num] = drag.NumericalSensitivity(0.5);
  check("Numerical dF/dU", dFdU_num, dFdU, 0.1);
  check("Numerical d2F/dU2", d2F_num, d2F, 0.5);
}

// ============================================================
// Test 3: Hull geometry (total submerged length, section count)
// ============================================================
void test_hull_geometry() {
  std::printf("\n--- Test 3: Hull geometry ---\n");
  auto hull = OC3HywindHull(1.05, 8);

  // Total submerged length = 120m
  double L = 0.0;
  for (const auto& s : hull) L += std::abs(s.z_top - s.z_bottom);
  check("Total submerged length [m]", L, 120.0, 0.01);

  // Number of sections: 1 upper + 8 taper + 1 lower = 10
  check("Number of sections", static_cast<double>(hull.size()), 10.0, 0.01);

  // Check upper section
  check("Upper D [m]", hull.front().diameter, 6.5, 0.01);
  check("Upper z_top [m]", hull.front().z_top, 0.0, 0.01);
  check("Upper z_bottom [m]", hull.front().z_bottom, -4.0, 0.01);

  // Check lower section
  check("Lower D [m]", hull.back().diameter, 9.4, 0.01);
  check("Lower z_top [m]", hull.back().z_top, -12.0, 0.01);
  check("Lower z_bottom [m]", hull.back().z_bottom, -120.0, 0.01);
}

// ============================================================
// Test 4: Generic current spectrum shape and normalization
// ============================================================
void test_generic_spectrum() {
  std::printf("\n--- Test 4: Generic current spectrum ---\n");

  double sigma_uc = 0.10;
  double T_peak = 1800.0;
  double f_peak = 1.0 / T_peak;

  auto f = MakeLogFrequencies(1e-5, 0.1, 4000);
  auto S = GenericCurrentSpectrum(f, sigma_uc, T_peak, 0.4);

  // Integral should equal sigma^2 = 0.01
  double var = Trapz(S, f);
  check("Spectrum integral (sigma^2)", var, sigma_uc * sigma_uc, 1.0);

  // Peak should be near f_peak = 1/1800
  int i_peak = 0;
  double S_max = 0.0;
  for (size_t i = 0; i < f.size(); ++i) {
    if (S[i] > S_max) {
      S_max = S[i];
      i_peak = static_cast<int>(i);
    }
  }
  check("Peak frequency [Hz]", f[i_peak], f_peak, 5.0);
}

// ============================================================
// Test 5: Internal wave spectrum
// ============================================================
void test_internal_wave_spectrum() {
  std::printf("\n--- Test 5: Internal wave spectrum ---\n");

  double sigma_uc = 0.10;
  auto f = MakeLogFrequencies(1e-5, 0.01, 2000);
  auto S = InternalWaveCurrentSpectrum(f, sigma_uc, 61.2, -1.0);

  double var = Trapz(S, f);
  check("IW spectrum integral", var, sigma_uc * sigma_uc, 2.0);
}

// ============================================================
// Test 6: Bimodal spectrum
// ============================================================
void test_bimodal_spectrum() {
  std::printf("\n--- Test 6: Bimodal spectrum ---\n");

  double sigma_tidal = 0.07;
  double T_tidal = 21600.0;
  double sigma_iw = 0.07;
  double T_iw = 600.0;

  auto f = MakeLogFrequencies(1e-6, 0.1, 4000);
  auto S = BimodalCurrentSpectrum(f, sigma_tidal, T_tidal, sigma_iw, T_iw);

  // Total variance should be sigma_tidal^2 + sigma_iw^2
  double var_expected = sigma_tidal * sigma_tidal + sigma_iw * sigma_iw;
  double var = Trapz(S, f);
  check("Bimodal spectrum integral", var, var_expected, 2.0);
}

// ============================================================
// Test 7: Time-domain synthesis statistics
// ============================================================
void test_synthesis_statistics() {
  std::printf("\n--- Test 7: Synthesis statistics ---\n");

  double sigma_uc = 0.10;
  double T_peak = 1800.0;
  double duration = 3.0 * 3600.0;  // 3 hours

  auto f = MakeLogFrequencies(1.0 / duration, 0.05, 200);
  auto S = GenericCurrentSpectrum(f, sigma_uc, T_peak, 0.4);

  // Time grid
  double dt = 5.0;
  int n_time = static_cast<int>(duration / dt) + 1;
  std::vector<double> times(n_time);
  for (int i = 0; i < n_time; ++i) times[i] = i * dt;

  std::mt19937_64 rng(42);
  auto x = SynthesizeFromSpectrum(f, S, times, rng);

  // Compute mean and std
  double mean = 0.0;
  for (auto v : x) mean += v;
  mean /= x.size();

  double var = 0.0;
  for (auto v : x) var += (v - mean) * (v - mean);
  var /= (x.size() - 1);
  double sigma_actual = std::sqrt(var);

  // Mean should be near zero (random phases)
  std::printf("  INFO  Synthesis mean = %.6f (should be ~0)\n", mean);
  std::printf("  INFO  Synthesis sigma = %.4f (target = %.4f)\n",
              sigma_actual, sigma_uc);

  // Sigma should be within ~15% of target (finite sample, 3h of broadband)
  check("Synthesis sigma_uc", sigma_actual, sigma_uc, 15.0);
}

// ============================================================
// Test 8: CurrentEnvironment integration
// ============================================================
void test_current_environment() {
  std::printf("\n--- Test 8: CurrentEnvironment ---\n");

  CurrentEnvironmentSettings settings;
  settings.mean_speed = 0.50;
  settings.mean_direction_deg = 0.0;
  settings.spectrum.type = CurrentSpectrumType::kGeneric;
  settings.spectrum.sigma_uc = 0.10;
  settings.spectrum.t_peak = 1800.0;
  settings.spectrum.n_components = 200;
  settings.spectrum.geometric_spacing = true;
  settings.seed = 42;

  double duration = 3.0 * 3600.0;

  CurrentEnvironment env(settings);
  env.Initialize(duration);

  // Check that it's active
  check("IsActive", env.IsActive() ? 1.0 : 0.0, 1.0, 0.01);

  // Speed at t=0 should be near mean (but not exactly, due to synthesis)
  double U0 = env.Speed(0.0);
  std::printf("  INFO  Speed(0) = %.4f m/s (mean = %.4f)\n", U0,
              settings.mean_speed);

  // Sample many timesteps and check statistics
  int n_samples = 1000;
  double dt = duration / n_samples;
  double sum = 0.0, sum2 = 0.0;
  for (int i = 0; i < n_samples; ++i) {
    double U = env.Speed(i * dt);
    sum += U;
    sum2 += U * U;
  }
  double mean_U = sum / n_samples;
  double var_U = sum2 / n_samples - mean_U * mean_U;
  double sigma_U = std::sqrt(std::max(0.0, var_U));

  // Mean should be close to settings.mean_speed
  // (slightly higher due to max(0, ...) clipping and rectification)
  check("Env mean speed [m/s]", mean_U, settings.mean_speed, 10.0);

  // Sigma of sampled speeds should be close to sigma_uc
  check("Env sigma speed [m/s]", sigma_U, settings.spectrum.sigma_uc, 20.0);

  // Direction should be constant at 0
  double dir = env.Direction(1000.0);
  check("Direction at t=1000 [deg]", dir, 0.0, 0.01);

  // Velocity components: Ux = speed, Uy = 0 for dir=0
  auto [Ux, Uy] = env.Velocity(0.0);
  check("Velocity Ux(0)", Ux, U0, 0.01);
  // Uy should be ~0 but not exactly due to floating point of sin(0)
  std::printf("  INFO  Velocity Uy(0) = %.2e (should be ~0)\n", Uy);

  // Depth profile: uniform by default
  check("DepthFactor(0)", env.DepthFactor(0.0), 1.0, 0.01);
  check("DepthFactor(60)", env.DepthFactor(60.0), 1.0, 0.01);
  check("DepthFactor(120)", env.DepthFactor(120.0), 1.0, 0.01);

  // Test spectrum retrieval
  auto [freq, spec] = env.GetSpectrum();
  check("Spectrum size", static_cast<double>(freq.size()), 200.0, 0.01);
}

// ============================================================
// Test 9: CurrentForceModel
// ============================================================
void test_current_force_model() {
  std::printf("\n--- Test 9: CurrentForceModel ---\n");

  using namespace floating_platform;

  // Setup environment
  CurrentEnvironmentSettings env_settings;
  env_settings.mean_speed = 0.50;
  env_settings.mean_direction_deg = 0.0;
  env_settings.spectrum.type = CurrentSpectrumType::kNone;  // constant current
  env_settings.seed = 42;

  auto env = std::make_shared<CurrentEnvironment>(env_settings);
  env->Initialize(3600.0);

  // Setup force model
  CurrentForceSettings force_settings;
  force_settings.cd_multiplier = 1.0;
  force_settings.relative_velocity = true;
  force_settings.nonlinear_drag = true;

  auto hull = OC3HywindHull(1.05, 8);
  CurrentForceModel model(force_settings, hull, env);

  // With constant current at 0 deg and zero body velocity:
  // Force should be entirely in X (surge) direction
  auto [Fx, Fy] = model.ComputeForce(0.0, 0.0, 0.0, 0.0);

  // Fx should equal Force(0.5) = ~148 kN
  check("ForceModel Fx [N]", Fx, 148095.4, 1.5);
  // Fy should be ~0
  std::printf("  INFO  ForceModel Fy = %.2e N (should be ~0)\n", Fy);

  // Mean drag force
  check("MeanDragForce [N]", model.MeanDragForce(), 148095.4, 1.5);

  // Sensitivity
  check("DragSensitivity [N/(m/s)]", model.DragSensitivity(), 592381.5, 1.0);

  // Test with body velocity: ForceRelative(0.5, 0.1) should be less
  auto [Fx_rel, Fy_rel] = model.ComputeForce(0.0, 0.1, 0.0, 0.0);
  std::printf("  INFO  Fx with body_vel=0.1: %.1f N (< %.1f N)\n",
              Fx_rel, Fx);
  check("Force with relative velocity < absolute", Fx_rel < Fx ? 1.0 : 0.0,
        1.0, 0.01);

  // Test with rotated heading: current at 0 deg, body heading at 90 deg
  // Current should appear as sway force
  auto [Fx_rot, Fy_rot] = model.ComputeForce(0.0, 0.0, 0.0, M_PI / 2.0);
  std::printf("  INFO  Heading=90deg: Fx=%.1f Fy=%.1f (expect Fx≈148kN, Fy≈0)\n",
              Fx_rot, Fy_rot);
  // In NED frame, force should still be in X direction regardless of heading
  check("Rotated Fx [N]", Fx_rot, 148095.4, 2.0);
}

// ============================================================
// Test 10: Depth profile
// ============================================================
void test_depth_profile() {
  std::printf("\n--- Test 10: Depth profile ---\n");

  // Power law
  CurrentEnvironmentSettings settings;
  settings.profile.type = CurrentProfileType::kPowerLaw;
  settings.profile.power_law_exponent = 1.0 / 7.0;
  settings.profile.power_law_reference_depth = 200.0;
  settings.spectrum.type = CurrentSpectrumType::kNone;

  CurrentEnvironment env(settings);
  env.Initialize(3600.0);

  // At surface (depth=0): factor = 1.0
  check("PowerLaw factor(0)", env.DepthFactor(0.0), 1.0, 0.01);

  // At mid-depth (100m): factor = ((200-100)/200)^(1/7) = 0.5^0.143 ≈ 0.906
  double f100 = std::pow(0.5, 1.0 / 7.0);
  check("PowerLaw factor(100)", env.DepthFactor(100.0), f100, 0.1);

  // At reference depth: factor = 0
  check("PowerLaw factor(200)", env.DepthFactor(200.0), 0.0, 0.01);

  // Bilinear profile
  CurrentEnvironmentSettings bl_settings;
  bl_settings.profile.type = CurrentProfileType::kBilinear;
  bl_settings.profile.transition_depth = 50.0;
  bl_settings.profile.transition_thickness = 10.0;
  bl_settings.profile.upper_layer_fraction = 1.0;
  bl_settings.profile.lower_layer_fraction = 0.3;
  bl_settings.spectrum.type = CurrentSpectrumType::kNone;

  CurrentEnvironment bl_env(bl_settings);
  bl_env.Initialize(3600.0);

  check("Bilinear factor(0)", bl_env.DepthFactor(0.0), 1.0, 0.01);
  check("Bilinear factor(30)", bl_env.DepthFactor(30.0), 1.0, 0.01);
  check("Bilinear factor(50)", bl_env.DepthFactor(50.0), 0.65, 1.0);
  check("Bilinear factor(60)", bl_env.DepthFactor(60.0), 0.3, 0.01);
  check("Bilinear factor(100)", bl_env.DepthFactor(100.0), 0.3, 0.01);
}

// ============================================================
// Test 11: Both bodies see the same current
// ============================================================
void test_correlation() {
  std::printf("\n--- Test 11: Correlated current (shared env) ---\n");

  using namespace floating_platform;

  CurrentEnvironmentSettings env_settings;
  env_settings.mean_speed = 0.50;
  env_settings.spectrum.type = CurrentSpectrumType::kGeneric;
  env_settings.spectrum.sigma_uc = 0.10;
  env_settings.spectrum.t_peak = 1800.0;
  env_settings.seed = 123;

  auto env = std::make_shared<CurrentEnvironment>(env_settings);
  env->Initialize(3.0 * 3600.0);

  // Two "bodies" querying the same environment at the same time
  // should get identical velocities
  double t_test = 5432.0;
  double U1 = env->Speed(t_test);
  double U2 = env->Speed(t_test);
  check("Correlation: same time, same speed", U1, U2, 0.0001);

  auto [Ux1, Uy1] = env->Velocity(t_test);
  auto [Ux2, Uy2] = env->Velocity(t_test);
  check("Correlation: Ux match", Ux1, Ux2, 0.0001);
}

// ============================================================
// Main
// ============================================================
int main() {
  std::printf("=== Current Model Validation Tests ===\n");
  std::printf("Comparing C++ implementation against Python prototype\n");

  test_static_drag();
  test_linearized_sensitivity();
  test_hull_geometry();
  test_generic_spectrum();
  test_internal_wave_spectrum();
  test_bimodal_spectrum();
  test_synthesis_statistics();
  test_current_environment();
  test_current_force_model();
  test_depth_profile();
  test_correlation();

  std::printf("\n=== Results: %d/%d passed, %d failed ===\n",
              n_pass, n_tests, n_fail);

  return (n_fail > 0) ? 1 : 0;
}
