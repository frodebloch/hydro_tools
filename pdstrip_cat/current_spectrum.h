// current_spectrum.h — Current variability spectrum models
//
// Generates one-sided power spectral densities S(f) [m²/s² / Hz] for
// ocean current velocity fluctuations.  Three models:
//
//   1. Generic:  Gaussian-in-log-f parametric spectrum
//   2. Internal wave:  Garrett-Munk f^-2 continuum
//   3. Bimodal:  tidal/inertial peak + internal wave peak
//
// All spectra are normalized so that ∫S(f)df = σ² (target variance).
//
// Reference: pdstrip_cat/slow_drift_swell.py, lines 392–528

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace brucon {
namespace simulator {

// ============================================================
// Free functions — spectrum generation
// ============================================================

/// Generate a logarithmically spaced frequency grid.
///
/// @param f_min   Minimum frequency [Hz] (must be > 0).
/// @param f_max   Maximum frequency [Hz].
/// @param n       Number of points.
/// @return        Vector of n frequencies, geometrically spaced.
inline std::vector<double> MakeLogFrequencies(double f_min, double f_max,
                                               int n) {
  std::vector<double> f(n);
  double log_min = std::log10(f_min);
  double log_max = std::log10(f_max);
  double d_log = (log_max - log_min) / (n - 1);
  for (int i = 0; i < n; ++i) {
    f[i] = std::pow(10.0, log_min + i * d_log);
  }
  return f;
}

/// Generate a linearly spaced frequency grid.
inline std::vector<double> MakeLinearFrequencies(double f_min, double f_max,
                                                  int n) {
  std::vector<double> f(n);
  double df = (f_max - f_min) / (n - 1);
  for (int i = 0; i < n; ++i) {
    f[i] = f_min + i * df;
  }
  return f;
}

/// Trapezoidal integration of y(x).
inline double Trapz(const std::vector<double>& y,
                    const std::vector<double>& x) {
  double sum = 0.0;
  for (size_t i = 1; i < x.size(); ++i) {
    sum += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
  }
  return sum;
}

// ============================================================
// Generic parametric spectrum — Gaussian in log-frequency
// ============================================================

/// Compute a generic single-peaked current variability spectrum.
///
/// Models the current velocity fluctuation as a Gaussian peak in
/// log-frequency space, centered at f_peak = 1/T_peak.  The shape
/// resembles typical ocean current spectra from ADCP measurements.
///
/// @param f           Frequency grid [Hz] (must be > 0).
/// @param sigma_uc    Target RMS current speed fluctuation [m/s].
/// @param t_peak      Peak period of variability [s].
/// @param sigma_log_f Width of the peak in decades (default 0.4).
///                    Smaller → narrower peak, larger → broader band.
/// @return S(f) spectral density [m²/s² / Hz].
///         Integrates to sigma_uc².
inline std::vector<double> GenericCurrentSpectrum(
    const std::vector<double>& f, double sigma_uc, double t_peak,
    double sigma_log_f = 0.4) {
  const int n = static_cast<int>(f.size());
  std::vector<double> S(n, 0.0);

  double f_peak = 1.0 / t_peak;

  // Unnormalized Gaussian in log-f
  for (int i = 0; i < n; ++i) {
    if (f[i] > 0.0) {
      double log_ratio = std::log10(f[i] / f_peak);
      S[i] = std::exp(-0.5 * (log_ratio / sigma_log_f) *
                       (log_ratio / sigma_log_f));
    }
  }

  // Normalize to target variance
  double var_raw = Trapz(S, f);
  if (var_raw > 0.0) {
    double scale = sigma_uc * sigma_uc / var_raw;
    for (auto& s : S) s *= scale;
  }

  return S;
}

// ============================================================
// Internal wave spectrum — Garrett-Munk f^-2
// ============================================================

/// Compute an internal-wave current variability spectrum.
///
/// Models the Garrett-Munk-like continuum of internal wave energy
/// producing current fluctuations in the band between the inertial
/// frequency f_i and the buoyancy frequency f_N.  The spectrum
/// follows the canonical f^(-2) shape.
///
/// @param f              Frequency grid [Hz].
/// @param sigma_uc       Target RMS fluctuation [m/s].
/// @param latitude_deg   Latitude for inertial frequency [degrees N].
/// @param buoyancy_freq  Buoyancy frequency [Hz].  If <= 0, uses 0.005 Hz.
/// @return S(f) [m²/s² / Hz], integrates to sigma_uc².
inline std::vector<double> InternalWaveCurrentSpectrum(
    const std::vector<double>& f, double sigma_uc,
    double latitude_deg = 61.2, double buoyancy_freq = -1.0) {
  const int n = static_cast<int>(f.size());
  std::vector<double> S(n, 0.0);

  // Inertial frequency
  constexpr double omega_earth = 7.2921e-5;  // rad/s
  double f_inertial =
      2.0 * omega_earth * std::abs(std::sin(latitude_deg * M_PI / 180.0)) /
      (2.0 * M_PI);

  // Buoyancy frequency
  double f_N = (buoyancy_freq > 0.0) ? buoyancy_freq : 0.005;

  // GM-like f^-2 in the internal wave band
  for (int i = 0; i < n; ++i) {
    if (f[i] > f_inertial && f[i] < f_N) {
      S[i] = std::pow(f[i] / f_inertial, -2.0);
    }
  }

  // Normalize
  double var_raw = Trapz(S, f);
  if (var_raw > 0.0) {
    double scale = sigma_uc * sigma_uc / var_raw;
    for (auto& s : S) s *= scale;
  }

  return S;
}

// ============================================================
// Bimodal spectrum — tidal/inertial + internal wave
// ============================================================

/// Compute a bimodal current spectrum combining a tidal/inertial
/// low-frequency peak with a higher-frequency internal wave peak.
///
/// This is the most physically realistic simple model for a site
/// like Tampen where both processes contribute.
///
/// @param f            Frequency grid [Hz].
/// @param sigma_tidal  RMS tidal/inertial fluctuation [m/s].
/// @param t_tidal      Period of tidal/inertial peak [s].
/// @param sigma_iw     RMS internal wave fluctuation [m/s].
/// @param t_iw         Period of internal wave peak [s].
/// @param sigma_log_f  Bandwidth parameter for both peaks [decades].
/// @return S(f) [m²/s² / Hz], integrates to (sigma_tidal² + sigma_iw²).
inline std::vector<double> BimodalCurrentSpectrum(
    const std::vector<double>& f, double sigma_tidal, double t_tidal,
    double sigma_iw, double t_iw, double sigma_log_f = 0.4) {
  auto S1 = GenericCurrentSpectrum(f, sigma_tidal, t_tidal, sigma_log_f);
  auto S2 = GenericCurrentSpectrum(f, sigma_iw, t_iw, sigma_log_f);

  std::vector<double> S(f.size());
  for (size_t i = 0; i < f.size(); ++i) {
    S[i] = S1[i] + S2[i];
  }
  return S;
}

// ============================================================
// Spectral synthesis — time-domain realization
// ============================================================

/// Synthesize a zero-mean random time series from a one-sided PSD.
///
/// Uses the random-phase method:
///   x(t) = Σ_k sqrt(2 * S(f_k) * Δf_k) * cos(2π f_k t + φ_k)
///
/// where φ_k are independent uniform random phases in [0, 2π).
///
/// @param freq     Frequencies [Hz].
/// @param spectrum One-sided PSD S(f) [unit²/Hz].
/// @param times    Time points at which to evaluate [s].
/// @param rng      Random number generator (modified in place).
/// @return         Time series values at each time point.
inline std::vector<double> SynthesizeFromSpectrum(
    const std::vector<double>& freq, const std::vector<double>& spectrum,
    const std::vector<double>& times, std::mt19937_64& rng) {
  const int n_f = static_cast<int>(freq.size());
  const int n_t = static_cast<int>(times.size());
  std::vector<double> x(n_t, 0.0);

  std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);

  // Pre-compute amplitudes and phases
  std::vector<double> amp(n_f);
  std::vector<double> phi(n_f);
  for (int k = 0; k < n_f; ++k) {
    // Frequency bandwidth (geometric or linear)
    double df;
    if (k == 0) {
      df = (n_f > 1) ? freq[1] - freq[0] : 1.0;
    } else if (k == n_f - 1) {
      df = freq[n_f - 1] - freq[n_f - 2];
    } else {
      df = 0.5 * (freq[k + 1] - freq[k - 1]);
    }
    amp[k] = std::sqrt(2.0 * spectrum[k] * df);
    phi[k] = phase_dist(rng);
  }

  // Superposition
  for (int k = 0; k < n_f; ++k) {
    if (amp[k] <= 0.0) continue;
    double omega_k = 2.0 * M_PI * freq[k];
    for (int j = 0; j < n_t; ++j) {
      x[j] += amp[k] * std::cos(omega_k * times[j] + phi[k]);
    }
  }

  return x;
}

}  // namespace simulator
}  // namespace brucon
