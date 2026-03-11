// current_drag.h — Morison drag force on submerged hull from ocean current
//
// Integrates quadratic drag F = 0.5 * rho * Cd * D * U|U| over the
// hull depth, using the same section geometry as MorisonDamping.
//
// Supports:
//   - Multiple hull sections (cylinder + taper + cylinder for OC3 Hywind)
//   - Relative-velocity formulation (current - platform velocity)
//   - Depth-dependent current profile
//   - Linearized sensitivity (dF/dU, d²F/dU²) for spectral analysis
//
// Reference: pdstrip_cat/slow_drift_swell.py, lines 314–389

#pragma once

#include <cmath>
#include <vector>

namespace brucon {
namespace simulator {
namespace floating_platform {

// ============================================================
// Hull section geometry (shared with MorisonDamping)
// ============================================================

/// A cylindrical hull section for Morison drag integration.
/// The hull is described as a stack of sections from the waterline
/// down to the keel.  Each section has a constant diameter and Cd.
///
/// Coordinate system: z = 0 at SWL, z < 0 below water.
/// Depth values z_top and z_bottom are negative (below SWL).
struct HullSection {
  double z_top;      ///< Top of section [m] (z <= 0, e.g., 0.0 for waterline)
  double z_bottom;   ///< Bottom of section [m] (z < z_top, e.g., -120.0)
  double diameter;   ///< Outer diameter [m]
  double cd;         ///< Drag coefficient (typically 1.0–1.2 for cylinder)

  /// Projected frontal area of this section [m²].
  double FrontalArea() const {
    return diameter * std::abs(z_top - z_bottom);
  }
};

/// Pre-computed OC3 Hywind spar geometry.
///
/// Returns the standard three-section spar hull:
///   - Upper column:  0 to -4 m,    D = 6.5 m,  Cd = 1.05
///   - Taper:        -4 to -12 m,   linearly 6.5 → 9.4 m
///   - Lower column: -12 to -120 m, D = 9.4 m,  Cd = 1.05
///
/// The taper is discretized into n_taper sub-sections.
///
/// @param cd  Drag coefficient (applied to all sections).
/// @param n_taper  Number of sub-sections for the taper region.
inline std::vector<HullSection> OC3HywindHull(double cd = 1.05,
                                               int n_taper = 8) {
  std::vector<HullSection> hull;

  // Upper column: 0 to -4 m
  hull.push_back({0.0, -4.0, 6.5, cd});

  // Taper: -4 to -12 m, linearly from D=6.5 to D=9.4
  double dz_taper = 8.0 / n_taper;  // total 8m / n segments
  for (int i = 0; i < n_taper; ++i) {
    double z_t = -4.0 - i * dz_taper;
    double z_b = z_t - dz_taper;
    double frac_mid = ((-z_t - dz_taper / 2.0) - 4.0) / 8.0;
    double d_mid = 6.5 + frac_mid * (9.4 - 6.5);
    hull.push_back({z_t, z_b, d_mid, cd});
  }

  // Lower column: -12 to -120 m
  hull.push_back({-12.0, -120.0, 9.4, cd});

  return hull;
}

// ============================================================
// CurrentDrag — Morison drag integration
// ============================================================

/// Computes drag force on a submerged hull from ocean current.
///
/// The force is integrated over the hull depth using:
///   dF = 0.5 * rho * Cd * D(z) * U_rel(z) * |U_rel(z)| * dz
///
/// where U_rel = U_current - U_platform is the relative velocity
/// (for the nonlinear formulation), or just U_current if the
/// platform velocity is ignored.
///
/// This class also provides linearized drag sensitivity for
/// spectral-domain analysis:
///   dF/dU = 2 * CdA_eff * U_mean
///   d²F/dU² = 2 * CdA_eff
/// where CdA_eff = 0.5 * rho * Σ(Cd_i * D_i * L_i) is the
/// effective drag area summed over all sections.
class CurrentDrag {
 public:
  /// Default constructor (no hull geometry — must call SetHull).
  CurrentDrag() = default;

  /// Construct with hull geometry.
  ///
  /// @param hull_sections  Hull section definitions.
  /// @param rho_water      Water density [kg/m³].
  /// @param cd_multiplier  Global Cd scaling factor (default 1.0).
  CurrentDrag(const std::vector<HullSection>& hull_sections,
              double rho_water = 1025.0, double cd_multiplier = 1.0)
      : hull_(hull_sections),
        rho_(rho_water),
        cd_mult_(cd_multiplier) {
    ComputeEffectiveDragArea();
  }

  /// Set or replace the hull geometry.
  void SetHull(const std::vector<HullSection>& hull_sections,
               double rho_water = 1025.0, double cd_multiplier = 1.0) {
    hull_ = hull_sections;
    rho_ = rho_water;
    cd_mult_ = cd_multiplier;
    ComputeEffectiveDragArea();
  }

  // ---- Force computation (nonlinear, per timestep) ----

  /// Compute total drag force for a uniform current speed.
  ///
  /// @param U_current  Current speed [m/s] (positive = in flow direction).
  /// @return Drag force [N] (positive = in flow direction).
  double Force(double U_current) const {
    return CdA_eff_ * U_current * std::abs(U_current);
  }

  /// Compute drag force with relative velocity.
  ///
  /// @param U_current   Current speed [m/s].
  /// @param U_platform  Platform speed in current direction [m/s].
  /// @return Drag force [N] on the platform.
  ///         Positive = pushes platform in current direction.
  double ForceRelative(double U_current, double U_platform) const {
    double U_rel = U_current - U_platform;
    return CdA_eff_ * U_rel * std::abs(U_rel);
  }

  /// Compute drag force with a depth-dependent current profile.
  ///
  /// @param U_surface  Current speed at surface [m/s].
  /// @param depth_factor  Function returning the speed ratio U(z)/U_surface
  ///                       at a given depth z (z < 0).
  /// @param U_platform   Platform velocity [m/s] (uniform with depth,
  ///                     or set to 0 for absolute current drag).
  /// @return Total drag force [N].
  ///
  /// The integration is section-by-section using the midpoint depth
  /// of each hull section.
  template <typename DepthFunc>
  double ForceWithProfile(double U_surface, DepthFunc depth_factor,
                          double U_platform = 0.0) const {
    double F_total = 0.0;
    for (const auto& s : hull_) {
      double z_mid = 0.5 * (s.z_top + s.z_bottom);
      double L = std::abs(s.z_top - s.z_bottom);
      double U_local = U_surface * depth_factor(z_mid) - U_platform;
      double dF = 0.5 * rho_ * s.cd * cd_mult_ * s.diameter * L *
                  U_local * std::abs(U_local);
      F_total += dF;
    }
    return F_total;
  }

  // ---- Linearized quantities (for spectral analysis) ----

  /// Mean drag force at a given mean current speed [N].
  double MeanForce(double U_mean) const {
    return Force(U_mean);
  }

  /// Linearized drag sensitivity dF/dU at mean speed [N/(m/s)].
  /// For F = CdA * U|U|: dF/dU = 2 * CdA * |U|.
  double Sensitivity(double U_mean) const {
    return 2.0 * CdA_eff_ * std::abs(U_mean);
  }

  /// Second derivative d²F/dU² [N/(m/s)²].
  /// For F = CdA * U|U|: d²F/dU² = 2 * CdA (constant for U > 0).
  double SecondDerivative() const { return 2.0 * CdA_eff_; }

  /// Effective drag area CdA = 0.5 * rho * Σ(Cd * D * L) [N/(m/s)²].
  /// This is the coefficient in F = CdA * U|U|.
  double EffectiveDragArea() const { return CdA_eff_; }

  /// Rectification mean offset increase [m].
  ///
  /// When the current fluctuates with zero-mean σ_uc about a positive
  /// mean, the quadratic drag produces a net positive force bias:
  ///   ΔF = 0.5 * d²F/dU² * σ_uc²
  /// which shifts the equilibrium by Δx = ΔF / K_mooring.
  ///
  /// @param sigma_uc   RMS current fluctuation [m/s].
  /// @param k_mooring  Tangent mooring stiffness [N/m].
  double RectificationOffset(double sigma_uc, double k_mooring) const {
    double d2F = SecondDerivative();
    double delta_F = 0.5 * d2F * sigma_uc * sigma_uc;
    return delta_F / k_mooring;
  }

  /// Numerical drag sensitivity (central differences).
  /// More accurate than the analytical formula when the hull has
  /// complex geometry (tapers, varying Cd).
  ///
  /// @param U_mean  Mean current speed [m/s].
  /// @param eps     Perturbation size [m/s].
  /// @return {dF/dU, d²F/dU²}.
  std::pair<double, double> NumericalSensitivity(double U_mean,
                                                  double eps = 0.01) const {
    double F0 = Force(U_mean);
    double Fp = Force(U_mean + eps);
    double Fm = Force(std::max(0.0, U_mean - eps));
    double dFdU = (U_mean > eps) ? (Fp - Fm) / (2.0 * eps) : Fp / eps;
    double d2FdU2 =
        (U_mean > eps) ? (Fp - 2.0 * F0 + Fm) / (eps * eps)
                       : (Fp - F0) / (eps * eps);
    return {dFdU, d2FdU2};
  }

  /// Access hull sections.
  const std::vector<HullSection>& Hull() const { return hull_; }

  /// Total submerged length of all sections [m].
  double TotalSubmergedLength() const {
    double L = 0.0;
    for (const auto& s : hull_) {
      L += std::abs(s.z_top - s.z_bottom);
    }
    return L;
  }

 private:
  std::vector<HullSection> hull_;
  double rho_ = 1025.0;
  double cd_mult_ = 1.0;
  double CdA_eff_ = 0.0;  ///< Pre-computed effective drag area

  /// Pre-compute the effective drag area from hull geometry.
  void ComputeEffectiveDragArea() {
    CdA_eff_ = 0.0;
    for (const auto& s : hull_) {
      double L = std::abs(s.z_top - s.z_bottom);
      CdA_eff_ += 0.5 * rho_ * s.cd * cd_mult_ * s.diameter * L;
    }
  }
};

}  // namespace floating_platform
}  // namespace simulator
}  // namespace brucon
