// Transit propulsion power study for the cable lay vessel propulsion comparison
// (config_cablelay_azi vs config_cablelay_shaft).
//
// Not a unit test - a study driver. Run from build/bin:
//   ./CablelayTransitStudy config_cablelay_azi   csov_pdstrip.dat cablelay_study/transit_azi   0.97
//   ./CablelayTransitStudy config_cablelay_shaft csov_pdstrip.dat cablelay_study/transit_shaft 0.99
// Arguments: <config> <pdstrip> <prefix> <transmission efficiency> <thrust deduction t> <wake fraction w>
// Diameter and max power are read from the config (main propellers / azimuths with thrust_power_data or
// all_speed_data). The propeller model is independent of the (simple) DP model in the config.
//
// The two main propulsors (azimuths or main propellers with all_speed_data) are modelled as FIXED PITCH
// ducted propellers (Ka / D4-55 series), 2 units sharing the thrust equally, with two design philosophies:
//   A (bollard design):  P/D chosen for max bollard thrust at max power; nominal rpm absorbs max power at zero
//                        speed. In transit the propeller is light, motor overspeed (constant power) up to
//                        kOverspeed x nominal rpm.
//   B (transit design):  P/D chosen for max top speed at max power (calm water + kDesignSeaMargin); nominal rpm
//                        absorbs max power at that speed. At low speed the propeller is torque limited
//                        (motor torque = max power / nominal rpm).
// The config max_rpm is NOT used: nominal propeller rpm is a design result (motor / gear matched to it).
// Limits for both: power <= max power, torque <= max power / (2 pi n_nominal), rpm <= rpm limit.
//
// Resistance: ITTC-57 friction + (1+k) form factor + Holtrop-Mennen wave making (VesselResistance) + appendage
// drag (Holtrop: 0.5 rho V^2 S (1+k2) Cf) + tunnel opening drag (Holtrop: rho V^2 pi d^2 C_BTO).
// Conditions: calm water, and Bf 5 head weather (wind 10.7 m/s + wind sea Hs 2.1 m / Tp 7.5 s from ahead).

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "brucon/dp/thruster_model/d4_55.h"
#include "brucon/dp/vessel_model/response_function.h"
#include "brucon/dp/vessel_model/wave_response.h"
#include "brucon/dp/vessel_model/wave_spectrum.h"
#include "brucon/dp/vessel_model/wind_force_model.h"
#include "brucon/propulsion_optimiser_simulator/vessel_resistance.h"
#include "brucon/proto_utils/proto_utils.h"
#include "brucon/proto_utils/vessel_section_config.h"
#include "config_tree.pb.h"

using namespace brucon::dp;
using brucon::propulsion_optimiser::VesselResistance;

namespace {

constexpr double kKnot = 0.514444;
constexpr double kRho = 1025.0;
constexpr double kNu = 1.19e-6;
constexpr double kPi = 3.14159265358979;
constexpr double kOverspeed = 1.2;   // Design A: allowed motor overspeed (constant power)
constexpr double kFormFactor = 1.3;  // (1+k) hull form factor
constexpr double kDesignSeaMargin = 0.15;  // Design B: resistance margin at the design point

// ----- Appendages (estimates, edit as needed) -----
struct Appendage {
  std::string name;
  double wetted_area;  // m2 (total for all units)
  double one_plus_k2;  // Holtrop appendage form factor
};
struct Tunnel {
  std::string name;
  double diameter;  // m
  int count;
};
constexpr double kCbto = 0.006;  // Holtrop tunnel opening coefficient (0.003 - 0.012)

struct AppendageSet {
  std::vector<Appendage> appendages;
  std::vector<Tunnel> tunnels;
};

AppendageSet AppendagesFor(bool shaft_line) {
  AppendageSet a;
  a.tunnels.push_back({"Bow tunnels", 2.6, 2});
  if (shaft_line) {
    a.appendages.push_back({"Rudders (2 x 6.9 m2, both sides)", 2 * 2 * 6.9, 1.5});
    a.appendages.push_back({"Shafts (2 x 8 m exposed, D0.45)", 2 * kPi * 0.45 * 8.0, 2.0});
    a.appendages.push_back({"Shaft brackets (2 x 4 m2)", 2 * 4.0, 3.0});
    a.tunnels.push_back({"Stern tunnels", 1.5, 2});
  } else {
    a.appendages.push_back({"Azimuth struts (2 x 7.2 m2)", 2 * 7.2, 3.0});
    a.appendages.push_back({"Azimuth gear housings (2 x 11 m2)", 2 * 11.0, 2.0});
  }
  return a;
}

double Cf(double V, double L) {
  if (V < 0.01) return 0.0;
  const double re = V * L / kNu;
  return 0.075 / std::pow(std::log10(re) - 2.0, 2);
}

// ----- Propeller -----
struct Propeller {
  double diameter, max_power, n_nominal, wake, thrust_deduction, eta_r;
  bool nozzle;
};

struct OperatingPoint {
  bool feasible;
  double n, thrust, power, torque, eta0;
  std::string limit;
};

class FixedPitchPropeller {
 public:
  FixedPitchPropeller(const Propeller& p, double pitch, double n_nominal, double n_limit)
      : p_(p), series_(pitch, p.diameter), pitch_(pitch), n_nominal_(n_nominal), n_limit_(n_limit),
        q_max_(p.max_power / (2.0 * kPi * n_nominal)) {}

  // Operating point for required thrust T [kN] at advance speed Va [m/s]
  OperatingPoint ForThrust(double T, double Va) const {
    OperatingPoint op{};
    op.thrust = T;
    op.n = series_.N(pitch_, Va, T);
    op.power = series_.Power(pitch_, op.n, Va) / p_.eta_r;
    op.torque = op.power / (2.0 * kPi * op.n);
    op.eta0 = op.power > 0 ? T * Va / op.power : 0.0;
    op.feasible = true;
    if (op.n > n_limit_ * 1.0001) {
      op.feasible = false;
      op.limit = "rpm";
    } else if (op.power > p_.max_power * 1.0001) {
      op.feasible = false;
      op.limit = "power";
    } else if (op.torque > q_max_ * 1.0001) {
      op.feasible = false;
      op.limit = "torque";
    }
    return op;
  }

  // Max thrust at advance speed Va within rpm, power and torque limits
  OperatingPoint MaxThrust(double Va) const {
    double lo = 0.0, hi = n_limit_;
    for (int i = 0; i < 40; ++i) {
      double mid = 0.5 * (lo + hi);
      double P = series_.Power(pitch_, mid, Va) / p_.eta_r;
      bool ok = P <= p_.max_power && P / (2 * kPi * mid) <= q_max_;
      (ok ? lo : hi) = mid;
    }
    OperatingPoint op = ForThrust(series_.Thrust(pitch_, lo, Va), Va);
    op.limit = lo > n_limit_ * 0.999 ? "rpm" : (op.power > p_.max_power * 0.999 ? "power" : "torque");
    return op;
  }
  double pitch() const { return pitch_; }
  double n_nominal() const { return n_nominal_; }
  const CdSeries& series() const { return series_; }

 private:
  Propeller p_;
  D455 series_;
  double pitch_, n_nominal_, n_limit_, q_max_;
};

class Transit {
 public:
  Transit(const brucon::configuration::ConfigTree& cfg, const std::string& pdstrip, double eta_trans,
          double thrust_deduction, double wake)
      : eta_trans_(eta_trans) {
    const auto& vd = cfg.vessel_data();
    auto [off, br, dr, ar] = VesselSectionConfig(vd.sections()).SectionData();
    // Main propulsors: first two with all_speed_data
    bool shaft = false;
    for (const auto& p : cfg.propulsors().propulsor()) {
      const brucon::configuration::ThrusterDetails* d = nullptr;
      if (p.has_main_propeller()) {
        d = &p.main_propeller().details();
        shaft = true;
      } else if (p.has_azimuth() && p.azimuth().details().propeller_diameter() > 3.0) {  // main azimuths
        d = &p.azimuth().details();
      }
      if (d && !prop_set_ && (d->has_all_speed_data() || d->has_thrust_power_data())) {
        const double pmax = d->has_all_speed_data() ? d->all_speed_data().max_power()
                                                    : d->thrust_power_data().max_power_positive_thrust();
        prop_ = {d->propeller_diameter(), pmax, 0.0, wake, thrust_deduction, 1.0, d->nozzle()};
        prop_set_ = true;
      }
    }
    shaft_line_ = shaft;
    res_ = std::make_unique<VesselResistance>(off, br, dr, ar, vd.lpp(), vd.breadth(), vd.design_draft(), kFormFactor,
                                              VesselResistance::PropulsionFactors{prop_.wake, prop_.thrust_deduction});
    lpp_ = vd.lpp();
    app_ = AppendagesFor(shaft_line_);

    const auto& wd = cfg.vessel_wind_data();
    wind_ = std::make_unique<WindForceModel>(WindForceModel({static_cast<ShipType>(wd.ship_type()), vd.loa(),
                                                             wd.lateral_wind_area(), wd.frontal_wind_area(),
                                                             wd.lateral_wind_area_centre(),
                                                             wd.height_of_wind_area_centre()}));
    ResponseFunctions rf;
    if (rf.ReadExportedPdStripDatFile("vessel_simulator_config/" + pdstrip) > 0) {
      wave_ = std::make_unique<WaveResponse>(std::move(rf));
      wave_->set_calculate_drift_forces(true);
      spectrum_.SetSpectrumType(kBretschneider);  // DNV-ST-0111 (two-parameter PM)
      wave_->AddWaveSpectrum(&spectrum_);
    }
  }

  bool shaft_line() const { return shaft_line_; }
  const Propeller& prop() const { return prop_; }
  const VesselResistance& resistance() const { return *res_; }

  double AppendageResistance(double V) const {
    double r = 0.0;
    const double cf = Cf(V, lpp_);
    for (const auto& a : app_.appendages) r += 0.5 * kRho * V * V * a.wetted_area * a.one_plus_k2 * cf;
    for (const auto& t : app_.tunnels) r += t.count * kRho * V * V * kPi * t.diameter * t.diameter * kCbto;
    return r / 1000.0;
  }

  // Total resistance [kN]. weather = false: calm, true: Bf 5 head weather.
  double Resistance(double V, bool weather, double* r_calm = nullptr, double* r_app = nullptr,
                    double* r_weather = nullptr) {
    double calm = res_->CalmWaterResistance(V);
    double app = AppendageResistance(V);
    double wx = 0.0;
    if (weather) {
      res_->SetWindModel(wind_.get());
      if (wave_) {
        spectrum_.SetParameters({2.1, 0.0, 7.5, 2.0});
        res_->SetWaveResponse(wave_.get());
      }
      wx = res_->TotalResistance(V, 0.0, 10.7, 0.0) - calm;
      res_->SetWindModel(nullptr);
      res_->SetWaveResponse(nullptr);
    }
    if (r_calm) *r_calm = calm;
    if (r_app) *r_app = app;
    if (r_weather) *r_weather = wx;
    return calm + app + wx;
  }

  double ThrustPerProp(double V, bool weather) { return Resistance(V, weather) / (1.0 - prop_.thrust_deduction) / 2.0; }
  double Va(double V) const { return V * (1.0 - prop_.wake); }

  // Top speed [m/s] for propeller, limited by its rpm / power / torque limits
  double TopSpeed(const FixedPitchPropeller& fp, bool weather) {
    double lo = 0.0, hi = 12.0;
    for (int i = 0; i < 40; ++i) {
      double mid = 0.5 * (lo + hi);
      (fp.MaxThrust(Va(mid)).thrust >= ThrustPerProp(mid, weather) ? lo : hi) = mid;
    }
    return lo;
  }

  struct DesignResult {
    double pitch, n_nominal, design_speed;
  };

  // Design A: P/D for max bollard thrust at max power; nominal rpm absorbs max power at zero speed
  DesignResult BollardDesign() const {
    DesignResult best{0, 0, 0};
    double best_t = 0.0;
    for (double pd = 0.5; pd <= 1.4001; pd += 0.01) {
      D455 s(pd, prop_.diameter);
      double n = s.NGivenPower(pd, 0.0, prop_.max_power * prop_.eta_r);
      double t = s.Thrust(pd, n, 0.0);
      if (t > best_t) {
        best_t = t;
        best = {pd, n, 0.0};
      }
    }
    return best;
  }

  // Design B: P/D for max speed at max power (calm + sea margin); nominal rpm absorbs max power at that speed
  DesignResult TransitDesign() {
    DesignResult best{0, 0, 0};
    for (double pd = 0.5; pd <= 1.4001; pd += 0.01) {
      D455 s(pd, prop_.diameter);
      double lo = 0.0, hi = 12.0;
      for (int k = 0; k < 40; ++k) {
        double v = 0.5 * (lo + hi);
        double T = (1.0 + kDesignSeaMargin) * ThrustPerProp(v, false);
        double n = s.N(pd, Va(v), T);
        (s.Power(pd, n, Va(v)) / prop_.eta_r < prop_.max_power ? lo : hi) = v;
      }
      if (lo > best.design_speed) {
        double T = (1.0 + kDesignSeaMargin) * ThrustPerProp(lo, false);
        best = {pd, s.N(pd, Va(lo), T), lo};
      }
    }
    return best;
  }

  double eta_trans() const { return eta_trans_; }
  const AppendageSet& appendages() const { return app_; }

 private:
  double eta_trans_;
  Propeller prop_{};
  bool prop_set_ = false;
  bool shaft_line_ = false;
  double lpp_ = 0.0;
  AppendageSet app_;
  std::unique_ptr<VesselResistance> res_;
  std::unique_ptr<WindForceModel> wind_;
  std::unique_ptr<WaveResponse> wave_;
  WaveSpectrum spectrum_{0.0, 5.0, 0.0, 2.0};
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0]
              << " <config_dir> <pdstrip file> <output prefix> <transmission efficiency> <thrust deduction> <wake>\n";
    return 1;
  }
  brucon::configuration::ConfigTree cfg;
  if (!brucon::proto_utils::ParseConfigTree(cfg, argv[1])) {
    std::cerr << "Failed to parse " << argv[1] << "\n";
    return 1;
  }
  const std::string prefix = argv[3];
  Transit tr(cfg, argv[2], std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6]));
  const auto& p = tr.prop();

  std::cout << std::fixed << std::setprecision(3) << argv[1] << (tr.shaft_line() ? " (shaft line)" : " (azimuth)")
            << "\n  Propeller: D " << p.diameter << " m, " << (p.nozzle ? "ducted" : "open") << ", max power "
            << std::setprecision(0) << p.max_power << " kW, w "
            << std::setprecision(2) << p.wake << ", t " << p.thrust_deduction << ", eta_r " << p.eta_r
            << ", eta_trans " << tr.eta_trans() << "\n  Hull: S " << std::setprecision(0)
            << tr.resistance().hull_params().wetted_surface_m2 << " m2, disp "
            << tr.resistance().hull_params().volume_displacement_m3 << " m3, (1+k) " << std::setprecision(2)
            << kFormFactor << "\n  Appendages:";
  for (const auto& a : tr.appendages().appendages) std::cout << "\n    " << a.name << ", 1+k2 " << a.one_plus_k2;
  for (const auto& t : tr.appendages().tunnels)
    std::cout << "\n    " << t.count << " x " << t.name << " D" << t.diameter << ", C_BTO " << kCbto;
  std::cout << '\n';

  struct Design {
    std::string name;
    double pitch;
    double n_nominal;
    double n_limit;
  };
  const auto a = tr.BollardDesign();
  const auto b = tr.TransitDesign();
  const std::vector<Design> designs{{"A_bollard", a.pitch, a.n_nominal, kOverspeed * a.n_nominal},
                                    {"B_transit", b.pitch, b.n_nominal, b.n_nominal}};
  std::cout << "  Design B point: " << std::setprecision(1) << b.design_speed / kKnot << " kn (calm + "
            << std::setprecision(0) << 100 * kDesignSeaMargin << "% margin)\n";

  std::ofstream sum(prefix + "_summary.out");
  sum << "design\tP/D\tnominal_rpm\tbollard_T_per_prop(kN)\tbollard_P_per_prop(kW)\tbollard_limit\ttop_speed_calm(kn)"
         "\ttop_speed_bf5(kn)\n";
  for (const auto& d : designs) {
    FixedPitchPropeller fp(p, d.pitch, d.n_nominal, d.n_limit);
    auto bp = fp.MaxThrust(0.0);
    double v_calm = tr.TopSpeed(fp, false) / kKnot, v_bf5 = tr.TopSpeed(fp, true) / kKnot;
    std::cout << "\n  Design " << d.name << ": P/D " << std::setprecision(2) << d.pitch << ", nominal rpm "
              << std::setprecision(0) << d.n_nominal * 60 << ", rpm limit " << d.n_limit * 60 << ", bollard " << bp.thrust << " kN/prop at " << bp.power
              << " kW (" << bp.limit << " limited), top speed calm " << std::setprecision(1) << v_calm
              << " kn, Bf5 head " << v_bf5 << " kn\n";
    sum << d.name << '\t' << d.pitch << '\t' << d.n_nominal * 60 << '\t' << bp.thrust << '\t' << bp.power << '\t' << bp.limit << '\t' << v_calm
        << '\t' << v_bf5 << '\n';

    for (bool weather : {false, true}) {
      const std::string cond = weather ? "Bf5head" : "calm";
      std::ofstream out(prefix + "_" + d.name + "_" + cond + ".out");
      out << "speed(kn)\tR_calm(kN)\tR_app(kN)\tR_weather(kN)\tR_total(kN)\tT_per_prop(kN)\tVa(m/s)\trpm\trpm%"
             "\tP_prop_per_prop(kW)\tP_total_motor(kW)\teta0\teta_D\tfeasible\tlimit\n";
      std::cout << "    " << cond << ":  kn   R_tot   T/prop   rpm  rpm%  P/prop  P_motor_total  eta0  etaD\n";
      for (double kn = 8.0; kn <= 12.001; kn += 0.5) {
        const double V = kn * kKnot;
        double rc, ra, rw;
        double R = tr.Resistance(V, weather, &rc, &ra, &rw);
        double T = R / (1.0 - p.thrust_deduction) / 2.0;
        auto op = fp.ForThrust(T, tr.Va(V));
        double p_motor = 2.0 * op.power / tr.eta_trans();
        double eta_d = R * V / (2.0 * op.power);
        out << kn << '\t' << rc << '\t' << ra << '\t' << rw << '\t' << R << '\t' << T << '\t' << tr.Va(V) << '\t'
            << op.n * 60 << '\t' << 100 * op.n / d.n_nominal << '\t' << op.power << '\t' << p_motor << '\t' << op.eta0
            << '\t' << eta_d << '\t' << op.feasible << '\t' << (op.feasible ? "-" : op.limit) << '\n';
        std::cout << "          " << std::setprecision(1) << std::setw(5) << kn << std::setprecision(0)
                  << std::setw(7) << R << std::setw(8) << T << std::setw(7) << op.n * 60 << std::setw(5)
                  << 100 * op.n / d.n_nominal << std::setw(8) << op.power << std::setw(10) << p_motor
                  << std::setprecision(3) << std::setw(10) << op.eta0 << std::setw(6) << eta_d
                  << (op.feasible ? "" : "  NOT FEASIBLE (" + op.limit + ")") << '\n';
      }
    }
  }
  return 0;
}
