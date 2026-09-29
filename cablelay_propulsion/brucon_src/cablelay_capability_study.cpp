// Static DP capability study for the cable lay vessel propulsion comparison
// (config_cablelay_azi vs config_cablelay_shaft).
//
// Not a unit test - a study driver. Run from build/bin:
//   ./CablelayCapabilityStudy config_cablelay_azi   csov_pdstrip.dat cablelay_azi
//   ./CablelayCapabilityStudy config_cablelay_shaft csov_pdstrip.dat cablelay_shaft
//
// Operation modes:
//   DP   : Station keeping, zero speed, no external force.
//   LAY  : Ploughing / lay mode. 0.8 kn ahead through water and 80 t bollard pull (tow force acting astern
//          on the vessel centre line).
// Failure cases (from power_system.prototxt):
//   Intact : All buses connected (bus ties closed), one bus group with all generators + ESS.
//   Bus<k> : Loss of bus k. Thrusters (and connected rudders) on bus k lost. Remaining buses split into
//            separate bus groups (bus ties opened), each limited by its own generator / ESS capacity.
// Weather: DNV-ST-0111 Beaufort relation (wind, wind sea and current collinear, from 0 deg); the vessel heading
// relative to the weather is swept 0..350 deg. Result is the maximum wind speed (and Beaufort) that can be held.
// DNV-ST-0111 dynamic allowance: environmental forces (wind, wave drift, current) are multiplied by
// kDynamicAllowance (1.25, override with CABLELAY_DYNAMIC_ALLOWANCE). Operational loads (tow force and the hull
// resistance / wind from the vessel's own speed in LAY mode) are not factored.
// Thrust bias (azimuth pairs) is off by default, CABLELAY_BIAS=1/2/3 for Low / Medium / High.
// No auxiliary / cable-lay equipment load is included.
//
// Output files (format used by tools/CapabilityAnalysisReport):
//   <prefix>_<mode>_<case>.out, <prefix>_<mode>_<case>_ThrusterStates.out, <prefix>_<mode>_<case>_PowerStates.out
//   <prefix>_summary.out

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "brucon/dp/vessel_model/response_function.h"
#include "brucon/dp/vessel_model/simple_section_generator.h"
#include "brucon/dp/vessel_model/vessel_model.h"
#include "brucon/dp/vessel_model/vessel_sections.h"
#include "brucon/dp/vessel_model/wave_response.h"
#include "brucon/dp/vessel_model/wave_spectrum.h"
#include "brucon/dp/vessel_model/wind_force_model.h"
#include "brucon/math/math.h"
#include "brucon/propulsor_configuration/propulsor_configuration.h"
#include "brucon/proto_utils/proto_utils.h"
#include "brucon/proto_utils/vessel_section_config.h"
#include "brucon/thrust_allocation/basic_allocation.h"
#include "config_tree.pb.h"

using namespace brucon::dp;
using brucon::util::Math;

namespace {

constexpr double kKnot = 0.514444;
constexpr double kGravity = 9.81;

// DNV-ST-0111 Beaufort relation (same as libs/dp/capability_analysis)
const std::vector<double> kBf{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const std::vector<double> kBfWind{0.0, 1.5, 3.4, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6};
const std::vector<double> kBfHs{0.0, 0.1, 0.4, 0.8, 1.3, 2.1, 3.1, 4.2, 5.7, 7.4, 9.5, 12.1};
const std::vector<double> kBfTp{3.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 10.5, 11.5, 12.0};
const std::vector<double> kBfCur{0.0, 0.25, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
constexpr double kMaxWind = 40.0;
constexpr double kDynamicAllowance = 1.25;  // DNV-ST-0111

struct Mode {
  std::string name;
  double ship_speed;  // m/s through water (excluding current)
  double tow_force;   // kN, acting astern on the vessel
};

struct FailureCase {
  std::string name;
  int failed_bus;                   // -1 = no bus failure
  std::set<int> thrusters_off{};    // Additional thrusters off (thruster index)
  std::set<int> rudders_off{};      // Additional rudders off (rudder index)
};

// Parses "name=t3,t5,r0" (thruster / rudder indices) or "name=b0,t3" (bus loss + thruster)
FailureCase ParseCase(const std::string& arg) {
  FailureCase fc{arg.substr(0, arg.find('=')), -1};
  std::string list = arg.find('=') == std::string::npos ? "" : arg.substr(arg.find('=') + 1);
  size_t pos = 0;
  while (pos < list.size()) {
    size_t end = list.find(',', pos);
    std::string item = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    int idx = std::stoi(item.substr(1));
    if (item[0] == 't') fc.thrusters_off.insert(idx);
    if (item[0] == 'r') fc.rudders_off.insert(idx);
    if (item[0] == 'b') fc.failed_bus = idx;
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  return fc;
}

struct Environment {
  double wind, hs, tp, cur, u_rel, v_rel;
  double wind_f[3], cur_f[3], drift_f[3];
  double tau[3];  // Required thruster force
};

WindForceModel MakeWindModel(const brucon::configuration::ConfigTree& config) {
  const auto& wd = config.vessel_wind_data();
  if (wd.has_wind_force_coefficients()) {
    std::vector<double> dir, cx, cy, cmz, cmx;
    for (const auto& c : wd.wind_force_coefficients().coefficients()) {
      dir.push_back(c.direction());
      cx.push_back(c.surge_drag_coefficient());
      cy.push_back(c.sway_drag_coefficient());
      cmz.push_back(c.yaw_drag_coefficient());
      cmx.push_back(c.roll_drag_coefficent());
    }
    return WindForceModel({dir, cx, cy, cmz, cmx, config.vessel_data().loa(), wd.lateral_wind_area(),
                           wd.frontal_wind_area()});
  }
  return WindForceModel({static_cast<ShipType>(wd.ship_type()), config.vessel_data().loa(), wd.lateral_wind_area(),
                         wd.frontal_wind_area(), wd.lateral_wind_area_centre(), wd.height_of_wind_area_centre()});
}

class Study {
 public:
  Study(const brucon::configuration::ConfigTree& config, VesselModel vessel, WaveResponse wave, WindForceModel wind)
      : config_(config), vessel_(std::move(vessel)), wave_(std::move(wave)), wind_(std::move(wind)) {
    wave_.set_calculate_drift_forces(true);
    spectrum_.SetSpectrumType(kBretschneider);  // DNV-ST-0111: two-parameter PM (Bretschneider), Hs / Tp
    wave_.AddWaveSpectrum(&spectrum_);

    PropulsorConfiguration prop_config;
    prop_config.Configure(config);
    alloc_ = std::make_unique<BasicAllocator>(prop_config.GetPropulsorVector());
    const auto& ps = config.power_system();
    n_bus_ = ps.bus_size();
    alloc_->SetNumberOfBuses(n_bus_);

    // Producers per bus
    bus_capacity_.assign(n_bus_, 0.0);
    for (const auto& g : ps.generator()) {
      bus_capacity_[g.breaker_size() > 0 ? g.breaker(0).bus_index() : 0] += g.nominal_power();
    }
    for (const auto& e : ps.ess_ac()) {
      bus_capacity_[e.breaker_size() > 0 ? e.breaker(0).bus_index() : 0] += e.nominal_power();
    }
    // Thruster -> bus
    thruster_bus_.assign(alloc_->NumberOfThrusters(), -1);
    for (const auto& t : ps.thruster()) {
      thruster_bus_.at(t.thruster_index()) = t.breaker_size() > 0 ? t.breaker(0).bus_index() : 0;
    }
    // Rudder -> main propeller (same y position) -> bus
    std::vector<std::pair<double, int>> main_props;  // y, thruster index
    std::vector<double> rudder_y;
    int thr = 0;
    for (const auto& p : config.propulsors().propulsor()) {
      if (p.has_rudder()) {
        rudder_y.push_back(p.position().y_position());
      } else {
        if (p.has_main_propeller()) main_props.emplace_back(p.position().y_position(), thr);
        ++thr;
      }
      names_.push_back(p.short_name().empty() ? p.name() : p.short_name());
    }
    for (double y : rudder_y) {
      int bus = -1;
      for (const auto& [my, idx] : main_props) {
        if (std::fabs(my - y) < 0.5) bus = thruster_bus_[idx];
      }
      rudder_bus_.push_back(bus);
    }
    // Configs without power system mapping: all thrusters on one bus with unlimited power.
    if (std::all_of(thruster_bus_.begin(), thruster_bus_.end(), [](int b) { return b < 0; })) {
      std::cerr << "WARNING: no thrusters in power_system, using one bus with unlimited power\n";
      n_bus_ = std::max<size_t>(n_bus_, 1);
      alloc_->SetNumberOfBuses(n_bus_);
      bus_capacity_.assign(n_bus_, 0.0);
      bus_capacity_[0] = 1.0e9;
      std::fill(thruster_bus_.begin(), thruster_bus_.end(), 0);
      for (auto& b : rudder_bus_) b = 0;
    }
    for (size_t i = 0; i < thruster_bus_.size(); ++i) {
      if (thruster_bus_[i] < 0) std::cerr << "WARNING: thruster " << i << " not connected in power_system\n";
    }
  }

  // Thruster power [kW]. For azimuths the allocator gives thrust = force magnitude at low inflow (thrust
  // deduction not applied in the inverse), so power is computed for thrust = force / (1 - t) here.
  double ThrusterPower(int i, const Environment& e) {
    if (alloc_->IsAzimuth(i)) {
      const double t = AzimuthThrustDeduction(i);
      if (t > 0.0 && std::fabs(alloc_->Thrust(i)) > 1e-6) {
        const double va = std::max(0.0, e.u_rel) * (1.0 - AzimuthWake(i));
        return alloc_->PowerForThrust(i, std::fabs(alloc_->Thrust(i)) / (1.0 - t), va);
      }
    }
    return alloc_->ThrusterPowerConsumption(i);
  }
  const brucon::configuration::AllSpeedData* AzimuthAllSpeed(int i) const {
    const auto& p = config_.propulsors().propulsor(static_cast<int>(ThrusterPropulsorIndex(i)));
    if (p.has_azimuth() && p.azimuth().details().has_all_speed_data()) return &p.azimuth().details().all_speed_data();
    return nullptr;
  }
  double AzimuthThrustDeduction(int i) const {
    auto a = AzimuthAllSpeed(i);
    return a ? a->ahead_coefficients().thrust_deduction() : 0.0;
  }
  double AzimuthWake(int i) const {
    auto a = AzimuthAllSpeed(i);
    return a ? a->ahead_coefficients().wake_fraction() : 0.0;
  }

  // Power consumption at fixed weather (Beaufort bf, DNV relation), all headings. Mean forces, no dynamic allowance.
  // Writes <prefix>_<mode>_<case>_Bf<bf>_power.out and returns {mean, min, max} total propulsion power [kW].
  std::array<double, 3> RunFixedWeather(const std::string& prefix, const Mode& mode, const FailureCase& fc,
                                        double bf) {
    SetFailureCase(fc);
    const double saved = dynamic_allowance_;
    dynamic_allowance_ = 1.0;
    const double wind = Math::Interpolate(kBf, kBfWind, bf, false);
    std::ofstream out(prefix + "_" + mode.name + "_" + fc.name + "_Bf" + std::to_string(static_cast<int>(bf)) +
                      "_power.out");
    out << "dir\tFeasible\tP_total(kW)";
    for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) out << "\tP_" << names_.at(ThrusterPropulsorIndex(i));
    out << '\n';
    double sum = 0.0, mn = 1e9, mx = 0.0;
    int n = 0;
    for (double h = 0.0; h < 359.9; h += 10.0) {
      const bool ok = Feasible(Env(mode, h, wind, mode.tow_force));
      double p = 0.0;
      std::vector<double> pi;
      for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) {
        pi.push_back(ThrusterPower(i, Env(mode, h, wind, mode.tow_force)));
        p += pi.back();
      }
      out << h << '\t' << ok << '\t' << p;
      for (double x : pi) out << '\t' << x;
      if (std::getenv("CABLELAY_DEBUG")) {
        for (int i = 0; i < alloc_->NumberOfThrusters(); ++i)
          std::cerr << "h" << h << " thr" << i << " T " << alloc_->Thrust(i) << " az " << alloc_->AzimuthAngle(i) << '\n';
      }
      out << '\n';
      sum += p;
      mn = std::min(mn, p);
      mx = std::max(mx, p);
      ++n;
    }
    dynamic_allowance_ = saved;
    return {sum / n, mn, mx};
  }

  // Saturate all thrusters with pure surge / pure sway demand (zero yaw) and print per thruster thrust and power.
  void PrintThrusterData() {
    SetFailureCase({"Intact", -1});
    alloc_->UpdateVesselSpeedThroughWater(0.0, 0.0, 0.0);
    std::vector<std::array<double, 4>> res(alloc_->NumberOfThrusters());
    const double demand[2][3] = {{5000.0, 0.0, 0.0}, {0.0, 5000.0, 0.0}};
    double total[2][3];
    for (int k = 0; k < 2; ++k) {
      alloc_->AllocateWithLimitedPower(demand[k][0], demand[k][1], demand[k][2], true, true, true);
      total[k][0] = alloc_->ResultSurge();
      total[k][1] = alloc_->ResultSway();
      total[k][2] = alloc_->ResultYaw();
      for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) {
        res[i][2 * k] = alloc_->Thrust(i);
        res[i][2 * k + 1] = alloc_->ThrusterPowerConsumption(i);
      }
    }
    std::cout << "  Saturated allocation, zero speed, intact (thrust kN / power kW per thruster):\n"
              << "  idx name        bus | max surge: T      P | max sway: T      P\n";
    for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) {
      std::cout << "  " << std::setw(3) << i << ' ' << std::setw(10) << names_.at(ThrusterPropulsorIndex(i)) << ' '
                << std::setw(4) << thruster_bus_[i] << " |" << std::fixed << std::setprecision(0) << std::setw(12)
                << res[i][0] << std::setw(7) << res[i][1] << " |" << std::setw(11) << res[i][2] << std::setw(7)
                << res[i][3] << '\n';
    }
    std::cout << "  total Fx/Fy/Mz: surge case " << total[0][0] << '/' << total[0][1] << '/' << total[0][2]
              << ", sway case " << total[1][0] << '/' << total[1][1] << '/' << total[1][2] << '\n';
  }

  size_t NumberOfBuses() const { return n_bus_; }
  const std::vector<double>& BusCapacity() const { return bus_capacity_; }

  void SetFailureCase(const FailureCase& fc) {
    const int nb = static_cast<int>(n_bus_);
    alloc_->ResetBusGroups(fc.failed_bus < 0 ? 1 : n_bus_);
    group_capacity_.clear();
    if (fc.failed_bus < 0) {
      double total = 0.0;
      for (int b = 0; b < nb; ++b) {
        alloc_->SetBusBusGroupIndexMap(b, 0);
        total += bus_capacity_[b];
      }
      group_capacity_.push_back(total);
    } else {
      for (int b = 0; b < nb; ++b) {
        alloc_->SetBusBusGroupIndexMap(b, b);
        group_capacity_.push_back(b == fc.failed_bus ? 0.0 : bus_capacity_[b]);
      }
    }
    for (size_t g = 0; g < group_capacity_.size(); ++g) alloc_->SetBusGroupCapacity(g, group_capacity_[g]);
    for (size_t i = 0; i < thruster_bus_.size(); ++i) {
      alloc_->SetThrusterBusIndices(i, {static_cast<size_t>(thruster_bus_[i])});
      alloc_->ThrusterActive(i, thruster_bus_[i] >= 0 && thruster_bus_[i] != fc.failed_bus &&
                                     !fc.thrusters_off.count(static_cast<int>(i)));
    }
    for (size_t r = 0; r < rudder_bus_.size(); ++r) {
      alloc_->RudderActive(r, rudder_bus_[r] != fc.failed_bus && !fc.rudders_off.count(static_cast<int>(r)));
    }
    // Thrust bias off (default). CABLELAY_BIAS=1..3 selects Low / Medium / High.
    alloc_->Allocate(0.0, 0.0, 0.0, true, true, true);  // Make sure the allocator is configured
    const char* bias = std::getenv("CABLELAY_BIAS");
    alloc_->SetBiasSettings(static_cast<BiasSetting>(bias ? std::atoi(bias) : 0));
  }

  Environment Env(const Mode& mode, double heading, double wind_speed, double tow_force) {
    Environment e{};
    e.wind = wind_speed;
    e.hs = Math::Interpolate(kBfWind, kBfHs, wind_speed, false);
    e.tp = Math::Interpolate(kBfWind, kBfTp, wind_speed, false);
    e.cur = Math::Interpolate(kBfWind, kBfCur, wind_speed, false);
    // Relative water velocity in body frame (same convention as CapabilityAnalysis): ship speed ahead adds to u.
    e.u_rel = mode.ship_speed + e.cur * Math::CosDeg(heading);
    e.v_rel = e.cur * Math::SinDeg(-heading);
    wind_.VesselStates(wind_speed, 0.0, mode.ship_speed, 0.0, heading);
    spectrum_.SetParameters({e.hs, 0.0, e.tp, 2.0});
    auto drift = wave_.MeanDriftForces(heading, e.u_rel);
    for (int k = 0; k < 3; ++k) e.drift_f[k] = drift[k] / 1000.0;
    e.cur_f[0] = vessel_.X(e.u_rel, e.v_rel, 0) / 1000.0;
    e.cur_f[1] = vessel_.Y(e.u_rel, e.v_rel, 0) / 1000.0;
    e.cur_f[2] = vessel_.N(e.u_rel, e.v_rel, 0) / 1000.0;
    e.wind_f[0] = wind_.SurgeForce() / 1000.0;
    e.wind_f[1] = wind_.SwayForce() / 1000.0;
    e.wind_f[2] = wind_.YawMoment() / 1000.0;
    const double tow[3] = {-tow_force, 0.0, 0.0};  // Tow on centre line, astern
    // Forces from the vessel's own speed only (calm, no current / true wind): operational, not factored
    wind_.VesselStates(0.0, 0.0, mode.ship_speed, 0.0, heading);
    const double own[3] = {vessel_.X(mode.ship_speed, 0.0, 0) / 1000.0 + wind_.SurgeForce() / 1000.0,
                           vessel_.Y(mode.ship_speed, 0.0, 0) / 1000.0 + wind_.SwayForce() / 1000.0,
                           vessel_.N(mode.ship_speed, 0.0, 0) / 1000.0 + wind_.YawMoment() / 1000.0};
    for (int k = 0; k < 3; ++k) {
      const double environment = e.drift_f[k] + e.cur_f[k] + e.wind_f[k] - own[k];
      e.tau[k] = -(dynamic_allowance_ * environment + own[k] + tow[k]);
    }
    return e;
  }

  bool Feasible(const Environment& e) {
    alloc_->UpdateVesselSpeedThroughWater(e.u_rel, e.v_rel, 0.0);
    alloc_->AllocateWithLimitedPower(e.tau[0], e.tau[1], e.tau[2], true, true, true);
    if (std::getenv("CABLELAY_DEBUG")) {
      std::cerr << "uv " << e.u_rel << ' ' << e.v_rel << " tau " << e.tau[0] << ' ' << e.tau[1] << ' ' << e.tau[2] << " -> " << alloc_->ResultSurge() << ' '
                << alloc_->ResultSway() << ' ' << alloc_->ResultYaw() << " |";
      for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) std::cerr << ' ' << alloc_->ThrustPercent(i);
      for (int r = 0; r < alloc_->NumberOfRudders(); ++r) std::cerr << " R" << alloc_->RudderAngle(r);
      std::cerr << '\n';
    }
    const double tol_f = std::max(1.0, 0.005 * std::hypot(e.tau[0], e.tau[1]));
    return std::fabs(alloc_->ResultSurge() - e.tau[0]) <= tol_f && std::fabs(alloc_->ResultSway() - e.tau[1]) <= tol_f &&
           std::fabs(alloc_->ResultYaw() - e.tau[2]) <= 10.0;
  }

  // Returns max wind speed, or -1 if not feasible in calm water.
  double MaxWind(const Mode& mode, double heading) {
    if (!Feasible(Env(mode, heading, 0.0, mode.tow_force))) return -1.0;
    if (Feasible(Env(mode, heading, kMaxWind, mode.tow_force))) return kMaxWind;
    double lo = 0.0, hi = kMaxWind;
    for (int i = 0; i < 14; ++i) {
      double mid = 0.5 * (lo + hi);
      (Feasible(Env(mode, heading, mid, mode.tow_force)) ? lo : hi) = mid;
    }
    return lo;
  }

  // Max bollard pull [kN] in calm water at the mode speed.
  double MaxTowForce(const Mode& mode) {
    double lo = 0.0, hi = 3000.0;
    if (!Feasible(Env(mode, 0.0, 0.0, 0.0))) return 0.0;
    for (int i = 0; i < 16; ++i) {
      double mid = 0.5 * (lo + hi);
      (Feasible(Env(mode, 0.0, 0.0, mid)) ? lo : hi) = mid;
    }
    return lo;
  }

  void Run(const std::string& prefix, const Mode& mode, const FailureCase& fc, std::ofstream& summary) {
    SetFailureCase(fc);
    const std::string base = prefix + "_" + mode.name + "_" + fc.name;
    std::ofstream out(base + ".out"), thr(base + "_ThrusterStates.out"), pwr(base + "_PowerStates.out");
    out << "dir\twindSpeed\twindX\twindY\twindZ\tCurX\tCurY\tCurZ\tDriftX\tDriftY\tDriftZ\tBf[-]\tCurrent(m/s)"
           "\tHs(m)\tTp(s)\tTowX\tFeasible\n";
    thr << "heading\tthr_idx\tThrust%\tThrust_kN\tPower_kW\tIsAzimuth\tAngle_deg\n";
    pwr << "heading\tbus_group_idx\tAvail_kW\tConsumed_kW\tConsumed%\tReserve_kW\tReserve%\n";

    double min_wind = 1e9, sum_bf = 0.0;
    int n = 0, n_infeasible = 0;
    const char* only_heading = std::getenv("CABLELAY_HEADING");  // Debug: run a single heading
    for (double h = 0.0; h < 359.9; h += 10.0) {
      if (only_heading && std::fabs(h - std::atof(only_heading)) > 0.1) continue;
      double ws = MaxWind(mode, h);
      bool feasible = ws >= 0.0;
      if (!feasible) ++n_infeasible;
      Environment e = Env(mode, h, std::max(ws, 0.0), mode.tow_force);
      Feasible(e);  // Allocate at limit for thruster / power states
      double bf = feasible ? Math::Interpolate(kBfWind, kBf, e.wind, false) : 0.0;
      min_wind = std::min(min_wind, std::max(ws, 0.0));
      sum_bf += bf;
      ++n;
      out << h << '\t' << e.wind << '\t' << e.wind_f[0] << '\t' << e.wind_f[1] << '\t' << e.wind_f[2] << '\t'
          << e.cur_f[0] << '\t' << e.cur_f[1] << '\t' << e.cur_f[2] << '\t' << e.drift_f[0] << '\t' << e.drift_f[1]
          << '\t' << e.drift_f[2] << '\t' << bf << '\t' << e.cur << '\t' << e.hs << '\t' << e.tp << '\t'
          << -mode.tow_force << '\t' << feasible << '\n';
      for (int i = 0; i < alloc_->NumberOfThrusters(); ++i) {
        thr << h << '\t' << i << '\t' << alloc_->ThrustPercent(i) << '\t' << alloc_->Thrust(i) << '\t'
            << alloc_->ThrusterPowerConsumption(i) << '\t' << alloc_->IsAzimuth(i) << '\t'
            << (alloc_->IsAzimuth(i) ? alloc_->AzimuthAngle(i) : 0.0) << '\n';
      }
      for (size_t g = 0; g < group_capacity_.size(); ++g) {
        double avail = group_capacity_[g], used = alloc_->BusGroupPowerConsumedDp(g);
        double pct = avail > 0 ? 100.0 * used / avail : 0.0;
        pwr << h << '\t' << g << '\t' << avail << '\t' << used << '\t' << pct << '\t' << avail - used << '\t'
            << (avail > 0 ? 100.0 - pct : 0.0) << '\n';
      }
    }
    double max_tow = MaxTowForce(mode);
    summary << mode.name << '\t' << fc.name << '\t' << std::fixed << std::setprecision(2) << min_wind << '\t'
            << Math::Interpolate(kBfWind, kBf, min_wind, false) << '\t' << sum_bf / n << '\t' << n_infeasible << '\t'
            << max_tow << '\t' << max_tow / kGravity << '\n';
    std::cout << std::left << std::setw(5) << mode.name << std::setw(8) << fc.name << " min wind " << std::fixed
              << std::setprecision(1) << min_wind << " m/s, mean Bf " << sum_bf / n << ", infeasible headings "
              << n_infeasible << ", calm max pull " << max_tow / kGravity << " t\n";
  }

 private:
  size_t ThrusterPropulsorIndex(int thruster_index) const {
    int t = -1;
    for (int p = 0; p < config_.propulsors().propulsor_size(); ++p) {
      if (!config_.propulsors().propulsor(p).has_rudder() && ++t == thruster_index) return p;
    }
    return 0;
  }

  const brucon::configuration::ConfigTree& config_;
  VesselModel vessel_;
  WaveResponse wave_;
  WindForceModel wind_;
  WaveSpectrum spectrum_{0.0, 0.0, 0.0, 2.0};
  std::unique_ptr<BasicAllocator> alloc_;
  double dynamic_allowance_ = std::getenv("CABLELAY_DYNAMIC_ALLOWANCE")
                                  ? std::atof(std::getenv("CABLELAY_DYNAMIC_ALLOWANCE"))
                                  : kDynamicAllowance;
  size_t n_bus_{0};
  std::vector<double> bus_capacity_, group_capacity_;
  std::vector<int> thruster_bus_, rudder_bus_;
  std::vector<std::string> names_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <config_dir> <pdstrip file in vessel_simulator_config/> <output prefix> [case=t<i>,r<i>,b<i> ...]\n"
              << "  Without extra cases: Intact + loss of each bus. With extra cases: only those cases.\n";
    return 1;
  }
  const std::string config_dir = argv[1], pdstrip = argv[2], prefix = argv[3];

  brucon::configuration::ConfigTree config;
  if (!brucon::proto_utils::ParseConfigTree(config, config_dir)) {
    std::cerr << "Failed to parse config " << config_dir << "\n";
    return 1;
  }
  const auto& vd = config.vessel_data();
  auto sections = [&]() {
    if (vd.has_sections()) {
      VesselSectionConfig section_config(vd.sections());
      return VesselSections(section_config.SectionData(), 21).SectionData();
    }
    return SimpleSectionGenerator(vd.lpp(), vd.breadth(), vd.design_draft(), vd.block_coefficient()).SectionData();
  }();
  VesselModel vessel(sections, vd.lpp(), vd.design_draft(), vd.breadth());
  ResponseFunctions resp;
  if (resp.ReadExportedPdStripDatFile("vessel_simulator_config/" + pdstrip) == 0) {
    std::cerr << "Missing response file vessel_simulator_config/" << pdstrip << "\n";
    return 1;
  }
  Study study(config, std::move(vessel), WaveResponse(std::move(resp)), MakeWindModel(config));

  // Lay mode speed / pull can be overridden with CABLELAY_LAY_SPEED_KN and CABLELAY_LAY_PULL_T
  auto env_or = [](const char* name, double def) { const char* v = std::getenv(name); return v ? std::atof(v) : def; };
  const double lay_speed = env_or("CABLELAY_LAY_SPEED_KN", 0.8) * kKnot;
  const double lay_pull = env_or("CABLELAY_LAY_PULL_T", 80.0) * kGravity;
  const std::vector<Mode> modes{{"DP", 0.0, 0.0}, {"LAY", lay_speed, lay_pull}};
  std::vector<FailureCase> cases;
  for (int a = 4; a < argc; ++a) cases.push_back(ParseCase(argv[a]));
  if (cases.empty()) {
    cases.push_back({"Intact", -1});
    for (size_t b = 0; b < study.NumberOfBuses(); ++b) {
      cases.push_back({"Bus" + std::to_string(b), static_cast<int>(b)});
    }
  }

  std::cout << config_dir << ": bus capacities [kW]:";
  for (double c : study.BusCapacity()) std::cout << ' ' << c;
  std::cout << '\n';
  study.PrintThrusterData();

  std::ofstream summary(prefix + "_summary.out");
  summary << "mode\tcase\tmin_wind(m/s)\tmin_Bf\tmean_Bf\tinfeasible_headings\tcalm_max_pull(kN)\tcalm_max_pull(t)\n";
  for (const auto& m : modes) {
    for (const auto& c : cases) study.Run(prefix, m, c, summary);
  }

  // Power consumption in operation, intact, fixed weather
  std::ofstream psum(prefix + "_power_summary.out");
  psum << "mode\tBf\tP_mean(kW)\tP_min(kW)\tP_max(kW)\n";
  std::cout << "Power consumption, intact, mean / min / max over headings [kW] (propeller shaft power):\n";
  for (const auto& m : modes) {
    for (double bf : {3.0, 6.0}) {
      auto [mean, mn, mx] = study.RunFixedWeather(prefix, m, {"Intact", -1}, bf);
      psum << m.name << '\t' << bf << '\t' << mean << '\t' << mn << '\t' << mx << '\n';
      std::cout << "  " << std::left << std::setw(5) << m.name << "Bf" << bf << std::right << std::fixed
                << std::setprecision(0) << std::setw(8) << mean << std::setw(8) << mn << std::setw(8) << mx << '\n';
    }
  }
  return 0;
}
