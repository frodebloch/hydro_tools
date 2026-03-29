"""Speed sweep summary text report."""

from simulation.results import SpeedSweepResult


def print_speed_sweep_summary(sweep: list[SpeedSweepResult],
                              idle_pct: float = 15.0,
                              fuel_price_eur_per_t: float = 0.0):
    """Print a compact comparison table across speeds."""
    if not sweep:
        print("No speed sweep results to display.")
        return

    print("\n" + "=" * 110)
    print("SPEED SENSITIVITY SUMMARY")
    print("=" * 110)
    print(f"  Idle: {idle_pct:.0f}%    Route distance: same for all speeds")
    if sweep[0].hull_ks_um > 0 or sweep[0].blade_ks_um > 0:
        print(f"  Hull roughness: {sweep[0].hull_ks_um:.0f} \u00b5m, "
              f"blade roughness: {sweep[0].blade_ks_um:.0f} \u00b5m")

    # --- Per-voyage table ---
    print(f"\n  PER-VOYAGE AVERAGES:")
    hdr = (f"  {'Speed':>6s}  {'Transit':>8s}  {'Voy/yr':>6s}  "
           f"{'Fac_NF':>8s}  {'Opt+Fl':>8s}  "
           f"{'Total%':>7s}  {'P/R%':>6s}  {'Flet%':>6s}  "
           f"{'Sav kg':>7s}  {'P/R kg':>7s}  {'Flet kg':>8s}  "
           f"{'%infeas':>7s}")
    print(hdr)
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 8}  "
          f"{'-' * 7}  {'-' * 6}  {'-' * 6}  "
          f"{'-' * 7}  {'-' * 7}  {'-' * 8}  "
          f"{'-' * 7}")
    for s in sweep:
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.transit_hours:6.1f} h  "
              f"{s.voyages_per_year:6.0f}  "
              f"{s.mean_fuel_factory_nf_kg:7.0f}kg  "
              f"{s.mean_fuel_opt_fl_kg:7.0f}kg  "
              f"{s.mean_saving_pct:>+6.1f}%  "
              f"{s.pct_pitch_rpm:>+5.1f}%  {s.pct_flettner:>+5.1f}%  "
              f"{s.mean_saving_total_kg:>7.0f}  "
              f"{s.mean_saving_pitch_rpm_kg:>7.0f}  "
              f"{s.mean_saving_flettner_kg:>8.0f}  "
              f"{s.pct_factory_infeasible:>6.0f}%")

    # --- Annualized table ---
    print(f"\n  ANNUALIZED FUEL (tonnes/year, {idle_pct:.0f}% idle):")
    hdr2 = (f"  {'Speed':>6s}  {'Fac_NF':>9s}  {'Opt+Fl':>9s}  "
            f"{'Saving':>9s}  {'P/R':>8s}  {'Flettner':>9s}  "
            f"{'Total%':>7s}")
    print(hdr2)
    print(f"  {'-' * 6}  {'-' * 9}  {'-' * 9}  "
          f"{'-' * 9}  {'-' * 8}  {'-' * 9}  "
          f"{'-' * 7}")
    for s in sweep:
        pct_total = 100.0 * s.ann_saving_total_t / s.ann_fuel_factory_nf_t \
            if s.ann_fuel_factory_nf_t > 0 else 0.0
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.ann_fuel_factory_nf_t:8.1f} t  "
              f"{s.ann_fuel_opt_fl_t:8.1f} t  "
              f"{s.ann_saving_total_t:8.1f} t  "
              f"{s.ann_saving_pitch_rpm_t:7.1f} t  "
              f"{s.ann_saving_flettner_t:8.1f} t  "
              f"{pct_total:>+6.1f}%")

    # --- Weather conditions ---
    print(f"\n  MEAN WEATHER CONDITIONS (voyage average across year):")
    hdr3 = (f"  {'Speed':>6s}  {'Hs [m]':>7s}  {'Wind [m/s]':>10s}  "
            f"{'R_aw [kN]':>9s}  {'R_wind [kN]':>11s}  {'F_flett [kN]':>12s}  "
            f"{'P_rotor [kW]':>12s}")
    print(hdr3)
    print(f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 9}  {'-' * 11}  {'-' * 12}  {'-' * 12}")
    for s in sweep:
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.mean_hs:>7.2f}  "
              f"{s.mean_wind:>10.1f}  "
              f"{s.mean_R_aw_kN:>9.1f}  "
              f"{s.mean_R_wind_kN:>11.1f}  "
              f"{s.mean_F_flettner_kN:>12.1f}  "
              f"{s.mean_rotor_power_kW:>12.1f}")

    if fuel_price_eur_per_t > 0:
        fp = fuel_price_eur_per_t
        print(f"\n  FUEL COST ESTIMATES (@ \u20ac{fp:.0f}/tonne):")
        hdr4 = (f"  {'Speed':>6s}  {'Fac_NF':>12s}  {'Opt+Fl':>12s}  "
                f"{'Saving':>12s}  {'P/R':>10s}  {'Flettner':>10s}")
        print(hdr4)
        print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  "
              f"{'-' * 12}  {'-' * 10}  {'-' * 10}")
        for s in sweep:
            cost_fac = s.ann_fuel_factory_nf_t * fp
            cost_opt = s.ann_fuel_opt_fl_t * fp
            cost_sav = s.ann_saving_total_t * fp
            cost_pr = s.ann_saving_pitch_rpm_t * fp
            cost_fl = s.ann_saving_flettner_t * fp
            print(f"  {s.speed_kn:5.1f}kn  "
                  f"\u20ac{cost_fac:>10,.0f}  "
                  f"\u20ac{cost_opt:>10,.0f}  "
                  f"\u20ac{cost_sav:>10,.0f}  "
                  f"\u20ac{cost_pr:>8,.0f}  "
                  f"\u20ac{cost_fl:>8,.0f}")

    print()
