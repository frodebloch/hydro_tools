"""Summary and departure timing analysis text reports."""

import numpy as np

from simulation.results import VoyageResult


def print_summary(results: list[VoyageResult], speed_kn: float,
                  idle_pct: float = 15.0, fuel_price_eur_per_t: float = 0.0,
                  round_trip: bool = True):
    """Print summary statistics for the annual comparison.

    Parameters
    ----------
    idle_pct : float
        Percentage of year spent idle / in port.  Used to estimate
        the number of voyages per year and annualize fuel.
    fuel_price_eur_per_t : float
        Fuel price in EUR per tonne.  If > 0, cost estimates are printed.
    round_trip : bool
        If True, results represent round trips and labels reflect that.
    """

    if not results:
        print("No results to summarise.")
        return

    transit_h = results[0].total_hours

    # Annualization: how many voyages fit in a year?
    # Available sailing hours = 365.25 * 24 * (1 - idle_pct/100)
    # Each voyage takes transit_h hours (round-trip or one-way).
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voyages_per_year = sailing_hours_year / transit_h
    voy_label = "round-trip" if round_trip else "one-way"

    savings_pct = np.array([r.saving_pct for r in results])
    savings_kg = np.array([r.saving_kg for r in results])
    savings_pitch_rpm_kg = np.array([r.saving_pitch_rpm_kg for r in results])
    savings_flettner_kg = np.array([r.saving_flettner_kg for r in results])
    fuel_factory = np.array([r.total_fuel_factory_kg for r in results])
    fuel_factory_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_optimised = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_noflettner = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_wind = np.array([r.mean_wind for r in results])
    mean_R_aw = np.array([r.mean_R_aw_kN for r in results])
    mean_R_wind = np.array([r.mean_R_wind_kN for r in results])
    mean_F_flett = np.array([r.mean_F_flettner_kN for r in results])
    mean_rotor_pwr = np.array([r.mean_rotor_power_kW for r in results])
    rotor_fuel = np.array([r.total_rotor_fuel_kg for r in results])

    print("\n" + "=" * 78)
    print("ANNUAL SUMMARY")
    print("=" * 78)

    print(f"\n  Voyages simulated: {len(results)} ({voy_label})")
    print(f"  Transit speed: {speed_kn:.0f} kn")
    print(f"  Transit time per voyage: {transit_h:.1f} h ({transit_h / 24:.1f} days)")
    print(f"  Idle / port time: {idle_pct:.0f}%")
    print(f"  Estimated {voy_label} voyages per year: {voyages_per_year:.0f}")

    # Roughness state (from first result)
    r0 = results[0]
    if r0.hull_ks_um > 0 or r0.blade_ks_um > 0:
        print(f"\n  Hull roughness: {r0.hull_ks_um:.0f} \u00b5m "
              f"(\u0394R = {r0.R_roughness_kN:.1f} kN at {speed_kn:.0f} kn)")
        print(f"  Blade roughness: {r0.blade_ks_um:.0f} \u00b5m "
              f"(fuel factor = {r0.blade_fuel_factor:.3f})")

    # Factory feasibility statistics
    total_hours_all = sum(r.n_hours_total for r in results)
    total_both_feasible = sum(r.n_hours_both_feasible for r in results)
    total_factory_infeasible = sum(r.n_hours_factory_infeasible for r in results)
    pct_infeasible = 100.0 * total_factory_infeasible / total_hours_all if total_hours_all else 0

    print(f"\n  Factory combinator feasibility:")
    print(f"    Total evaluation hours: {total_hours_all}")
    print(f"    Both feasible:          {total_both_feasible} "
          f"({100 * total_both_feasible / total_hours_all:.0f}%)")
    print(f"    Factory infeasible:     {total_factory_infeasible} "
          f"({pct_infeasible:.0f}%)")
    print(f"    (Factory infeasible = engine over-power at factory combinator's")
    print(f"     fixed RPM schedule; optimiser adapts pitch/RPM to stay feasible)")

    n_voyages_with_infeasible = sum(1 for r in results if r.n_hours_factory_infeasible > 0)
    print(f"    Voyages with >= 1 infeasible hour: {n_voyages_with_infeasible} "
          f"({100 * n_voyages_with_infeasible / len(results):.0f}%)")

    # --- Per-voyage statistics ---
    print(f"\n  PER-VOYAGE AVERAGES (across {len(results)} simulated voyages):")
    print(f"\n  {'Metric':<35s}  {'Mean':>8s}  {'Median':>8s}  "
          f"{'P10':>8s}  {'P90':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-' * 35}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    def _row(label, arr, fmt=".1f"):
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        print(f"  {label:<35s}  {np.mean(arr):>8{fmt}}  {p50:>8{fmt}}  "
              f"{p10:>8{fmt}}  {p90:>8{fmt}}  {np.min(arr):>8{fmt}}  "
              f"{np.max(arr):>8{fmt}}")

    _row("Saving [%] (fac vs opt+Fl)", savings_pct)
    _row("Saving [kg/voy] (fac vs opt+Fl)", savings_kg, ".0f")
    _row("  of which pitch/RPM [kg]", savings_pitch_rpm_kg, ".0f")
    _row("  of which Flettner [kg]", savings_flettner_kg, ".0f")
    _row("Factory no-Fl fuel [kg/voy]", fuel_factory_nf, ".0f")
    _row("Factory+Fl fuel [kg/voy]", fuel_factory, ".0f")
    _row("Opt no-Flettner [kg/voy]", fuel_opt_noflettner, ".0f")
    _row("Opt + Flettner [kg/voy]", fuel_optimised, ".0f")
    _row("Mean Hs [m]", mean_hs)
    _row("Mean wind speed [m/s]", mean_wind)
    _row("Mean added resistance [kN]", mean_R_aw)
    _row("Mean wind resistance [kN]", mean_R_wind)
    _row("Mean Flettner thrust [kN]", mean_F_flett)
    _row("Mean rotor power [kW]", mean_rotor_pwr)
    _row("Rotor fuel [kg/voy]", rotor_fuel, ".0f")

    # Compute split percentages relative to factory-no-Flettner baseline
    pct_pitch_rpm = np.where(fuel_factory_nf > 0,
                             savings_pitch_rpm_kg / fuel_factory_nf * 100.0, 0.0)
    pct_flettner = np.where(fuel_factory_nf > 0,
                            savings_flettner_kg / fuel_factory_nf * 100.0, 0.0)
    _row("  Pitch/RPM saving [%]", pct_pitch_rpm)
    _row("  Flettner saving [%]", pct_flettner)

    # --- Annualized figures ---
    # Per-voyage means (on feasible-hours basis)
    mean_fac_nf_kg = np.mean(fuel_factory_nf)
    mean_fac_fl_kg = np.mean(fuel_factory)
    mean_opt_nf_kg = np.mean(fuel_opt_noflettner)
    mean_opt_fl_kg = np.mean(fuel_optimised)
    mean_sav_pr_kg = np.mean(savings_pitch_rpm_kg)
    mean_sav_fl_kg = np.mean(savings_flettner_kg)

    ann_fac_nf = mean_fac_nf_kg * voyages_per_year / 1000.0   # tonnes
    ann_fac_fl = mean_fac_fl_kg * voyages_per_year / 1000.0
    ann_opt_nf = mean_opt_nf_kg * voyages_per_year / 1000.0
    ann_opt_fl = mean_opt_fl_kg * voyages_per_year / 1000.0
    ann_sav_pr = mean_sav_pr_kg * voyages_per_year / 1000.0
    ann_sav_fl = mean_sav_fl_kg * voyages_per_year / 1000.0
    ann_sav_total = ann_sav_pr + ann_sav_fl

    print(f"\n  ANNUALIZED FUEL ({idle_pct:.0f}% idle -> {voyages_per_year:.0f} {voy_label} voyages/year):")
    print(f"    Factory no-Flettner:     {ann_fac_nf:7.1f} tonnes/year  <- baseline")
    print(f"    Factory + Flettner:      {ann_fac_fl:7.1f} tonnes/year")
    print(f"    Opt no-Flettner:         {ann_opt_nf:7.1f} tonnes/year")
    print(f"    Opt + Flettner:          {ann_opt_fl:7.1f} tonnes/year")
    if ann_fac_nf > 0:
        print(f"\n    Saving (fac_NF - opt_FL):  {ann_sav_total:7.1f} tonnes/year "
              f"({100 * ann_sav_total / ann_fac_nf:.1f}%)")
        print(f"      Pitch/RPM (fac_NF-opt_NF): {ann_sav_pr:7.1f} tonnes/year "
              f"({100 * ann_sav_pr / ann_fac_nf:.1f}%)")
        print(f"      Flettner (opt_NF-opt_FL):  {ann_sav_fl:7.1f} tonnes/year "
              f"({100 * ann_sav_fl / ann_fac_nf:.1f}%)")
        print(f"      Check: P/R + Fl = {ann_sav_total:.1f} tonnes")

    if fuel_price_eur_per_t > 0:
        fp = fuel_price_eur_per_t
        print(f"\n  FUEL COST ESTIMATES (MGO @ \u20ac{fp:.0f}/tonne):")
        print(f"    Factory no-Flettner:     \u20ac{ann_fac_nf * fp:>10,.0f}/year")
        print(f"    Factory + Flettner:      \u20ac{ann_fac_fl * fp:>10,.0f}/year")
        print(f"    Opt no-Flettner:         \u20ac{ann_opt_nf * fp:>10,.0f}/year")
        print(f"    Opt + Flettner:          \u20ac{ann_opt_fl * fp:>10,.0f}/year")
        if ann_fac_nf > 0:
            cost_saving = ann_sav_total * fp
            cost_pr = ann_sav_pr * fp
            cost_fl = ann_sav_fl * fp
            print(f"\n    Annual saving:       \u20ac{cost_saving:>10,.0f}/year "
                  f"({100 * ann_sav_total / ann_fac_nf:.1f}%)")
            print(f"      Pitch/RPM:        \u20ac{cost_pr:>10,.0f}/year")
            print(f"      Flettner:         \u20ac{cost_fl:>10,.0f}/year")

    # --- Seasonal breakdown ---
    print(f"\n  SEASONAL BREAKDOWN (per-voyage averages):")
    print(f"\n  {'Season':<12s}  {'Voyages':>7s}  {'Fac_NF':>8s}  {'Opt+Fl':>8s}  "
          f"{'Total %':>8s}  {'P/R %':>6s}  {'Flet %':>7s}  "
          f"{'Mean Hs':>8s}  {'Mean wind':>9s}  {'R_aw':>6s}  {'R_wind':>6s}  "
          f"{'%h infeas':>9s}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 6}  {'-' * 7}  "
          f"{'-' * 8}  {'-' * 9}  {'-' * 6}  {'-' * 6}  {'-' * 9}")

    quarters = {"Q1 (Jan-Mar)": (1, 3), "Q2 (Apr-Jun)": (4, 6),
                "Q3 (Jul-Sep)": (7, 9), "Q4 (Oct-Dec)": (10, 12)}
    for name, (m1, m2) in quarters.items():
        q_results = [r for r in results if m1 <= r.departure.month <= m2]
        if q_results:
            q_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in q_results])
            q_opt_fl = np.mean([r.total_fuel_optimised_kg for r in q_results])
            q_save = np.mean([r.saving_pct for r in q_results])
            q_fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in q_results])
            q_sav_pr = np.array([r.saving_pitch_rpm_kg for r in q_results])
            q_sav_fl = np.array([r.saving_flettner_kg for r in q_results])
            q_pct_pr = np.mean(q_sav_pr / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_pct_fl = np.mean(q_sav_fl / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_hs = np.mean([r.mean_hs for r in q_results])
            q_wind = np.mean([r.mean_wind for r in q_results])
            q_raw = np.mean([r.mean_R_aw_kN for r in q_results])
            q_rwind = np.mean([r.mean_R_wind_kN for r in q_results])
            q_total_h = sum(r.n_hours_total for r in q_results)
            q_infeas = sum(r.n_hours_factory_infeasible for r in q_results)
            q_pct_inf = 100 * q_infeas / q_total_h if q_total_h else 0
            print(f"  {name:<12s}  {len(q_results):>7d}  "
                  f"{q_fac_nf:>7.0f}kg  {q_opt_fl:>7.0f}kg  "
                  f"{q_save:>+7.1f}%  "
                  f"{q_pct_pr:>+5.1f}%  {q_pct_fl:>+6.1f}%  "
                  f"{q_hs:>8.1f}  {q_wind:>9.1f}  {q_raw:>5.1f}  {q_rwind:>5.1f}  "
                  f"{q_pct_inf:>8.0f}%")


def print_departure_analysis(results: list[VoyageResult], speed_kn: float,
                              idle_pct: float = 15.0,
                              fuel_price_eur_per_t: float = 650.0,
                              round_trip: bool = True,
                              windows: tuple[int, ...] = (3, 5, 7, 14)):
    """Analyse the value of departure timing flexibility.

    For each scheduling window size, computes what fuel saving could be
    achieved by choosing the best departure date within consecutive
    non-overlapping windows across the year.

    Also examines correlation between Flettner benefit and weather, and
    quantifies how much "smart scheduling" could enhance the Flettner value.

    Parameters
    ----------
    results : list[VoyageResult]
        One per departure day (365), sorted by departure date.
    speed_kn : float
        Transit speed.
    idle_pct : float
        Percentage of year idle (for annualization).
    fuel_price_eur_per_t : float
        Fuel price EUR/tonne.
    round_trip : bool
        Whether results are round-trips.
    windows : tuple[int, ...]
        Scheduling flexibility windows in days.
    """
    if not results or len(results) < 14:
        print("Insufficient results for departure analysis.")
        return

    n = len(results)
    transit_h = results[0].total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voyages_per_year = sailing_hours_year / transit_h
    voy_label = "round-trip" if round_trip else "one-way"

    # Extract arrays
    dates = [r.departure for r in results]
    fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    fuel_fac_fl = np.array([r.total_fuel_factory_kg for r in results])
    sav_total_kg = fuel_fac_nf - fuel_opt_fl       # total saving vs baseline
    sav_flettner_kg = fuel_opt_nf - fuel_opt_fl     # Flettner contribution
    sav_pitch_rpm_kg = fuel_fac_nf - fuel_opt_nf    # pitch/RPM contribution
    mean_wind = np.array([r.mean_wind for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_F_fl = np.array([r.mean_F_flettner_kN for r in results])

    print("\n" + "=" * 78)
    print("DEPARTURE TIMING ANALYSIS")
    print("=" * 78)

    # ---- 1. Overall fuel variability ----
    print(f"\n  1. FUEL VARIABILITY ACROSS {n} DEPARTURE DATES")
    print(f"  {'':>3s}  {'':>33s} {'Mean':>8s} {'Std':>8s} "
          f"{'Min':>8s} {'Max':>8s} {'Range':>8s}")

    for label, arr in [("Factory NF [kg/voy]", fuel_fac_nf),
                       ("Opt+Flettner [kg/voy]", fuel_opt_fl),
                       ("Total saving [kg/voy]", sav_total_kg),
                       ("Flettner saving [kg]", sav_flettner_kg)]:
        print(f"     {label:<33s} {np.mean(arr):>8.0f} {np.std(arr):>8.0f} "
              f"{np.min(arr):>8.0f} {np.max(arr):>8.0f} "
              f"{np.max(arr) - np.min(arr):>8.0f}")

    cv_fac = 100 * np.std(fuel_fac_nf) / np.mean(fuel_fac_nf)
    cv_opt = 100 * np.std(fuel_opt_fl) / np.mean(fuel_opt_fl)
    print(f"\n     Coefficient of variation:  Factory NF = {cv_fac:.1f}%,  "
          f"Opt+Fl = {cv_opt:.1f}%")
    print(f"     => Weather causes ~{cv_fac:.0f}% voyage-to-voyage fuel variation")

    # ---- 2. Scheduling window analysis ----
    print(f"\n  2. VALUE OF DEPARTURE FLEXIBILITY")
    print(f"     (choosing the lowest-fuel departure within a sliding window)\n")
    print(f"     {'Window':>8s}  {'Mean best':>10s}  {'Mean worst':>11s}  "
          f"{'Mean saving':>12s}  {'Save %':>7s}  "
          f"{'Annual':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'[kg/voy]':>10s}  {'[kg/voy]':>11s}  "
          f"{'[kg/voy]':>12s}  {'':>7s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 11}  "
          f"{'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 10}")

    # Use the factory NF fuel for scheduling analysis (baseline, no optimiser)
    # This shows the value of timing alone.
    # Also compute for opt+Fl to show combined value.
    for w in windows:
        if w > n:
            continue
        # Sliding window: for each possible window start, find min and max
        best_fac_nf = []
        worst_fac_nf = []
        best_opt_fl = []
        worst_opt_fl = []
        for i in range(n - w + 1):
            chunk_fac = fuel_fac_nf[i:i + w]
            chunk_opt = fuel_opt_fl[i:i + w]
            best_fac_nf.append(np.min(chunk_fac))
            worst_fac_nf.append(np.max(chunk_fac))
            best_opt_fl.append(np.min(chunk_opt))
            worst_opt_fl.append(np.max(chunk_opt))

        mean_best = np.mean(best_fac_nf)
        mean_worst = np.mean(worst_fac_nf)
        mean_sav = mean_worst - mean_best
        pct = 100 * mean_sav / mean_worst if mean_worst > 0 else 0
        annual_t = mean_sav * voyages_per_year / 1000.0
        annual_eur = annual_t * fuel_price_eur_per_t

        print(f"     {w:>8d}  {mean_best:>10.0f}  {mean_worst:>11.0f}  "
              f"{mean_sav:>12.0f}  {pct:>6.1f}%  "
              f"{annual_t:>8.1f}  {annual_eur:>10,.0f}")

    # ---- 3. Scheduling + optimiser + Flettner combined ----
    print(f"\n  3. COMBINED VALUE: TIMING + OPTIMISER + FLETTNER")
    print(f"     (best Opt+Fl day in window vs mean Factory NF)\n")
    print(f"     {'Window':>8s}  {'Mean best':>10s}  {'Baseline':>10s}  "
          f"{'Mean saving':>12s}  {'Save %':>7s}  "
          f"{'Annual':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'Opt+Fl':>10s}  {'Fac NF':>10s}  "
          f"{'[kg/voy]':>12s}  {'':>7s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 10}")

    mean_fac_nf = np.mean(fuel_fac_nf)
    # Without any flexibility (baseline)
    mean_opt_fl = np.mean(fuel_opt_fl)
    base_sav_kg = mean_fac_nf - mean_opt_fl
    base_pct = 100 * base_sav_kg / mean_fac_nf if mean_fac_nf > 0 else 0
    base_t = base_sav_kg * voyages_per_year / 1000.0
    base_eur = base_t * fuel_price_eur_per_t
    print(f"     {'none':>8s}  {mean_opt_fl:>10.0f}  {mean_fac_nf:>10.0f}  "
          f"{base_sav_kg:>12.0f}  {base_pct:>6.1f}%  "
          f"{base_t:>8.1f}  {base_eur:>10,.0f}")

    for w in windows:
        if w > n:
            continue
        best_combined = []
        for i in range(n - w + 1):
            best_combined.append(np.min(fuel_opt_fl[i:i + w]))
        mean_best_c = np.mean(best_combined)
        sav_c = mean_fac_nf - mean_best_c
        pct_c = 100 * sav_c / mean_fac_nf if mean_fac_nf > 0 else 0
        annual_t = sav_c * voyages_per_year / 1000.0
        annual_eur = annual_t * fuel_price_eur_per_t
        print(f"     {w:>8d}  {mean_best_c:>10.0f}  {mean_fac_nf:>10.0f}  "
              f"{sav_c:>12.0f}  {pct_c:>6.1f}%  "
              f"{annual_t:>8.1f}  {annual_eur:>10,.0f}")

    # ---- 4. Correlation analysis ----
    print(f"\n  4. WEATHER CORRELATIONS")
    # Pearson correlation between Flettner saving and weather
    # Use only voyages where Flettner saving > 0 (i.e. rotor was active)
    corr_fl_wind = np.corrcoef(sav_flettner_kg, mean_wind)[0, 1]
    corr_fl_hs = np.corrcoef(sav_flettner_kg, mean_hs)[0, 1]
    corr_fl_thrust = np.corrcoef(sav_flettner_kg, mean_F_fl)[0, 1]
    corr_fuel_hs = np.corrcoef(fuel_fac_nf, mean_hs)[0, 1]
    corr_fuel_wind = np.corrcoef(fuel_fac_nf, mean_wind)[0, 1]

    print(f"     Pearson r between Flettner saving and mean wind:   {corr_fl_wind:>+.3f}")
    print(f"     Pearson r between Flettner saving and mean Hs:     {corr_fl_hs:>+.3f}")
    print(f"     Pearson r between Flettner saving and mean F_fl:   {corr_fl_thrust:>+.3f}")
    print(f"     Pearson r between Factory NF fuel and mean Hs:     {corr_fuel_hs:>+.3f}")
    print(f"     Pearson r between Factory NF fuel and mean wind:   {corr_fuel_wind:>+.3f}")

    # The key insight: when wind is high, BOTH total fuel goes up (waves, wind
    # drag) AND Flettner saving goes up. Does the Flettner saving outweigh the
    # extra fuel? Look at the net effect.
    corr_opt_fl_wind = np.corrcoef(fuel_opt_fl, mean_wind)[0, 1]
    corr_opt_fl_hs = np.corrcoef(fuel_opt_fl, mean_hs)[0, 1]
    print(f"\n     Pearson r between Opt+Fl fuel and mean wind:      {corr_opt_fl_wind:>+.3f}")
    print(f"     Pearson r between Opt+Fl fuel and mean Hs:        {corr_opt_fl_hs:>+.3f}")

    # Does the Flettner reduce weather sensitivity?
    print(f"\n     Fuel std dev:  Factory NF = {np.std(fuel_fac_nf):.0f} kg,  "
          f"Opt+Fl = {np.std(fuel_opt_fl):.0f} kg")
    if np.std(fuel_opt_fl) < np.std(fuel_fac_nf):
        reduction = 100 * (1 - np.std(fuel_opt_fl) / np.std(fuel_fac_nf))
        print(f"     => Flettner + optimiser reduces fuel variability by "
              f"{reduction:.0f}%")
    else:
        increase = 100 * (np.std(fuel_opt_fl) / np.std(fuel_fac_nf) - 1)
        print(f"     => Opt+Fl has {increase:.0f}% MORE fuel variability than "
              f"Factory NF")
        print(f"        (Flettner adds variance: large saving in wind, "
              f"near-zero in calm)")

    # ---- 5. Monthly breakdown ----
    print(f"\n  5. MONTHLY BREAKDOWN")
    print(f"     {'Month':<10s}  {'N':>4s}  {'FacNF':>7s}  {'Opt+Fl':>7s}  "
          f"{'Save':>6s}  {'Fl save':>8s}  {'Wind':>5s}  {'Hs':>5s}  "
          f"{'F_fl':>5s}")
    print(f"     {'':>10s}  {'':>4s}  {'kg/voy':>7s}  {'kg/voy':>7s}  "
          f"{'%':>6s}  {'kg/voy':>8s}  {'m/s':>5s}  {'m':>5s}  "
          f"{'kN':>5s}")
    print(f"     {'-' * 10}  {'-' * 4}  {'-' * 7}  {'-' * 7}  "
          f"{'-' * 6}  {'-' * 8}  {'-' * 5}  {'-' * 5}  {'-' * 5}")

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    monthly_sav_pct = []
    for m in range(1, 13):
        idx = [i for i, d in enumerate(dates) if d.month == m]
        if not idx:
            continue
        m_fac_nf = np.mean(fuel_fac_nf[idx])
        m_opt_fl = np.mean(fuel_opt_fl[idx])
        m_sav_pct = 100 * (m_fac_nf - m_opt_fl) / m_fac_nf if m_fac_nf > 0 else 0
        m_fl_sav = np.mean(sav_flettner_kg[idx])
        m_wind = np.mean(mean_wind[idx])
        m_hs = np.mean(mean_hs[idx])
        m_ffl = np.mean(mean_F_fl[idx])
        monthly_sav_pct.append(m_sav_pct)
        print(f"     {months[m - 1]:<10s}  {len(idx):>4d}  {m_fac_nf:>7.0f}  "
              f"{m_opt_fl:>7.0f}  {m_sav_pct:>5.1f}%  {m_fl_sav:>8.0f}  "
              f"{m_wind:>5.1f}  {m_hs:>5.2f}  {m_ffl:>5.1f}")

    # ---- 6. "Smart scheduling" Flettner premium ----
    # If an operator could choose WHICH days to sail (within flexibility),
    # how much more Flettner benefit could they capture?
    print(f"\n  6. SMART SCHEDULING: FLETTNER-AWARE DEPARTURE SELECTION")
    print(f"     (within each window, choose the day with maximum Flettner saving)\n")
    print(f"     {'Window':>8s}  {'Mean Fl':>9s}  {'vs random':>10s}  "
          f"{'Fl boost':>9s}")
    print(f"     {'[days]':>8s}  {'save kg':>9s}  {'Fl save':>10s}  "
          f"{'':>9s}")
    print(f"     {'-' * 8}  {'-' * 9}  {'-' * 10}  {'-' * 9}")

    mean_fl_sav = np.mean(sav_flettner_kg)
    print(f"     {'none':>8s}  {mean_fl_sav:>9.0f}  {'(baseline)':>10s}  "
          f"{'':>9s}")

    for w in windows:
        if w > n:
            continue
        best_fl = []
        for i in range(n - w + 1):
            chunk = sav_flettner_kg[i:i + w]
            best_fl.append(np.max(chunk))
        mean_best_fl = np.mean(best_fl)
        boost = 100 * (mean_best_fl / mean_fl_sav - 1) if mean_fl_sav > 0 else 0
        print(f"     {w:>8d}  {mean_best_fl:>9.0f}  {mean_fl_sav:>10.0f}  "
              f"{boost:>+8.0f}%")

    # ---- 7. Worst-case avoidance ----
    print(f"\n  7. WORST-CASE AVOIDANCE")
    print(f"     (value of avoiding the highest-fuel departure in each window)\n")
    p95_fuel = np.percentile(fuel_fac_nf, 95)
    p99_fuel = np.percentile(fuel_fac_nf, 99)
    worst_fuel = np.max(fuel_fac_nf)
    worst_idx = np.argmax(fuel_fac_nf)
    worst_date = dates[worst_idx]
    print(f"     Worst departure:  {worst_date.strftime('%Y-%m-%d')}  "
          f"fuel = {worst_fuel:.0f} kg  "
          f"(Hs = {mean_hs[worst_idx]:.2f} m, wind = {mean_wind[worst_idx]:.1f} m/s)")
    best_fuel = np.min(fuel_fac_nf)
    best_idx = np.argmin(fuel_fac_nf)
    best_date = dates[best_idx]
    print(f"     Best departure:   {best_date.strftime('%Y-%m-%d')}  "
          f"fuel = {best_fuel:.0f} kg  "
          f"(Hs = {mean_hs[best_idx]:.2f} m, wind = {mean_wind[best_idx]:.1f} m/s)")
    print(f"     Range: {worst_fuel - best_fuel:.0f} kg  "
          f"({100 * (worst_fuel - best_fuel) / best_fuel:.0f}% of best)")
    print(f"     P95 fuel: {p95_fuel:.0f} kg,  P99: {p99_fuel:.0f} kg")

    # Best Opt+Fl departure (may differ from best Factory NF!)
    best_opt_idx = np.argmin(fuel_opt_fl)
    best_opt_date = dates[best_opt_idx]
    print(f"\n     Best Opt+Fl departure: {best_opt_date.strftime('%Y-%m-%d')}  "
          f"fuel = {fuel_opt_fl[best_opt_idx]:.0f} kg  "
          f"(Hs = {mean_hs[best_opt_idx]:.2f} m, "
          f"wind = {mean_wind[best_opt_idx]:.1f} m/s, "
          f"Fl thrust = {mean_F_fl[best_opt_idx]:.1f} kN)")
    if best_opt_idx != best_idx:
        print(f"     NOTE: best Opt+Fl departure differs from best Factory NF "
              f"departure!")
        print(f"           Factory NF prefers calm; Opt+Fl can prefer windy "
              f"(Flettner benefit).")

    n_above_p95 = np.sum(fuel_fac_nf > p95_fuel)
    print(f"\n     Departures above P95 ({p95_fuel:.0f} kg): {n_above_p95}")
    print(f"     With 7-day flexibility, {n_above_p95}->{{}}-of-these avoidable:".format(
        "most" if n_above_p95 > 5 else "some"))

    # Actually compute: within each 7-day window, could the P95 departures
    # have been avoided?
    w = 7
    n_avoidable = 0
    n_p95_checked = 0
    for i in range(n):
        if fuel_fac_nf[i] > p95_fuel:
            n_p95_checked += 1
            # Check if any day in [i-w+1 .. i+w-1] (clamped) is below P95
            lo = max(0, i - w + 1)
            hi = min(n, i + w)
            if np.min(fuel_fac_nf[lo:hi]) <= p95_fuel:
                n_avoidable += 1
    if n_p95_checked > 0:
        print(f"     With +/-{w} day flexibility: {n_avoidable}/{n_p95_checked} "
              f"P95 departures avoidable "
              f"({100 * n_avoidable / n_p95_checked:.0f}%)")
