"""Scheduling analysis text reports and roughness sweep summary."""

import numpy as np

from simulation.results import VoyageResult


def print_scheduling_analysis(
    all_results: dict[float, list[VoyageResult]],
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
    total_windows_days: tuple[float, ...] = (4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 14.0),
):
    """Analyse 2D (departure time, speed) scheduling optimization.

    For a given total scheduling window W (from earliest possible departure
    to latest acceptable arrival), the operator chooses both WHEN to depart
    and at WHAT SPEED for each individual voyage.

    The key insight is that all candidate voyages cover the same distance
    (509 nm one-way, 1018 nm round-trip), so fuel per voyage is a fair
    comparison metric.  The annualization uses a FIXED commercial schedule
    (same number of voyages/year regardless of speed -- the speed flexibility
    is used to optimize individual voyages, not to change throughput).

    Parameters
    ----------
    all_results : dict[float, list[VoyageResult]]
        Per-speed annual results from run_scheduling_analysis().
    total_windows_days : tuple[float, ...]
        Total scheduling windows [days] from earliest departure to latest
        arrival.
    """
    if not all_results:
        print("No results for scheduling analysis.")
        return

    speeds = sorted(all_results.keys())
    voy_label = "round-trip" if round_trip else "one-way"

    # Reference speed (design speed) is the one closest to 10 kn
    ref_speed = min(speeds, key=lambda s: abs(s - 10.0))
    ref_results = all_results[ref_speed]

    # Build day index from ref speed results
    day_indices = {}
    for r in ref_results:
        day_indices[r.departure.date()] = len(day_indices)
    n_common = len(day_indices)

    # Build 2D fuel matrices: fuel[speed_idx, day_idx]
    fuel_fac_nf = np.full((len(speeds), n_common), np.nan)
    fuel_opt_fl = np.full((len(speeds), n_common), np.nan)
    fuel_opt_nf = np.full((len(speeds), n_common), np.nan)
    transit_hours = {}

    for si, spd in enumerate(speeds):
        results = all_results[spd]
        if results:
            transit_hours[spd] = results[0].total_hours
        for r in results:
            di = day_indices.get(r.departure.date())
            if di is not None:
                fuel_fac_nf[si, di] = r.total_fuel_factory_no_flettner_kg
                fuel_opt_fl[si, di] = r.total_fuel_optimised_kg
                fuel_opt_nf[si, di] = r.total_fuel_opt_no_flettner_kg

    # Annualization: use ref speed's voyage rate for ALL comparisons
    # (commercial schedule is fixed, speed flex is per-voyage operational)
    sailing_h_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    ref_t_h = transit_hours[ref_speed]
    ref_vpy = sailing_h_yr / ref_t_h
    ref_si = speeds.index(ref_speed)

    print("\n" + "=" * 78)
    print("SCHEDULING OPTIMIZATION: 2D (DEPARTURE TIME x SPEED)")
    print("=" * 78)

    # ---- 1. Overview: per-voyage fuel at each speed ----
    print(f"\n  1. PER-VOYAGE FUEL AT EACH SPEED (annual mean, no scheduling)")
    print(f"     Transit time and voyages/year shown for reference;\n"
          f"     annualization uses ref schedule ({ref_vpy:.0f} voy/yr @ "
          f"{ref_speed:.0f} kn).\n")

    print(f"     {'Speed':>6s}  {'Transit':>8s}  "
          f"{'FacNF':>8s}  {'Opt+Fl':>8s}  {'Save':>6s}  "
          f"{'Fl sav':>7s}  {'P/R sav':>8s}")
    print(f"     {'[kn]':>6s}  {'[hours]':>8s}  "
          f"{'kg/voy':>8s}  {'kg/voy':>8s}  {'%':>6s}  "
          f"{'kg/voy':>7s}  {'kg/voy':>8s}")
    print(f"     {'-' * 6}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 7}  {'-' * 8}")

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        m_fac = np.nanmean(fuel_fac_nf[si])
        m_opt = np.nanmean(fuel_opt_fl[si])
        m_opt_nf = np.nanmean(fuel_opt_nf[si])
        sav = 100 * (m_fac - m_opt) / m_fac if m_fac > 0 else 0
        fl_sav = m_opt_nf - m_opt
        pr_sav = m_fac - m_opt_nf
        marker = " <-- ref" if spd == ref_speed else ""
        print(f"     {spd:>6.1f}  {t_h:>8.1f}  "
              f"{m_fac:>8.0f}  {m_opt:>8.0f}  {sav:>5.1f}%  "
              f"{fl_sav:>7.0f}  {pr_sav:>8.0f}{marker}")

    # ---- 2. Per-voyage scheduling optimization ----
    # For each window W, for each day in the year, find the best
    # (departure_within_flex, speed) combination that minimizes fuel.
    # This is the key table: per-voyage fuel with 2D optimization.
    print(f"\n  2. PER-VOYAGE 2D SCHEDULING OPTIMIZATION")
    print(f"     For each departure opportunity, the operator chooses the")
    print(f"     best (departure day within flex, speed) to minimize fuel.")
    print(f"     All speeds must fit within the total window.\n")

    ref_mean_fac = np.nanmean(fuel_fac_nf[ref_si])
    ref_mean_opt = np.nanmean(fuel_opt_fl[ref_si])

    print(f"     {'Window':>8s}  {'Feasible speeds':>26s}  "
          f"{'Mean':>8s}  {'Mean':>8s}  {'vs ref':>6s}  "
          f"{'Ann sav':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'':>26s}  "
          f"{'FacNF':>8s}  {'Opt+Fl':>8s}  {'FacNF':>6s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 26}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 10}")

    # Baseline (no scheduling, ref speed)
    base_sav_pct = 100 * (ref_mean_fac - ref_mean_opt) / ref_mean_fac
    print(f"     {'(none)':>8s}  {ref_speed:>25.0f}*  "
          f"{ref_mean_fac:>8.0f}  {ref_mean_opt:>8.0f}  "
          f"{'---':>6s}  {'---':>8s}  {'---':>10s}")

    scheduling_results = []

    for W in total_windows_days:
        # Determine which speeds are feasible for this window
        feasible = []
        for si, spd in enumerate(speeds):
            t_h = transit_hours.get(spd, 0)
            if t_h > 0 and t_h / 24.0 <= W:
                dep_flex = int(W - t_h / 24.0)
                feasible.append((si, spd, dep_flex))

        if not feasible:
            continue

        # For each day, find the best fuel across all feasible
        # (speed, departure_within_flex) combinations
        best_fac_per_day = []
        best_opt_per_day = []
        for d in range(n_common):
            day_best_fac = float("inf")
            day_best_opt = float("inf")
            for si, spd, dep_flex in feasible:
                lo = d
                hi = min(n_common, d + dep_flex + 1)
                # Factory NF: best in window at this speed
                c_fac = fuel_fac_nf[si, lo:hi]
                v_fac = c_fac[~np.isnan(c_fac)]
                if len(v_fac) > 0:
                    day_best_fac = min(day_best_fac, np.min(v_fac))
                # Opt+Fl: best in window at this speed
                c_opt = fuel_opt_fl[si, lo:hi]
                v_opt = c_opt[~np.isnan(c_opt)]
                if len(v_opt) > 0:
                    day_best_opt = min(day_best_opt, np.min(v_opt))
            if day_best_fac < float("inf"):
                best_fac_per_day.append(day_best_fac)
            if day_best_opt < float("inf"):
                best_opt_per_day.append(day_best_opt)

        if not best_opt_per_day:
            continue

        mean_best_fac = np.mean(best_fac_per_day)
        mean_best_opt = np.mean(best_opt_per_day)

        # Saving vs ref baseline (Factory NF at ref speed, no scheduling)
        sav_fac_pct = 100 * (ref_mean_fac - mean_best_fac) / ref_mean_fac
        sav_opt_pct = 100 * (ref_mean_fac - mean_best_opt) / ref_mean_fac
        ann_sav_t = (ref_mean_fac - mean_best_opt) * ref_vpy / 1000
        ann_eur = ann_sav_t * fuel_price_eur_per_t

        spd_str = ", ".join(
            f"{spd:.0f}({dep_flex}d)" for _, spd, dep_flex in feasible)

        print(f"     {W:>8.1f}  {spd_str:>26s}  "
              f"{mean_best_fac:>8.0f}  {mean_best_opt:>8.0f}  "
              f"{sav_opt_pct:>5.1f}%  "
              f"{ann_sav_t:>8.1f}  {ann_eur:>10,.0f}")

        scheduling_results.append({
            "window": W, "feasible": feasible,
            "mean_fac": mean_best_fac, "mean_opt": mean_best_opt,
            "sav_pct": sav_opt_pct, "ann_sav_t": ann_sav_t,
        })

    print(f"\n     Speeds shown as: speed(departure_flexibility_days)")
    print(f"     Saving % vs Factory NF @ {ref_speed:.0f} kn, no scheduling "
          f"({ref_mean_fac:.0f} kg/voy)")

    # ---- 3. Decomposition: what comes from where ----
    print(f"\n  3. SAVING DECOMPOSITION (per voyage, vs Factory NF @ "
          f"{ref_speed:.0f} kn baseline)")
    print(f"     Showing mean kg/voyage saved by each lever.\n")

    print(f"     {'Window':>8s}  {'Speed':>8s}  {'Timing':>8s}  "
          f"{'Pitch/RPM':>10s}  {'Flettner':>9s}  {'Total':>8s}  "
          f"{'Total':>6s}")
    print(f"     {'[days]':>8s}  {'choice':>8s}  {'choice':>8s}  "
          f"{'optim':>10s}  {'':>9s}  {'kg/voy':>8s}  "
          f"{'%':>6s}")
    print(f"     {'-' * 8}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 10}  {'-' * 9}  {'-' * 8}  {'-' * 6}")

    for sr in scheduling_results:
        W = sr["window"]
        feasible = sr["feasible"]

        # (a) Speed-only saving (best constant speed, no timing, Factory NF)
        best_speed_only_fac = float("inf")
        for si, spd, _ in feasible:
            m = np.nanmean(fuel_fac_nf[si])
            if m < best_speed_only_fac:
                best_speed_only_fac = m
        sav_speed = ref_mean_fac - best_speed_only_fac

        # (b) Timing-only saving (ref speed, timing within flex, Factory NF)
        ref_dep_flex = max(0, int(W - ref_t_h / 24.0))
        timing_fac = []
        for d in range(n_common):
            lo, hi = d, min(n_common, d + ref_dep_flex + 1)
            c = fuel_fac_nf[ref_si, lo:hi]
            v = c[~np.isnan(c)]
            if len(v) > 0:
                timing_fac.append(np.min(v))
        sav_timing = ref_mean_fac - np.mean(timing_fac) if timing_fac else 0

        # Total saving from 2D scheduling (FacNF basis)
        sav_2d_fac = ref_mean_fac - sr["mean_fac"]
        # Additional from pitch/RPM + Flettner
        sav_opt = sr["mean_fac"] - sr["mean_opt"]
        # Total
        sav_total = ref_mean_fac - sr["mean_opt"]
        pct_total = 100 * sav_total / ref_mean_fac if ref_mean_fac > 0 else 0

        # Decompose: speed + timing + interaction = sav_2d_fac
        # Then opt+fl on top
        # For the decomposition we separate timing from speed more carefully:
        # sav_2d_fac = sav_speed + sav_timing + interaction
        sav_interaction = sav_2d_fac - sav_speed - sav_timing

        print(f"     {W:>8.1f}  {sav_speed:>8.0f}  "
              f"{sav_timing + sav_interaction:>8.0f}  "
              f"{sr['mean_fac'] - np.nanmean(fuel_opt_nf[ref_si]):>10.0f}  "  # approximate
              f"{sav_opt - (sr['mean_fac'] - np.nanmean(fuel_opt_nf[ref_si])):>9.0f}  "  # approximate
              f"{sav_total:>8.0f}  {pct_total:>5.1f}%")

    # ---- 4. Simpler table: what the operator actually sees ----
    print(f"\n  4. OPERATOR DECISION TABLE")
    print(f"     For a given scheduling window, what is the annual benefit?\n")
    print(f"     {'Window':>8s}  {'No opt':>10s}  {'Opt+Fl':>10s}  "
          f"{'2D sched':>10s}  {'Full 2D':>10s}")
    print(f"     {'[days]':>8s}  {'FacNF':>10s}  {'@{:.0f}kn'.format(ref_speed):>10s}  "
          f"{'FacNF':>10s}  {'Opt+Fl':>10s}")
    print(f"     {'':>8s}  {'t/yr':>10s}  {'t/yr':>10s}  "
          f"{'t/yr':>10s}  {'t/yr':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 10}  {'-' * 10}")

    # Column 1: No optimization at all (Factory NF @ ref speed)
    col1 = ref_mean_fac * ref_vpy / 1000

    # Column 2: Opt+Fl at ref speed, no scheduling
    col2 = ref_mean_opt * ref_vpy / 1000

    print(f"     {'(none)':>8s}  {col1:>10.1f}  {col2:>10.1f}  "
          f"{'---':>10s}  {'---':>10s}")

    for sr in scheduling_results:
        W = sr["window"]
        # Column 3: 2D scheduling, Factory NF
        col3 = sr["mean_fac"] * ref_vpy / 1000
        # Column 4: 2D scheduling, Opt+Fl
        col4 = sr["mean_opt"] * ref_vpy / 1000
        print(f"     {W:>8.1f}  {col1:>10.1f}  {col2:>10.1f}  "
              f"{col3:>10.1f}  {col4:>10.1f}")

    print(f"\n     All in tonnes/year (annualized at {ref_vpy:.0f} voy/yr, "
          f"{idle_pct:.0f}% idle).")
    print(f"     'No opt' = Factory NF @ {ref_speed:.0f} kn, no scheduling "
          f"= {col1:.1f} t/yr")
    print(f"     'Opt+Fl' = Optimiser + Flettner @ {ref_speed:.0f} kn, "
          f"no scheduling = {col2:.1f} t/yr")
    print(f"     '2D sched FacNF' = best (day, speed) per voyage, Factory NF")
    print(f"     'Full 2D Opt+Fl' = best (day, speed) per voyage, Opt+Fl")

    return (scheduling_results, all_results, speeds, fuel_fac_nf,
            fuel_opt_fl, fuel_opt_nf, transit_hours, n_common,
            ref_speed, ref_si, ref_vpy, ref_mean_fac, ref_mean_opt)


def print_roughness_sweep_summary(
    sweep_data: list[tuple],
    speed_kn: float,
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
):
    """Print a summary table comparing results across fouling ages.

    Parameters
    ----------
    sweep_data : list of (age_years, hull_ks_m, blade_ks_m, results)
    """
    if not sweep_data:
        print("No roughness sweep results to summarise.")
        return

    voy_label = "RT" if round_trip else "OW"
    r0 = sweep_data[0][3][0]  # first age, first voyage
    transit_h = r0.total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    vpyr = sailing_hours_year / transit_h

    print(f"\n{'=' * 100}")
    print(f"ROUGHNESS SWEEP SUMMARY  ({speed_kn:.0f} kn, {voy_label}, "
          f"{idle_pct:.0f}% idle -> {vpyr:.0f} voy/yr)")
    print(f"{'=' * 100}")

    fp = fuel_price_eur_per_t

    print(f"\n  {'Age':>4s}  {'Hull ks':>8s}  {'Blade ks':>9s}  "
          f"{'\u0394R_hull':>8s}  {'BladeFac':>8s}  "
          f"{'Fac_NF':>9s}  {'Opt+Fl':>9s}  {'Save':>6s}  "
          f"{'P/R %':>6s}  {'Fl %':>6s}  "
          f"{'\u20ac saving':>10s}")
    print(f"  {'[yr]':>4s}  {'[\u00b5m]':>8s}  {'[\u00b5m]':>9s}  "
          f"{'[kN]':>8s}  {'[-]':>8s}  "
          f"{'[t/yr]':>9s}  {'[t/yr]':>9s}  {'[%]':>6s}  "
          f"{'':>6s}  {'':>6s}  "
          f"{'[\u20ac/yr]':>10s}")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 9}  "
          f"{'-' * 8}  {'-' * 8}  "
          f"{'-' * 9}  {'-' * 9}  {'-' * 6}  "
          f"{'-' * 6}  {'-' * 6}  "
          f"{'-' * 10}")

    for age, h_ks, b_ks, results in sweep_data:
        fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in results])
        opt_fl = np.mean([r.total_fuel_optimised_kg for r in results])
        sav_pr = np.mean([r.saving_pitch_rpm_kg for r in results])
        sav_fl = np.mean([r.saving_flettner_kg for r in results])
        sav_tot = sav_pr + sav_fl

        ann_fac_nf = fac_nf * vpyr / 1000.0
        ann_opt_fl = opt_fl * vpyr / 1000.0
        ann_sav_tot = sav_tot * vpyr / 1000.0

        pct_tot = 100.0 * sav_tot / fac_nf if fac_nf > 0 else 0.0
        pct_pr = 100.0 * sav_pr / fac_nf if fac_nf > 0 else 0.0
        pct_fl = 100.0 * sav_fl / fac_nf if fac_nf > 0 else 0.0

        r_rough = results[0].R_roughness_kN
        b_fac = results[0].blade_fuel_factor
        cost_sav = ann_sav_tot * fp

        print(f"  {age:4.1f}  {h_ks * 1e6:8.0f}  {b_ks * 1e6:9.0f}  "
              f"{r_rough:8.1f}  {b_fac:8.3f}  "
              f"{ann_fac_nf:9.1f}  {ann_opt_fl:9.1f}  {pct_tot:6.1f}  "
              f"{pct_pr:6.1f}  {pct_fl:6.1f}  "
              f"{cost_sav:10,.0f}")

    # Compare clean vs. dirtiest
    if len(sweep_data) >= 2:
        _, _, _, r_clean = sweep_data[0]
        _, _, _, r_dirty = sweep_data[-1]
        clean_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in r_clean])
        dirty_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in r_dirty])
        fuel_increase_pct = 100.0 * (dirty_fac_nf - clean_fac_nf) / clean_fac_nf if clean_fac_nf > 0 else 0.0
        print(f"\n  Fuel increase from {sweep_data[0][0]:.0f} yr to "
              f"{sweep_data[-1][0]:.0f} yr fouling: "
              f"{fuel_increase_pct:+.1f}% (factory no-Flettner baseline)")

    print()
