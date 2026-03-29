"""
Long-term hindcast-based voyage comparison: factory combinator vs optimiser.

Simulates repeated voyages (e.g. one departure per day over a year) on a
fixed route, using NORA3 wave hindcast data and PdStrip drift transfer
functions to compute realistic thrust demands at each hourly position.
Compares fuel consumption between the factory combinator schedule and the
fuel-optimal pitch/RPM selection.

Includes a Flettner rotor model (ported from BruCon C++) to quantify
wind-assist thrust reduction on the propeller.

Usage:
    # 1. Pre-download NORA3 data for the route (one-time):
    python voyage_comparison.py --download --year 2024

    # 2. Run the annual comparison:
    python voyage_comparison.py --year 2024

    # 3. Run with plots:
    python voyage_comparison.py --year 2024 --plot
"""

import argparse
from pathlib import Path

from models.constants import NORA3_DATA_DIR, PDSTRIP_DAT
from models.roughness import BLADE_FOULING, FOULING_LOW
from models.route import ROUTE_ROTTERDAM_GOTHENBURG
from models.weather import download_nora3_for_route

from simulation.orchestrator import (
    run_annual_comparison,
    run_scheduling_analysis,
    run_speed_sweep,
)

from reporting.summary import print_summary, print_departure_analysis
from reporting.scheduling import (
    print_scheduling_analysis,
    print_roughness_sweep_summary,
)
from reporting.speed_sweep import print_speed_sweep_summary

from plotting.departure import plot_departure_analysis
from plotting.scheduling import plot_scheduling_analysis
from plotting.comparison import plot_results, plot_comparison
from plotting.speed_sweep import plot_speed_sweep


def main():
    parser = argparse.ArgumentParser(
        description="Long-term hindcast voyage comparison: "
                    "factory combinator vs optimiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--year", type=int, default=2024,
                        help="Hindcast year (default: 2024)")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Transit speed [kn] (default: 10)")
    parser.add_argument("--download", action="store_true",
                        help="Download NORA3 data for the route (run first)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="NORA3 data directory (default: ./data/nora3_route)")
    parser.add_argument("--pdstrip", type=str, default=PDSTRIP_DAT,
                        help="Path to PdStrip .dat file")
    parser.add_argument("--no-flettner", action="store_true",
                        help="Disable Flettner rotor (waves only)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate summary plots")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-voyage output")
    parser.add_argument("--sg-load", type=float, default=0.0,
                        help="Actual shaft generator load [kW] for the "
                             "optimiser (default: 0)")
    parser.add_argument("--sg-factory-allowance", type=float, default=0.0,
                        help="SG power allowance in factory combinator "
                             "schedule [kW] (default: 0)")
    parser.add_argument("--sg-freq-min", type=float, default=0.0,
                        help="SG minimum frequency [Hz] (default: 0, "
                             "no constraint)")
    parser.add_argument("--sg-freq-max", type=float, default=0.0,
                        help="SG maximum frequency [Hz] (default: 0, "
                             "no constraint)")
    parser.add_argument("--idle-pct", type=float, default=15.0,
                        help="Percentage of year spent idle/in port "
                             "(default: 15). Used to annualize fuel figures.")
    parser.add_argument("--compare-sg", action="store_true",
                        help="Run both standard and SG modes, then produce "
                             "side-by-side comparison plots. Requires "
                             "--sg-load and --sg-factory-allowance.")
    parser.add_argument("--speed-sweep", nargs="+", type=float, default=None,
                        metavar="KN",
                        help="Run a speed sensitivity sweep at multiple "
                             "speeds [kn]. Example: --speed-sweep 8 10 12 14. "
                             "Produces comparison tables and plots.")
    parser.add_argument("--fuel-price", type=float, default=650.0,
                        help="Fuel price [EUR/tonne] for cost estimates "
                             "(default: 650, typical MGO in ECA zone)")
    parser.add_argument("--one-way", action="store_true",
                        help="Evaluate only the outbound leg (default is "
                             "round-trip to eliminate directional wind bias)")
    parser.add_argument("--hull-roughness", type=float, default=0.0,
                        metavar="KS_UM",
                        help="Hull roughness ks [um]. 0=clean (default). "
                             "Typical: 30=new AF, 150=2yr Nordic, 500=5yr.")
    parser.add_argument("--blade-roughness", type=float, default=0.0,
                        metavar="KS_UM",
                        help="Propeller blade roughness ks [um]. 0=clean "
                             "(default). Typical: 10=polished, 70=3yr Nordic.")
    parser.add_argument("--fouling-years", type=float, default=None,
                        metavar="T",
                        help="Time since last cleaning [years]. Uses "
                             "FOULING_LOW for hull and BLADE_FOULING for "
                             "propeller. Overrides --hull/blade-roughness.")
    parser.add_argument("--roughness-sweep", nargs="+", type=float,
                        default=None, metavar="T",
                        help="Run annual comparison at multiple fouling ages "
                             "[years], e.g. --roughness-sweep 0 1 2 3 4 5")
    parser.add_argument("--departure-analysis", action="store_true",
                        help="Analyse the value of departure timing "
                             "flexibility (implies --plot)")
    parser.add_argument("--scheduling-analysis", nargs="+", type=float,
                        default=None, metavar="KN",
                        help="Run 2D (departure time x speed) scheduling "
                             "optimization. Specify candidate speeds, e.g. "
                             "--scheduling-analysis 8 9 10 11 12")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else NORA3_DATA_DIR
    do_round_trip = not args.one_way

    # Resolve roughness parameters
    hull_ks_m = args.hull_roughness * 1e-6  # um -> m
    blade_ks_m = args.blade_roughness * 1e-6
    if args.fouling_years is not None:
        hull_ks_m = FOULING_LOW.ks_m_at(args.fouling_years)
        blade_ks_m = BLADE_FOULING.ks_m_at(args.fouling_years)

    if args.download:
        download_nora3_for_route(args.year, ROUTE_ROTTERDAM_GOTHENBURG,
                                 data_dir)
        return

    if args.roughness_sweep:
        ages = sorted(set(args.roughness_sweep))
        print(f"\n{'#' * 78}")
        print(f"# ROUGHNESS SWEEP: fouling ages = "
              f"{', '.join(f'{a:.1f}' for a in ages)} years")
        print(f"{'#' * 78}\n")
        sweep_summaries = []
        for i, age in enumerate(ages):
            h_ks = FOULING_LOW.ks_m_at(age)
            b_ks = BLADE_FOULING.ks_m_at(age)
            print(f"\n{'=' * 78}")
            print(f"  Fouling age: {age:.1f} yr  "
                  f"hull ks={h_ks * 1e6:.0f} um, blade ks={b_ks * 1e6:.0f} um"
                  f"  ({i + 1}/{len(ages)})")
            print(f"{'=' * 78}\n")
            results = run_annual_comparison(
                year=args.year,
                speed_kn=args.speed,
                data_dir=data_dir,
                pdstrip_path=args.pdstrip,
                flettner_enabled=not args.no_flettner,
                verbose=not args.quiet,
                sg_load_kw=args.sg_load,
                sg_factory_allowance_kw=args.sg_factory_allowance,
                sg_freq_min=args.sg_freq_min,
                sg_freq_max=args.sg_freq_max,
                round_trip=do_round_trip,
                hull_ks_m=h_ks,
                blade_ks_m=b_ks,
            )
            if results:
                sweep_summaries.append((age, h_ks, b_ks, results))
        print_roughness_sweep_summary(sweep_summaries, args.speed,
                                      idle_pct=args.idle_pct,
                                      fuel_price_eur_per_t=args.fuel_price,
                                      round_trip=do_round_trip)
        return

    if args.scheduling_analysis:
        sched_speeds = sorted(set(args.scheduling_analysis))
        print(f"\n{'#' * 78}")
        print(f"# SCHEDULING ANALYSIS: speeds = "
              f"{', '.join(f'{s:.0f}' for s in sched_speeds)} kn")
        print(f"{'#' * 78}\n")
        all_results = run_scheduling_analysis(
            speeds=sched_speeds,
            year=args.year,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            round_trip=do_round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
        )
        sched_data = print_scheduling_analysis(
            all_results, idle_pct=args.idle_pct,
            fuel_price_eur_per_t=args.fuel_price,
            round_trip=do_round_trip,
        )
        if sched_data:
            plot_scheduling_analysis(
                all_results, sched_data,
                idle_pct=args.idle_pct,
                fuel_price_eur_per_t=args.fuel_price,
                round_trip=do_round_trip,
            )
        return

    if args.speed_sweep:
        speeds = sorted(set(args.speed_sweep))
        print(f"\n{'#' * 78}")
        print(f"# SPEED SENSITIVITY SWEEP: {', '.join(f'{s:.0f}' for s in speeds)} kn")
        print(f"{'#' * 78}\n")
        sweep = run_speed_sweep(
            speeds=speeds,
            year=args.year,
            idle_pct=args.idle_pct,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
            round_trip=do_round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
        )
        print_speed_sweep_summary(sweep, idle_pct=args.idle_pct,
                                  fuel_price_eur_per_t=args.fuel_price)
        plot_speed_sweep(sweep)
        return

    if args.compare_sg:
        # Run both standard and SG modes
        print("\n" + "#" * 78)
        print("# STANDARD MODE (no shaft generator)")
        print("#" * 78 + "\n")
        results_std = run_annual_comparison(
            year=args.year,
            speed_kn=args.speed,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            round_trip=do_round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
        )
        print_summary(results_std, args.speed, idle_pct=args.idle_pct,
                      fuel_price_eur_per_t=args.fuel_price,
                      round_trip=do_round_trip)

        print("\n\n" + "#" * 78)
        print("# SHAFT GENERATOR MODE")
        print("#" * 78 + "\n")
        results_sg = run_annual_comparison(
            year=args.year,
            speed_kn=args.speed,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
            round_trip=do_round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
        )
        print_summary(results_sg, args.speed, idle_pct=args.idle_pct,
                      fuel_price_eur_per_t=args.fuel_price,
                      round_trip=do_round_trip)

        plot_comparison(results_std, results_sg, args.speed,
                        idle_pct=args.idle_pct)
        return

    results = run_annual_comparison(
        year=args.year,
        speed_kn=args.speed,
        data_dir=data_dir,
        pdstrip_path=args.pdstrip,
        flettner_enabled=not args.no_flettner,
        verbose=not args.quiet,
        sg_load_kw=args.sg_load,
        sg_factory_allowance_kw=args.sg_factory_allowance,
        sg_freq_min=args.sg_freq_min,
        sg_freq_max=args.sg_freq_max,
        round_trip=do_round_trip,
        hull_ks_m=hull_ks_m,
        blade_ks_m=blade_ks_m,
    )

    print_summary(results, args.speed, idle_pct=args.idle_pct,
                  fuel_price_eur_per_t=args.fuel_price,
                  round_trip=do_round_trip)

    if args.departure_analysis:
        print_departure_analysis(results, args.speed, idle_pct=args.idle_pct,
                                 fuel_price_eur_per_t=args.fuel_price,
                                 round_trip=do_round_trip)
        plot_departure_analysis(results, args.speed,
                                round_trip=do_round_trip)

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
