"""Departure timing analysis plots."""

import numpy as np

from simulation.results import VoyageResult


def plot_departure_analysis(results: list[VoyageResult], speed_kn: float,
                            round_trip: bool = True):
    """Generate departure timing analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available; skipping departure analysis plots.")
        return

    n = len(results)
    dates = [r.departure for r in results]
    fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    sav_flettner_kg = fuel_opt_nf - fuel_opt_fl
    sav_total_kg = fuel_fac_nf - fuel_opt_fl
    mean_wind = np.array([r.mean_wind for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_F_fl = np.array([r.mean_F_flettner_kN for r in results])

    voy_label = "round-trip" if round_trip else "one-way"

    # ---- Figure 1: Fuel time series with scheduling windows ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Departure Timing Analysis — {speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    # Panel 1: Fuel per voyage for both cases
    ax = axes[0]
    ax.plot(dates, fuel_fac_nf, "C3-", alpha=0.7, linewidth=0.8,
            label="Factory NF")
    ax.plot(dates, fuel_opt_fl, "C0-", alpha=0.7, linewidth=0.8,
            label="Opt+Flettner")
    # 7-day rolling min for opt+fl (achievable with 7-day flexibility)
    rolling_min = np.array([np.min(fuel_opt_fl[max(0, i - 3):min(n, i + 4)])
                            for i in range(n)])
    ax.plot(dates, rolling_min, "C2--", linewidth=1.2,
            label="Best Opt+Fl in 7-day window")
    ax.set_ylabel("Fuel [kg/voyage]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Flettner saving
    ax = axes[1]
    ax.bar(dates, sav_flettner_kg, width=1.0, color="C0", alpha=0.6,
           label="Flettner saving")
    ax.set_ylabel("Flettner saving [kg]")
    ax.axhline(np.mean(sav_flettner_kg), color="C0", linestyle="--",
               linewidth=1, alpha=0.8, label=f"Mean = {np.mean(sav_flettner_kg):.0f} kg")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean wind speed
    ax = axes[2]
    ax.bar(dates, mean_wind, width=1.0, color="C7", alpha=0.6)
    ax.set_ylabel("Mean wind [m/s]")
    ax.axhline(np.mean(mean_wind), color="k", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 4: Mean Hs
    ax = axes[3]
    ax.bar(dates, mean_hs, width=1.0, color="C4", alpha=0.6)
    ax.set_ylabel("Mean Hs [m]")
    ax.axhline(np.mean(mean_hs), color="k", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Departure date (2024)")

    fig.tight_layout()
    fig.savefig("departure_timing_timeseries.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved: departure_timing_timeseries.png")

    # ---- Figure 2: Scatter matrix — correlations ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Weather vs Fuel & Savings Correlations — "
                 f"{speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    # Scatter 1: Wind vs Flettner saving
    ax = axes[0, 0]
    sc = ax.scatter(mean_wind, sav_flettner_kg, c=mean_hs, cmap="viridis",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean wind speed [m/s]")
    ax.set_ylabel("Flettner saving [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_wind, sav_flettner_kg)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Hs [m]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Scatter 2: Wind vs total fuel (Factory NF)
    ax = axes[0, 1]
    sc = ax.scatter(mean_wind, fuel_fac_nf, c=mean_hs, cmap="viridis",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean wind speed [m/s]")
    ax.set_ylabel("Factory NF fuel [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_wind, fuel_fac_nf)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Hs [m]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Scatter 3: Hs vs Opt+Fl fuel (does Flettner reduce weather sensitivity?)
    ax = axes[1, 0]
    ax.scatter(mean_hs, fuel_fac_nf, s=12, alpha=0.5, color="C3",
               label="Factory NF")
    ax.scatter(mean_hs, fuel_opt_fl, s=12, alpha=0.5, color="C0",
               label="Opt+Flettner")
    ax.set_xlabel("Mean Hs [m]")
    ax.set_ylabel("Fuel [kg/voyage]")
    ax.set_title("Weather sensitivity: Factory vs Opt+Fl")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter 4: Mean Flettner thrust vs Flettner fuel saving
    ax = axes[1, 1]
    sc = ax.scatter(mean_F_fl, sav_flettner_kg, c=mean_wind, cmap="plasma",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean Flettner thrust [kN]")
    ax.set_ylabel("Flettner fuel saving [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_F_fl, sav_flettner_kg)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Wind [m/s]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("departure_timing_correlations.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: departure_timing_correlations.png")

    # ---- Figure 3: Scheduling window value ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Value of Scheduling Flexibility — {speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    windows = [1, 2, 3, 5, 7, 10, 14, 21, 28]
    # Left: mean best fuel within window (Factory NF and Opt+Fl)
    mean_best_fac = []
    mean_best_opt = []
    for w in windows:
        b_fac = [np.min(fuel_fac_nf[max(0, i - w // 2):min(n, i + (w + 1) // 2)])
                 for i in range(n)]
        b_opt = [np.min(fuel_opt_fl[max(0, i - w // 2):min(n, i + (w + 1) // 2)])
                 for i in range(n)]
        mean_best_fac.append(np.mean(b_fac))
        mean_best_opt.append(np.mean(b_opt))

    ax = axes[0]
    ax.plot(windows, mean_best_fac, "C3o-", label="Factory NF (best in window)")
    ax.plot(windows, mean_best_opt, "C0o-", label="Opt+Fl (best in window)")
    ax.axhline(np.mean(fuel_fac_nf), color="C3", linestyle="--", alpha=0.5,
               label=f"Factory NF mean = {np.mean(fuel_fac_nf):.0f}")
    ax.axhline(np.mean(fuel_opt_fl), color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+Fl mean = {np.mean(fuel_opt_fl):.0f}")
    ax.set_xlabel("Scheduling window [days]")
    ax.set_ylabel("Mean achievable fuel [kg/voyage]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: saving % vs baseline (Factory NF mean) for different windows
    ax = axes[1]
    sav_pct_fac = [100 * (1 - v / np.mean(fuel_fac_nf)) for v in mean_best_fac]
    sav_pct_opt = [100 * (1 - v / np.mean(fuel_fac_nf)) for v in mean_best_opt]
    sav_pct_opt_base = 100 * (1 - np.mean(fuel_opt_fl) / np.mean(fuel_fac_nf))
    ax.plot(windows, sav_pct_fac, "C3o-",
            label="Timing only (Factory NF)")
    ax.plot(windows, sav_pct_opt, "C0o-",
            label="Timing + Optimiser + Flettner")
    ax.axhline(sav_pct_opt_base, color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+Fl no timing = {sav_pct_opt_base:.1f}%")
    ax.axhline(0, color="C3", linestyle="--", alpha=0.5,
               label="Factory NF no timing = 0%")
    ax.set_xlabel("Scheduling window [days]")
    ax.set_ylabel("Saving vs Factory NF mean [%]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("departure_timing_windows.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: departure_timing_windows.png")

    plt.close("all")
