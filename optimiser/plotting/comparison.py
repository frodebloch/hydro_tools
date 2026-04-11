"""Summary and comparison plots for voyage results."""

from pathlib import Path

import numpy as np

from models.route import ROUTE_ROTTERDAM_GOTHENBURG
from simulation.results import VoyageResult
from .map_plot import plot_route


# Output directory: optimiser/ (two levels up from plotting/)
_OUT_DIR = Path(__file__).parent.parent


def plot_results(results: list[VoyageResult]):
    """Generate summary plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    dates = [r.departure for r in results]
    savings_pct = np.array([r.saving_pct for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_wind = np.array([r.mean_wind for r in results])
    mean_R_aw = np.array([r.mean_R_aw_kN for r in results])
    mean_F_flett = np.array([r.mean_F_flettner_kN for r in results])

    # Split saving percentages (relative to factory-no-Flettner baseline)
    fuel_factory_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_factory = np.array([r.total_fuel_factory_kg for r in results])
    sav_pr_pct = np.where(fuel_factory_nf > 0,
                          np.array([r.saving_pitch_rpm_kg for r in results])
                          / fuel_factory_nf * 100.0, 0.0)
    sav_fl_pct = np.where(fuel_factory_nf > 0,
                          np.array([r.saving_flettner_kg for r in results])
                          / fuel_factory_nf * 100.0, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Split saving: stacked area (pitch/RPM + Flettner)
    # All percentages relative to factory-no-Flettner baseline for
    # consistent additive decomposition.
    sav_total_pct = sav_pr_pct + sav_fl_pct
    ax = axes[0]
    ax.fill_between(dates, 0, sav_pr_pct, alpha=0.6, color="steelblue",
                    label=f"Propeller optimisation ({np.mean(sav_pr_pct):.1f}% mean)")
    ax.fill_between(dates, sav_pr_pct, sav_total_pct,
                    alpha=0.6, color="coral",
                    label=f"Wind-assist ({np.mean(sav_fl_pct):.1f}% mean)")
    ax.plot(dates, sav_total_pct, "k-", linewidth=0.5, alpha=0.6,
            label=f"Total ({np.mean(sav_total_pct):.1f}% mean)")
    ax.axhline(np.mean(sav_total_pct), color="k", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Fuel saving [% of standard baseline]")
    ax.set_title("Optimiser vs Standard Control: Daily Fuel Saving")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Significant wave height
    ax = axes[1]
    ax.plot(dates, mean_hs, "g-", linewidth=0.7, alpha=0.8)
    ax.set_ylabel("Mean Hs [m]")
    ax.set_title("Voyage-Mean Significant Wave Height")
    ax.grid(True, alpha=0.3)

    # 3. Added resistance and Flettner thrust
    ax = axes[2]
    ax.plot(dates, mean_R_aw, "r-", linewidth=0.7, alpha=0.8, label="Wave added resistance")
    ax.plot(dates, mean_F_flett, "b-", linewidth=0.7, alpha=0.8, label="Rotor thrust")
    ax.set_ylabel("[kN]")
    ax.set_title("Voyage-Mean Wave Resistance and Rotor Thrust")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    out_path = _OUT_DIR / "voyage_comparison_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")

    # Scatter: Saving vs Hs (separate figure)
    fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
    sc = ax_sc.scatter(mean_hs, sav_total_pct, s=15, alpha=0.5,
                       c=mean_wind, cmap="viridis")
    ax_sc.set_xlabel("Mean Hs [m]")
    ax_sc.set_ylabel("Fuel saving [% of standard baseline]")
    ax_sc.set_title("Fuel Saving vs Sea State (baseline = standard control, colour = wind speed)")
    ax_sc.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax_sc, label="Wind speed [m/s]")
    plt.tight_layout()
    out_path_sc = _OUT_DIR / "voyage_comparison_scatter.png"
    plt.savefig(out_path_sc, dpi=150)
    print(f"Scatter saved: {out_path_sc}")

    # Histogram: separate subplots for each saving component
    fig2, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(16, 5))

    # Total saving
    ax_h1.hist(savings_pct, bins=30, edgecolor="black", alpha=0.7,
               color="grey")
    ax_h1.axvline(np.mean(savings_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(savings_pct):.1f}%")
    ax_h1.set_xlabel("Fuel saving [%]")
    ax_h1.set_ylabel("Number of voyages")
    ax_h1.set_title("Total Saving")
    ax_h1.legend(fontsize=9)
    ax_h1.grid(True, alpha=0.3)

    # Propeller optimisation saving
    ax_h2.hist(sav_pr_pct, bins=30, edgecolor="black", alpha=0.7,
               color="steelblue")
    ax_h2.axvline(np.mean(sav_pr_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(sav_pr_pct):.1f}%")
    ax_h2.set_xlabel("Fuel saving [%]")
    ax_h2.set_title("Propeller Optimisation")
    ax_h2.legend(fontsize=9)
    ax_h2.grid(True, alpha=0.3)

    # Wind-assist saving
    ax_h3.hist(sav_fl_pct, bins=30, edgecolor="black", alpha=0.7,
               color="coral")
    ax_h3.axvline(np.mean(sav_fl_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(sav_fl_pct):.1f}%")
    ax_h3.set_xlabel("Fuel saving [%]")
    ax_h3.set_title("Wind-Assist (Flettner Rotor)")
    ax_h3.legend(fontsize=9)
    ax_h3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = _OUT_DIR / "voyage_comparison_histogram.png"
    plt.savefig(out_path2, dpi=150)
    print(f"Histogram saved: {out_path2}")

    # Route map
    plot_route(ROUTE_ROTTERDAM_GOTHENBURG)


def plot_comparison(results_std: list[VoyageResult],
                    results_sg: list[VoyageResult],
                    speed_kn: float,
                    idle_pct: float = 15.0):
    """Generate comparison plots: standard vs shaft-generator mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available; skipping comparison plots.")
        return

    out_dir = _OUT_DIR

    # --- Helper: extract arrays from result list ---
    def _arrays(results):
        d = {}
        d["dates"] = [r.departure for r in results]
        d["fuel_fac_nf"] = np.array([r.total_fuel_factory_no_flettner_kg
                                      for r in results])
        d["fuel_fac_fl"] = np.array([r.total_fuel_factory_kg for r in results])
        d["fuel_opt_nf"] = np.array([r.total_fuel_opt_no_flettner_kg
                                      for r in results])
        d["fuel_opt_fl"] = np.array([r.total_fuel_optimised_kg
                                      for r in results])
        d["sav_pr_kg"] = np.array([r.saving_pitch_rpm_kg for r in results])
        d["sav_fl_kg"] = np.array([r.saving_flettner_kg for r in results])
        d["sav_pct"] = np.array([r.saving_pct for r in results])
        d["mean_hs"] = np.array([r.mean_hs for r in results])
        d["mean_wind"] = np.array([r.mean_wind for r in results])
        d["mean_R_aw"] = np.array([r.mean_R_aw_kN for r in results])
        d["mean_F_fl"] = np.array([r.mean_F_flettner_kN for r in results])
        # Percentage savings (relative to factory no-Fl baseline)
        d["pct_pr"] = np.where(d["fuel_fac_nf"] > 0,
                               d["sav_pr_kg"] / d["fuel_fac_nf"] * 100.0, 0.0)
        d["pct_fl"] = np.where(d["fuel_fac_nf"] > 0,
                               d["sav_fl_kg"] / d["fuel_fac_nf"] * 100.0, 0.0)
        return d

    std = _arrays(results_std)
    sg = _arrays(results_sg)
    transit_h = results_std[0].total_hours
    voyages_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0) / transit_h

    # Consistent colours
    C_PR_STD = "#3574a3"     # steel blue
    C_FL_STD = "#e8774a"     # coral
    C_PR_SG = "#1a4a70"      # dark blue
    C_FL_SG = "#b84520"      # dark coral

    # ==================================================================
    # FIGURE 1: Daily savings time series — side by side
    # ==================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Panel 1: Standard mode stacked area
    ax1.fill_between(std["dates"], 0, std["pct_pr"], alpha=0.55,
                     color=C_PR_STD,
                     label=f"Propeller optimisation ({np.mean(std['pct_pr']):.1f}%)")
    ax1.fill_between(std["dates"], std["pct_pr"],
                     std["pct_pr"] + std["pct_fl"],
                     alpha=0.55, color=C_FL_STD,
                     label=f"Wind-assist ({np.mean(std['pct_fl']):.1f}%)")
    ax1.axhline(np.mean(std["sav_pct"]), color="k", ls="--", lw=0.8,
                alpha=0.5)
    ax1.set_ylabel("Fuel saving [%]")
    ax1.set_title("Standard Mode — Daily Saving (Prop. optimisation + Wind-assist)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Panel 2: SG mode stacked area
    ax2.fill_between(sg["dates"], 0, sg["pct_pr"], alpha=0.55,
                     color=C_PR_SG,
                     label=f"Propeller optimisation ({np.mean(sg['pct_pr']):.1f}%)")
    ax2.fill_between(sg["dates"], sg["pct_pr"],
                     sg["pct_pr"] + sg["pct_fl"],
                     alpha=0.55, color=C_FL_SG,
                     label=f"Wind-assist ({np.mean(sg['pct_fl']):.1f}%)")
    ax2.axhline(np.mean(sg["sav_pct"]), color="k", ls="--", lw=0.8,
                alpha=0.5)
    ax2.set_ylabel("Fuel saving [%]")
    ax2.set_title("Shaft Generator Mode — Daily Saving (Prop. optimisation + Wind-assist)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Match y-axis range across both panels
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    # Panel 3: Sea state (shared context)
    ax3.plot(std["dates"], std["mean_hs"], color="#2e8b57", lw=0.8,
             alpha=0.8, label="Hs")
    ax3_tw = ax3.twinx()
    ax3_tw.plot(std["dates"], std["mean_wind"], color="#666", lw=0.6,
                alpha=0.6, label="Wind")
    ax3.set_ylabel("Mean Hs [m]", color="#2e8b57")
    ax3_tw.set_ylabel("Mean wind [m/s]", color="#666")
    ax3.set_title("Weather Conditions")
    ax3.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    p = out_dir / "comparison_timeseries.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"\nPlot saved: {p}")

    # ==================================================================
    # FIGURE 2: Annualized fuel & savings bar chart
    # ==================================================================
    fig, (ax_fuel, ax_sav) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: annual fuel consumption (4 bars x 2 modes)
    labels = ["Std NR", "Std+R", "Opt NR", "Opt+R"]
    std_vals = np.array([np.mean(std["fuel_fac_nf"]),
                         np.mean(std["fuel_fac_fl"]),
                         np.mean(std["fuel_opt_nf"]),
                         np.mean(std["fuel_opt_fl"])]) * voyages_yr / 1000.0
    sg_vals = np.array([np.mean(sg["fuel_fac_nf"]),
                        np.mean(sg["fuel_fac_fl"]),
                        np.mean(sg["fuel_opt_nf"]),
                        np.mean(sg["fuel_opt_fl"])]) * voyages_yr / 1000.0

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax_fuel.bar(x - w / 2, std_vals, w, label="Standard",
                        color=C_PR_STD, alpha=0.8)
    bars2 = ax_fuel.bar(x + w / 2, sg_vals, w, label="SG mode",
                        color=C_PR_SG, alpha=0.8)
    for bar, val in zip(bars1, std_vals):
        ax_fuel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, sg_vals):
        ax_fuel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax_fuel.set_xticks(x)
    ax_fuel.set_xticklabels(labels)
    ax_fuel.set_ylabel("Fuel [tonnes/year]")
    ax_fuel.set_title(f"Annualized Fuel Consumption\n"
                      f"({idle_pct:.0f}% idle, {voyages_yr:.0f} voyages/yr)")
    ax_fuel.legend()
    ax_fuel.grid(True, alpha=0.3, axis="y")

    # Right: savings breakdown (stacked bars — Prop. opt. + Wind-assist)
    std_pr = np.mean(std["sav_pr_kg"]) * voyages_yr / 1000.0
    std_fl = np.mean(std["sav_fl_kg"]) * voyages_yr / 1000.0
    sg_pr = np.mean(sg["sav_pr_kg"]) * voyages_yr / 1000.0
    sg_fl = np.mean(sg["sav_fl_kg"]) * voyages_yr / 1000.0

    std_fac_nf_ann = np.mean(std["fuel_fac_nf"]) * voyages_yr / 1000.0
    sg_fac_nf_ann = np.mean(sg["fuel_fac_nf"]) * voyages_yr / 1000.0

    x2 = np.arange(2)
    pr_vals = [std_pr, sg_pr]
    fl_vals = [std_fl, sg_fl]
    bars_pr = ax_sav.bar(x2, pr_vals, 0.5, label="Propeller optimisation",
                         color=[C_PR_STD, C_PR_SG], alpha=0.85)
    bars_fl = ax_sav.bar(x2, fl_vals, 0.5, bottom=pr_vals,
                         label="Wind-assist",
                         color=[C_FL_STD, C_FL_SG], alpha=0.85)
    # Annotate totals and percentages
    for i, (pr, fl, baseline) in enumerate(
            zip(pr_vals, fl_vals, [std_fac_nf_ann, sg_fac_nf_ann])):
        total = pr + fl
        pct = 100 * total / baseline if baseline > 0 else 0
        ax_sav.text(i, total + 2, f"{total:.0f} t ({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        # Sub-labels
        ax_sav.text(i, pr / 2, f"Opt\n{pr:.0f} t",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
        ax_sav.text(i, pr + fl / 2, f"WA\n{fl:.0f} t",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")

    ax_sav.set_xticks(x2)
    ax_sav.set_xticklabels(["Standard", "SG mode"])
    ax_sav.set_ylabel("Fuel saving [tonnes/year]")
    ax_sav.set_title("Annual Savings Breakdown\n(vs standard control baseline)")
    ax_sav.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "comparison_annual.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 3: Seasonal per-voyage breakdown (grouped bars)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    quarters = {"Q1\nJan-Mar": (1, 3), "Q2\nApr-Jun": (4, 6),
                "Q3\nJul-Sep": (7, 9), "Q4\nOct-Dec": (10, 12)}
    q_names = list(quarters.keys())

    for ax_idx, (mode_label, res, c_pr, c_fl) in enumerate([
        ("Standard", results_std, C_PR_STD, C_FL_STD),
        ("SG Mode", results_sg, C_PR_SG, C_FL_SG),
    ]):
        ax = axes[ax_idx]
        q_pr_pct = []
        q_fl_pct = []
        q_fac_kg = []
        q_opt_kg = []
        for _, (m1, m2) in quarters.items():
            qr = [r for r in res if m1 <= r.departure.month <= m2]
            fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in qr])
            spr = np.array([r.saving_pitch_rpm_kg for r in qr])
            sfl = np.array([r.saving_flettner_kg for r in qr])
            q_pr_pct.append(np.mean(spr / fac_nf * 100) if np.all(fac_nf > 0) else 0)
            q_fl_pct.append(np.mean(sfl / fac_nf * 100) if np.all(fac_nf > 0) else 0)
            q_fac_kg.append(np.mean(fac_nf))
            q_opt_kg.append(np.mean([r.total_fuel_optimised_kg for r in qr]))

        xq = np.arange(len(q_names))
        ax.bar(xq, q_pr_pct, 0.6, label="Propeller optimisation", color=c_pr, alpha=0.85)
        ax.bar(xq, q_fl_pct, 0.6, bottom=q_pr_pct, label="Wind-assist",
               color=c_fl, alpha=0.85)
        for i in range(len(q_names)):
            total = q_pr_pct[i] + q_fl_pct[i]
            ax.text(i, total + 0.15, f"{total:.1f}%", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(xq)
        ax.set_xticklabels(q_names)
        ax.set_ylabel("Fuel saving [%]")
        ax.set_title(f"{mode_label} — Seasonal Savings")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # Match y-axis
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    plt.tight_layout()
    p = out_dir / "comparison_seasonal.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 4: Scatter — saving vs Hs, both modes
    # ==================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sc1 = ax1.scatter(std["mean_hs"], std["sav_pct"], s=12, alpha=0.5,
                      c=std["mean_wind"], cmap="viridis", vmin=2, vmax=17)
    ax1.set_xlabel("Mean Hs [m]")
    ax1.set_ylabel("Total saving [%]")
    ax1.set_title("Standard Mode")
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(sg["mean_hs"], sg["sav_pct"], s=12, alpha=0.5,
                      c=sg["mean_wind"], cmap="viridis", vmin=2, vmax=17)
    ax2.set_xlabel("Mean Hs [m]")
    ax2.set_title("SG Mode")
    ax2.grid(True, alpha=0.3)

    plt.colorbar(sc2, ax=[ax1, ax2], label="Wind speed [m/s]",
                 shrink=0.8, pad=0.02)
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.08)
    p = out_dir / "comparison_scatter.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 5: Overlaid histograms — saving distributions
    # ==================================================================
    fig, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(16, 5))

    bins_total = np.linspace(0, max(np.max(std["sav_pct"]),
                                     np.max(sg["sav_pct"])) * 1.05, 35)
    ax_h1.hist(std["sav_pct"], bins=bins_total, alpha=0.6,
               color=C_PR_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['sav_pct']):.1f}%)")
    ax_h1.hist(sg["sav_pct"], bins=bins_total, alpha=0.6,
               color=C_PR_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['sav_pct']):.1f}%)")
    ax_h1.axvline(np.mean(std["sav_pct"]), color=C_PR_STD, ls="--", lw=1.2)
    ax_h1.axvline(np.mean(sg["sav_pct"]), color=C_PR_SG, ls="--", lw=1.2)
    ax_h1.set_xlabel("Fuel saving [%]")
    ax_h1.set_ylabel("Number of voyages")
    ax_h1.set_title("Total Saving")
    ax_h1.legend(fontsize=9)
    ax_h1.grid(True, alpha=0.3)

    bins_pr = np.linspace(0, max(np.max(std["pct_pr"]),
                                  np.max(sg["pct_pr"])) * 1.05, 35)
    ax_h2.hist(std["pct_pr"], bins=bins_pr, alpha=0.6,
               color=C_PR_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['pct_pr']):.1f}%)")
    ax_h2.hist(sg["pct_pr"], bins=bins_pr, alpha=0.6,
               color=C_PR_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['pct_pr']):.1f}%)")
    ax_h2.axvline(np.mean(std["pct_pr"]), color=C_PR_STD, ls="--", lw=1.2)
    ax_h2.axvline(np.mean(sg["pct_pr"]), color=C_PR_SG, ls="--", lw=1.2)
    ax_h2.set_xlabel("Propeller optimisation saving [%]")
    ax_h2.set_title("Propeller Optimisation Saving")
    ax_h2.legend(fontsize=9)
    ax_h2.grid(True, alpha=0.3)

    bins_fl = np.linspace(0, max(np.max(std["pct_fl"]),
                                  np.max(sg["pct_fl"])) * 1.05, 35)
    ax_h3.hist(std["pct_fl"], bins=bins_fl, alpha=0.6,
               color=C_FL_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['pct_fl']):.1f}%)")
    ax_h3.hist(sg["pct_fl"], bins=bins_fl, alpha=0.6,
               color=C_FL_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['pct_fl']):.1f}%)")
    ax_h3.axvline(np.mean(std["pct_fl"]), color=C_FL_STD, ls="--", lw=1.2)
    ax_h3.axvline(np.mean(sg["pct_fl"]), color=C_FL_SG, ls="--", lw=1.2)
    ax_h3.set_xlabel("Wind-assist saving [%]")
    ax_h3.set_title("Wind-Assist Saving (Flettner Rotor)")
    ax_h3.legend(fontsize=9)
    ax_h3.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "comparison_histograms.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # Route map (shared — same route for both modes)
    plot_route(ROUTE_ROTTERDAM_GOTHENBURG)
