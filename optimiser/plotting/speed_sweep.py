"""Speed sensitivity sweep plots."""

from pathlib import Path

import numpy as np

from simulation.results import SpeedSweepResult


# Output directory: optimiser/ (two levels up from plotting/)
_OUT_DIR = Path(__file__).parent.parent


def plot_speed_sweep(sweep: list[SpeedSweepResult]):
    """Generate speed sensitivity plots.

    Produces two figures:
    1. speed_sweep_fuel.png -- Fuel consumption & savings vs speed
    2. speed_sweep_savings.png -- Savings breakdown vs speed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping speed sweep plots")
        return

    if len(sweep) < 2:
        print("Need at least 2 speeds for sweep plots — skipping")
        return

    out_dir = _OUT_DIR
    speeds = [s.speed_kn for s in sweep]

    # ---- Figure 1: Fuel & savings overview (3-panel) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Per-voyage fuel consumption
    ax = axes[0]
    ax.plot(speeds, [s.mean_fuel_factory_nf_kg for s in sweep],
            "s-", color="#c0392b", linewidth=2, markersize=8,
            label="Factory (no Flettner)")
    ax.plot(speeds, [s.mean_fuel_opt_nf_kg for s in sweep],
            "^-", color="#e67e22", linewidth=2, markersize=8,
            label="Optimiser (no Flettner)")
    ax.plot(speeds, [s.mean_fuel_opt_fl_kg for s in sweep],
            "o-", color="#27ae60", linewidth=2, markersize=8,
            label="Optimiser + Flettner")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Fuel per voyage [kg]")
    ax.set_title("Per-Voyage Fuel Consumption")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Add percentage annotations
    for s in sweep:
        ax.annotate(f"{s.mean_saving_pct:+.1f}%",
                     (s.speed_kn, s.mean_fuel_opt_fl_kg),
                     textcoords="offset points", xytext=(0, -18),
                     fontsize=8, ha="center", color="#27ae60",
                     fontweight="bold")

    # Panel 2: Annualized fuel
    ax = axes[1]
    ax.plot(speeds, [s.ann_fuel_factory_nf_t for s in sweep],
            "s-", color="#c0392b", linewidth=2, markersize=8,
            label="Factory (no Flettner)")
    ax.plot(speeds, [s.ann_fuel_opt_fl_t for s in sweep],
            "o-", color="#27ae60", linewidth=2, markersize=8,
            label="Optimiser + Flettner")
    # Shade the saving region
    ax.fill_between(speeds,
                     [s.ann_fuel_opt_fl_t for s in sweep],
                     [s.ann_fuel_factory_nf_t for s in sweep],
                     alpha=0.15, color="#27ae60")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Fuel [tonnes/year]")
    ax.set_title("Annualized Fuel Consumption")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Annotate annual saving
    for s in sweep:
        mid_y = (s.ann_fuel_factory_nf_t + s.ann_fuel_opt_fl_t) / 2
        ax.annotate(f"{s.ann_saving_total_t:.0f} t/yr",
                     (s.speed_kn, mid_y),
                     fontsize=8, ha="center", color="#27ae60",
                     fontweight="bold")

    # Panel 3: Saving percentages (stacked)
    ax = axes[2]
    bar_w = 0.6 * min(np.diff(speeds)) if len(speeds) > 1 else 0.6
    bar_w = min(bar_w, 0.8)
    pr_pcts = [s.pct_pitch_rpm for s in sweep]
    fl_pcts = [s.pct_flettner for s in sweep]
    bars1 = ax.bar(speeds, pr_pcts, width=bar_w,
                    color="#3574a3", label="Pitch/RPM optimisation")
    bars2 = ax.bar(speeds, fl_pcts, width=bar_w, bottom=pr_pcts,
                    color="#e8774a", label="Flettner wind assist")
    # Labels on bars
    for i, s in enumerate(sweep):
        total = s.pct_pitch_rpm + s.pct_flettner
        ax.text(s.speed_kn, total + 0.3, f"{total:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving [% of factory baseline]")
    ax.set_title("Savings Breakdown by Speed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    plt.tight_layout()
    p = out_dir / "speed_sweep_fuel.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ---- Figure 2: Savings detail (2-panel) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Per-voyage savings in kg (stacked bar)
    ax = axes[0]
    pr_kg = [s.mean_saving_pitch_rpm_kg for s in sweep]
    fl_kg = [s.mean_saving_flettner_kg for s in sweep]
    ax.bar(speeds, pr_kg, width=bar_w,
           color="#3574a3", label="Pitch/RPM")
    ax.bar(speeds, fl_kg, width=bar_w, bottom=pr_kg,
           color="#e8774a", label="Flettner")
    for i, s in enumerate(sweep):
        total = s.mean_saving_total_kg
        ax.text(s.speed_kn, total + 5, f"{total:.0f} kg",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving per voyage [kg]")
    ax.set_title("Per-Voyage Fuel Saving")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    # Panel 2: Annualized savings in tonnes (stacked bar)
    ax = axes[1]
    pr_t = [s.ann_saving_pitch_rpm_t for s in sweep]
    fl_t = [s.ann_saving_flettner_t for s in sweep]
    ax.bar(speeds, pr_t, width=bar_w,
           color="#3574a3", label="Pitch/RPM")
    ax.bar(speeds, fl_t, width=bar_w, bottom=pr_t,
           color="#e8774a", label="Flettner")
    for i, s in enumerate(sweep):
        total = s.ann_saving_total_t
        ax.text(s.speed_kn, total + 1, f"{total:.0f} t/yr",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving [tonnes/year]")
    ax.set_title("Annualized Fuel Saving")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    plt.tight_layout()
    p = out_dir / "speed_sweep_savings.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")
