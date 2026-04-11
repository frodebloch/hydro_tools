"""Scheduling analysis plots (speed x departure window optimisation)."""

import numpy as np

from simulation.results import VoyageResult


def plot_scheduling_analysis(
    all_results: dict[float, list[VoyageResult]],
    scheduling_data: tuple,
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
):
    """Generate scheduling analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping scheduling plots.")
        return

    (sched_results, _, speeds, fuel_fac_nf, fuel_opt_fl,
     fuel_opt_nf, transit_hours, n_common,
     ref_speed, ref_si, ref_vpy, ref_mean_fac, ref_mean_opt) = scheduling_data

    sailing_h_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voy_label = "round-trip" if round_trip else "one-way"

    # ---- Figure 1: Per-voyage fuel at each speed ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Speed & Scheduling — Per-Voyage Fuel — {voy_label}",
                 fontsize=14, fontweight="bold")

    mean_fac = [np.nanmean(fuel_fac_nf[si]) for si in range(len(speeds))]
    mean_opt = [np.nanmean(fuel_opt_fl[si]) for si in range(len(speeds))]

    ax = axes[0]
    ax.plot(speeds, mean_fac, "C3o-", label="Standard control", markersize=8)
    ax.plot(speeds, mean_opt, "C0o-", label="Optimiser + rotor", markersize=8)
    ax.set_xlabel("Transit speed [kn]")
    ax.set_ylabel("Fuel per voyage [kg]")
    ax.set_title("Mean fuel per voyage vs speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Figure 1b: Best per-voyage fuel with 2D scheduling ----
    windows = [3, 4, 5, 6, 7, 8, 10, 14]
    ax = axes[1]

    # For each window, compute the mean best per-voyage Opt+Fl fuel
    # (best across all feasible speeds and departure days within flex)
    mean_best_opt_by_window = []
    mean_best_fac_by_window = []
    valid_windows = []
    for W in windows:
        feasible = [(si, spd, max(0, int(W - transit_hours.get(spd, 999) / 24.0)))
                     for si, spd in enumerate(speeds)
                     if transit_hours.get(spd, 999) / 24.0 <= W]
        if not feasible:
            continue
        best_opt = []
        best_fac = []
        for d in range(n_common):
            d_best_opt = float("inf")
            d_best_fac = float("inf")
            for si, spd, dep_flex in feasible:
                lo, hi = d, min(n_common, d + dep_flex + 1)
                c_opt = fuel_opt_fl[si, lo:hi]
                v_opt = c_opt[~np.isnan(c_opt)]
                if len(v_opt) > 0:
                    d_best_opt = min(d_best_opt, np.min(v_opt))
                c_fac = fuel_fac_nf[si, lo:hi]
                v_fac = c_fac[~np.isnan(c_fac)]
                if len(v_fac) > 0:
                    d_best_fac = min(d_best_fac, np.min(v_fac))
            if d_best_opt < float("inf"):
                best_opt.append(d_best_opt)
            if d_best_fac < float("inf"):
                best_fac.append(d_best_fac)
        mean_best_opt_by_window.append(np.mean(best_opt))
        mean_best_fac_by_window.append(np.mean(best_fac))
        valid_windows.append(W)

    ax.plot(valid_windows, mean_best_fac_by_window, "C3o-",
            label="Best 2D (standard control)", markersize=7)
    ax.plot(valid_windows, mean_best_opt_by_window, "C0o-",
            label="Best 2D (Opt+rotor)", markersize=7)
    ax.axhline(ref_mean_fac, color="C3", linestyle="--", alpha=0.5,
               label=f"Std @ {ref_speed:.0f} kn = {ref_mean_fac:.0f}")
    ax.axhline(ref_mean_opt, color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+R @ {ref_speed:.0f} kn = {ref_mean_opt:.0f}")
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Mean fuel per voyage [kg]")
    ax.set_title("Per-voyage fuel with 2D scheduling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("scheduling_speed_window.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved: scheduling_speed_window.png")

    # ---- Figure 2: Heatmap — per-voyage fuel (speed x window) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Per-Voyage Opt+Rotor Fuel — Speed x Window — {voy_label}",
                 fontsize=14, fontweight="bold")

    windows_fine = [3, 4, 5, 6, 7, 8, 10, 12, 14]

    # Left: per-speed, best-in-window per-voyage fuel
    heat_voy = np.full((len(speeds), len(windows_fine)), np.nan)
    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        if t_h <= 0:
            continue
        for wi, W in enumerate(windows_fine):
            if W < t_h / 24.0:
                continue
            dep_flex = max(0, int(W - t_h / 24.0))
            bl = []
            for d in range(n_common):
                lo, hi = d, min(n_common, d + dep_flex + 1)
                c = fuel_opt_fl[si, lo:hi]
                v = c[~np.isnan(c)]
                if len(v) > 0:
                    bl.append(np.min(v))
            if bl:
                heat_voy[si, wi] = np.mean(bl)

    ax = axes[0]
    im = ax.imshow(heat_voy / 1000, aspect="auto", origin="lower",
                   cmap="RdYlGn_r",
                   extent=[windows_fine[0] - 0.5, windows_fine[-1] + 0.5,
                           speeds[0] - 0.25, speeds[-1] + 0.25])
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Transit speed [kn]")
    ax.set_title("Mean Opt+Rotor fuel per voyage [tonnes]")
    plt.colorbar(im, ax=ax, label="Fuel [t/voy]", shrink=0.8)

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        for wi, W in enumerate(windows_fine):
            val = heat_voy[si, wi]
            if not np.isnan(val):
                ax.text(W, spd, f"{val / 1000:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val > np.nanmedian(heat_voy[~np.isnan(heat_voy)]) else "black")
            elif t_h > 0 and W < t_h / 24.0:
                ax.text(W, spd, "X", ha="center", va="center",
                        fontsize=10, color="gray", alpha=0.5)
    ax.set_xticks(windows_fine)
    ax.set_yticks(speeds)

    # Right: saving % vs ref baseline
    heat_sav = np.full_like(heat_voy, np.nan)
    for si in range(len(speeds)):
        for wi in range(len(windows_fine)):
            if not np.isnan(heat_voy[si, wi]):
                heat_sav[si, wi] = 100 * (ref_mean_fac - heat_voy[si, wi]) / ref_mean_fac

    ax = axes[1]
    im2 = ax.imshow(heat_sav, aspect="auto", origin="lower",
                    cmap="RdYlGn",
                    extent=[windows_fine[0] - 0.5, windows_fine[-1] + 0.5,
                            speeds[0] - 0.25, speeds[-1] + 0.25])
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Transit speed [kn]")
    ax.set_title(f"Saving vs standard @ {ref_speed:.0f} kn [%]")
    plt.colorbar(im2, ax=ax, label="Saving [%]", shrink=0.8)

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        for wi, W in enumerate(windows_fine):
            val = heat_sav[si, wi]
            if not np.isnan(val):
                ax.text(W, spd, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val < np.nanmedian(heat_sav[~np.isnan(heat_sav)]) else "black")
            elif t_h > 0 and W < t_h / 24.0:
                ax.text(W, spd, "X", ha="center", va="center",
                        fontsize=10, color="gray", alpha=0.5)
    ax.set_xticks(windows_fine)
    ax.set_yticks(speeds)

    fig.tight_layout()
    fig.savefig("scheduling_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: scheduling_heatmap.png")

    plt.close("all")
