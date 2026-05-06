"""Empirical first-order lag estimation: brucon commanded sway thrust -> applied.

Three transfers of interest in brucon's actuator pipeline:
    OrderTauSway   :  controller command (FB + bias FF + integral)
    AllocTauSway   :  after thrust allocator solves for individual thrusters
                      and re-sums to a body-frame net force command
    Ty (SwayThrust):  actually-applied force on the vessel (after rate-limited
                      azimuths, RPM ramping, thruster dynamics)

Hypothesis: physical actuator dynamics (azimuth turn-rate, RPM ramp) act on
slow-band signals like a first-order lag with time constant tau. If true,
    H(jw) = Ty / OrderTau  ~  1 / (1 + jw*tau)   =>   |H|^2 = 1/(1+(w*tau)^2).

Fit tau from ensemble-averaged |H(jw)|^2 over the slow band (f < 0.05 Hz)
using 30 P7 waves-only seeds in the intact window. Reports both the
allocator-only transfer (Order -> Alloc) and the full physical transfer
(Order -> Ty) for comparison.

Usage
-----
    .venv/bin/python scripts/p7_brucon_validation/estimate_thrust_lag.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_sandbox_timeseries import load_seed_csv  # noqa: E402

T_START = 300.0
T_END = 555.0
DT = 0.1
F_FIT_LO = 1e-3   # Hz, lower edge of fitting band
F_FIT_HI = 0.05   # Hz, upper edge (above this, allocator nonlinearity / wave-rejection dominates)


def first_order_mag2(omega: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + (omega * tau) ** 2)


def fit_tau(f: np.ndarray, H2: np.ndarray, lo: float, hi: float) -> float:
    """Least-squares fit of |H|^2 = 1/(1+(2 pi f tau)^2) on band [lo,hi]."""
    m = (f >= lo) & (f <= hi)
    fb = f[m]
    H2b = H2[m]
    # Linearise: 1/H2 - 1 = (2 pi f tau)^2  =>  sqrt(1/H2 - 1) = 2 pi f tau (slope through 0)
    # But avoid dividing by tiny H2; use weighted nonlinear fit instead.
    from scipy.optimize import minimize_scalar

    def cost(tau):
        pred = first_order_mag2(2 * np.pi * fb, tau)
        return np.sum((np.log(pred) - np.log(H2b)) ** 2)

    res = minimize_scalar(cost, bounds=(0.1, 60.0), method="bounded")
    return float(res.x)


def main() -> None:
    work = Path(__file__).parent / "work"
    seeds = sorted(work.glob("pwo_seed*"))
    print(f"Found {len(seeds)} seeds")

    pairs = {
        "Order_to_Alloc": ("OrderTauSway", "AllocTauSway"),
        "Order_to_Ty":    ("OrderTauSway", "Ty"),       # full physical transfer
        "Alloc_to_Ty":    ("AllocTauSway", "Ty"),       # actuator-only
    }
    psd_acc: dict[str, dict[str, list[np.ndarray]]] = {
        k: {"Pxx": [], "Pyy": [], "Pxy": []} for k in pairs
    }

    for sd in seeds:
        try:
            cols = load_seed_csv(sd)
        except Exception as e:
            print(f"  skip {sd.name}: {e}")
            continue
        t = cols["t"]
        m = (t >= T_START) & (t <= T_END)
        nps = 1024
        for label, (in_name, out_name) in pairs.items():
            u = cols[in_name][m]; u = u - u.mean()
            y = cols[out_name][m]; y = y - y.mean()
            f, Pxx = scipy.signal.welch(u, fs=1 / DT, nperseg=nps)
            _, Pyy = scipy.signal.welch(y, fs=1 / DT, nperseg=nps)
            _, Pxy = scipy.signal.csd(u, y, fs=1 / DT, nperseg=nps)
            psd_acc[label]["Pxx"].append(Pxx)
            psd_acc[label]["Pyy"].append(Pyy)
            psd_acc[label]["Pxy"].append(Pxy)
    n_seeds = len(psd_acc["Order_to_Alloc"]["Pxx"])
    print(f"Used {n_seeds} seeds\n")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    summary = {}
    for label, (in_name, out_name) in pairs.items():
        Pxx = np.mean(psd_acc[label]["Pxx"], axis=0)
        Pyy = np.mean(psd_acc[label]["Pyy"], axis=0)
        Pxy = np.mean(psd_acc[label]["Pxy"], axis=0)
        H = Pxy / np.where(Pxx > 0, Pxx, np.nan)
        H_mag2 = (np.abs(H)) ** 2
        H_phase_deg = np.degrees(np.angle(H))
        coh2 = (np.abs(Pxy) ** 2) / (Pxx * Pyy)
        band = (f >= F_FIT_LO) & (f <= F_FIT_HI) & (coh2 > 0.5)
        if band.sum() < 4:
            band = (f >= F_FIT_LO) & (f <= F_FIT_HI)
        tau = fit_tau(f[band], H_mag2[band], F_FIT_LO, F_FIT_HI)
        f_c = 1.0 / (2 * np.pi * tau)
        idx = np.argmin(np.abs(f - f_c))
        summary[label] = (tau, f_c, H_mag2[idx], H_phase_deg[idx], band.sum())
        print(f"{label:<18} {in_name} -> {out_name}")
        print(f"  fitted tau = {tau:.2f} s  (fit pts: {band.sum()})")
        print(f"  predicted f_c = {f_c:.4f} Hz; observed |H|^2 there = {H_mag2[idx]:.3f}; phase = {H_phase_deg[idx]:.1f} deg\n")

    # Plotting per-pair
    colors = {"Order_to_Alloc": "C0", "Order_to_Ty": "C1", "Alloc_to_Ty": "C2"}
    ax = axes[0]
    for label, (in_name, out_name) in pairs.items():
        Pxx = np.mean(psd_acc[label]["Pxx"], axis=0)
        Pxy = np.mean(psd_acc[label]["Pxy"], axis=0)
        H_mag2 = np.abs(Pxy / np.where(Pxx > 0, Pxx, np.nan)) ** 2
        tau = summary[label][0]
        ax.loglog(f, H_mag2, color=colors[label], lw=1.2,
                  label=f"{label}: tau_fit={tau:.2f}s")
        ax.loglog(f, first_order_mag2(2*np.pi*f, tau), color=colors[label], ls="--", alpha=0.6)
    ax.loglog(f, first_order_mag2(2*np.pi*f, 5.0), "k:", alpha=0.5, label="ref tau=5s")
    ax.axvspan(F_FIT_LO, F_FIT_HI, alpha=0.1, color="gray", label="fit band")
    ax.set_ylabel("|H|^2"); ax.set_ylim(1e-3, 2.0)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Brucon thrust-pipeline transfers (n={n_seeds} P7 seeds, intact window)")

    ax = axes[1]
    for label in pairs:
        Pxx = np.mean(psd_acc[label]["Pxx"], axis=0)
        Pxy = np.mean(psd_acc[label]["Pxy"], axis=0)
        H = Pxy / np.where(Pxx > 0, Pxx, np.nan)
        ax.semilogx(f, np.degrees(np.angle(H)), color=colors[label], lw=1.2, label=label)
        tau = summary[label][0]
        f_th = np.logspace(-3, 0, 200)
        ax.semilogx(f_th, np.degrees(-np.arctan(2*np.pi*f_th*tau)),
                    color=colors[label], ls="--", alpha=0.5)
    ax.axvspan(F_FIT_LO, F_FIT_HI, alpha=0.1, color="gray")
    ax.set_ylabel("phase [deg]"); ax.set_ylim(-180, 30)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for label in pairs:
        Pxx = np.mean(psd_acc[label]["Pxx"], axis=0)
        Pyy = np.mean(psd_acc[label]["Pyy"], axis=0)
        Pxy = np.mean(psd_acc[label]["Pxy"], axis=0)
        coh2 = (np.abs(Pxy) ** 2) / (Pxx * Pyy)
        ax.semilogx(f, coh2, color=colors[label], lw=1.2, label=label)
    ax.axhline(0.5, color="r", ls=":", alpha=0.5)
    ax.axvspan(F_FIT_LO, F_FIT_HI, alpha=0.1, color="gray")
    ax.set_ylabel("coherence^2"); ax.set_xlabel("f [Hz]")
    ax.set_ylim(0, 1.05); ax.set_xlim(1e-3, 0.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    out = Path(__file__).parent / "thrust_lag_estimate.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
