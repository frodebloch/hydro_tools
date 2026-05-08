"""Decompose the brucon delivered-thrust deficit at pwq30:

Question raised by user (paraphrased): we are trying to predict the MEAN
LF envelope of the post-WCF excursion. The driver of that envelope is
not the raw instantaneous thrust deficit (which contains wave-frequency
controller activity) but the LF component of the realised force on the
hull. The cqa scenario tau_lost = -(1-beta)*tau_env, beta=1+(gamma-1)*
exp(-t/T) is meant to be the LF DRIVE -- the cqa-27 closed-loop
response (LF observer + LF controller) is responsible for the rest of
the dynamics, including any controller-integral overshoot.

If that view is right, then:

  1. The brucon tau_lost(t) = Order_pre - T(t) trace (which I plotted
     in calibrate_wcfdi_scenario_pwq30) contains WAVE-FREQUENCY content
     from the closed-loop controller fighting waves.
  2. LP-filtering the brucon trace to the LF band should give a clean
     initial-spike + monotone-decay shape that the parametric scenario
     CAN fit, with sensible (gamma, T_realloc) parameters.
  3. The 'overshoot' I saw at +10-25 s in the raw trace would then be
     either (a) wave-frequency content that the cqa-27 model has built-
     in rejection for, or (b) a real LF closed-loop transient that
     should EMERGE from cqa-27 when driven with a clean exponential
     tau_lost.

This script:

  1. Loads per-seed brucon tau_lost(t).
  2. Splits into LF and WF components via a low-pass filter at
     omega_c = 0.5 * omega_p ~ 2*pi/(2*Tp) (cut off at 2x wave period).
     Tp ~ 10 s -> cut at 20 s -> omega_c = 0.31 rad/s.
  3. Plots LF, WF, and total deficits per axis with the cqa scenario
     overlay.
  4. Refits the parametric cqa scenario against the LF component only.
  5. Drives the cqa-27 model with the parametric scenario and overlays
     the predicted T(t) to see whether the cqa closed-loop response
     reproduces the brucon overshoot (validation of the user's
     hypothesis).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))


WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
DT = 0.1
T_PRE_WIN = (T_WCF - 30.0, T_WCF - 5.0)
T_PLOT_PRE = 5.0
T_PLOT_POST = 80.0

T_FIT_END = 60.0

AXES = [("surge", "OrderTauSurge", "Tx"),
        ("sway",  "OrderTauSway",  "Ty"),
        ("yaw",   "OrderTauYaw",   "Tz")]


# Low-pass cut: 2x wave period at Tp~10s -> 1/20 Hz = 0.05 Hz.
LP_CUTOFF_HZ = 0.05
LP_ORDER = 4


def _load_seed(seed: int):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    if main_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    data = np.loadtxt(main_p, skiprows=1, delimiter="\t")
    cols = {h: data[:, i] for i, h in enumerate(hdr)}
    if cols["t"][-1] < T_WCF + T_PLOT_POST:
        return None
    return cols


def lp_filter(x: np.ndarray, fs_hz: float):
    b, a = butter(LP_ORDER, LP_CUTOFF_HZ / (0.5 * fs_hz), btype="low")
    return filtfilt(b, a, x)


def main():
    seed_data = []
    for s in SEEDS:
        c = _load_seed(s)
        if c is None:
            continue
        seed_data.append((s, c))
    print(f"Loaded {len(seed_data)} seeds")

    t_grid = np.arange(-T_PLOT_PRE, T_PLOT_POST + DT / 2, DT)
    fs_hz = 1.0 / DT

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col")

    fits = {}
    for k, (axis, ord_col, t_col) in enumerate(AXES):
        tau_env_per_seed = []
        tl_raw_arr = np.zeros((len(seed_data), len(t_grid)))
        tl_lf_arr = np.zeros_like(tl_raw_arr)
        tl_wf_arr = np.zeros_like(tl_raw_arr)

        for j, (_, c) in enumerate(seed_data):
            t_rel = c["t"] - T_WCF
            order = c[ord_col]
            T_act = c[t_col]
            pre_mask = (c["t"] >= T_PRE_WIN[0]) & (c["t"] <= T_PRE_WIN[1])
            order_pre = float(order[pre_mask].mean())
            tau_env_per_seed.append(-order_pre)

            # Raw deficit referenced to pre-WCF demand
            tau_lost_raw = order_pre - T_act
            # LP-filter THE FULL TRACE on the original time grid (filtfilt
            # does its own padding), then interp.
            tau_lost_lf = lp_filter(tau_lost_raw, fs_hz)
            tau_lost_wf = tau_lost_raw - tau_lost_lf
            tl_raw_arr[j] = np.interp(t_grid, t_rel, tau_lost_raw)
            tl_lf_arr[j] = np.interp(t_grid, t_rel, tau_lost_lf)
            tl_wf_arr[j] = np.interp(t_grid, t_rel, tau_lost_wf)

        tau_env_mean = float(np.mean(tau_env_per_seed))
        tl_raw_mean = tl_raw_arr.mean(axis=0)
        tl_lf_mean = tl_lf_arr.mean(axis=0)
        tl_wf_mean = tl_wf_arr.mean(axis=0)
        tl_lf_p5 = np.percentile(tl_lf_arr, 5, axis=0)
        tl_lf_p95 = np.percentile(tl_lf_arr, 95, axis=0)

        # Fit parametric cqa scenario to the LF mean only, post-WCF.
        post_mask = (t_grid >= 0.0) & (t_grid <= T_FIT_END)
        t_fit = t_grid[post_mask]
        f_fit = tl_lf_mean[post_mask] / (-tau_env_mean)

        def model(t, gamma, T_re):
            return (1.0 - gamma) * np.exp(-t / max(T_re, 1e-3))

        try:
            popt, _ = curve_fit(model, t_fit, f_fit,
                                p0=[0.5, 10.0],
                                bounds=([-2.0, 0.5], [1.5, 100.0]))
            gamma_fit, T_fit = float(popt[0]), float(popt[1])
        except Exception as exc:
            print(f"  {axis}: fit failed: {exc}")
            gamma_fit, T_fit = 0.5, 10.0

        f_model = model(t_grid, gamma_fit, T_fit)
        f_default = model(t_grid, 0.5, 10.0)
        rms_lf = float(np.sqrt(np.mean((tl_lf_mean[post_mask] - (-tau_env_mean) * f_model[post_mask]) ** 2)))
        rms_default = float(np.sqrt(np.mean((tl_lf_mean[post_mask] - (-tau_env_mean) * f_default[post_mask]) ** 2)))
        fits[axis] = dict(gamma=gamma_fit, T=T_fit, rms=rms_lf, rms_default=rms_default,
                          tau_env=tau_env_mean)

        print(f"\n--- axis: {axis} ---")
        print(f"  -tau_env (ensemble mean)      : {-tau_env_mean:+.2f} kN")
        print(f"  raw tau_lost peak (ens. mean) : {tl_raw_mean[np.argmax(np.abs(tl_raw_mean))]:+.2f} kN "
              f"at t={t_grid[np.argmax(np.abs(tl_raw_mean))]:+.1f} s")
        print(f"  LF  tau_lost peak (ens. mean) : {tl_lf_mean[np.argmax(np.abs(tl_lf_mean))]:+.2f} kN "
              f"at t={t_grid[np.argmax(np.abs(tl_lf_mean))]:+.1f} s")
        print(f"  LF/raw peak ratio             : "
              f"{abs(tl_lf_mean[np.argmax(np.abs(tl_lf_mean))])/max(abs(tl_raw_mean[np.argmax(np.abs(tl_raw_mean))]), 1e-6):.2f}")
        print(f"  fitted (LF only) gamma_imm = {gamma_fit:+.3f}  T_realloc = {T_fit:.2f} s   "
              f"RMS = {rms_lf:.3f}  vs default {rms_default:.3f}")

        # ---- left column: LF vs raw breakdown ----
        ax = axes[k, 0]
        ax.fill_between(t_grid, tl_lf_p5, tl_lf_p95, alpha=0.15, color="C0",
                        label="LF 5-95%")
        ax.plot(t_grid, tl_raw_mean, color="0.5", lw=0.8, alpha=0.6,
                label="raw ens. mean")
        ax.plot(t_grid, tl_lf_mean, color="C0", lw=2,
                label=f"LF (LP {LP_CUTOFF_HZ:.2f} Hz) ens. mean")
        ax.plot(t_grid, tl_wf_mean, color="C2", lw=0.8, alpha=0.6,
                label="WF residual ens. mean")
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_ylabel(f"tau_lost {axis} [kN" + ("m]" if axis == "yaw" else "]"))
        ax.set_title(f"{axis}: raw vs LP-filtered ensemble-mean deficit")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # ---- right column: parametric fit on LF ----
        ax = axes[k, 1]
        ax.fill_between(t_grid, tl_lf_p5, tl_lf_p95, alpha=0.15, color="C0",
                        label="brucon LF 5-95%")
        ax.plot(t_grid, tl_lf_mean, color="C0", lw=2, label="brucon LF ens. mean")
        ax.plot(t_grid, -tau_env_mean * f_model, color="C3", lw=2, ls="--",
                label=f"fit on LF: gamma={gamma_fit:.2f}, T={T_fit:.1f}s "
                      f"(RMS {rms_lf:.2f})")
        ax.plot(t_grid, -tau_env_mean * f_default, color="C2", lw=1.5, ls=":",
                label=f"cqa default 0.5/10s (RMS {rms_default:.2f})")
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_ylabel(f"tau_lost {axis} [kN" + ("m]" if axis == "yaw" else "]"))
        ax.set_title(f"{axis}: parametric scenario fit on LF only")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("t - t_WCF [s]")
    axes[-1, 1].set_xlabel("t - t_WCF [s]")

    plt.suptitle(f"Brucon tau_lost LF/WF decomposition + scenario refit at {TAG}",
                 fontsize=12)
    plt.tight_layout()
    out = THIS / "calibrate_wcfdi_scenario_lf_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    print(f"\n=== Summary (LF-only fits) ===")
    for axis, f in fits.items():
        print(f"  {axis:5s} : gamma_imm = {f['gamma']:+.3f}   "
              f"T_realloc = {f['T']:6.2f} s   "
              f"-tau_env = {-f['tau_env']:+8.2f} kN")
    g_xy = float(np.mean([fits["surge"]["gamma"], fits["sway"]["gamma"]]))
    T_xy = float(np.mean([fits["surge"]["T"], fits["sway"]["T"]]))
    print(f"\n  surge+sway average : gamma_imm = {g_xy:.3f}   T_realloc = {T_xy:.2f} s")
    print(f"  (cqa default 0.500 / 10.00 s)")


if __name__ == "__main__":
    main()
