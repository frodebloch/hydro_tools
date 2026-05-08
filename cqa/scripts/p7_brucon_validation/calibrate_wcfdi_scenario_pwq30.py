"""Calibrate WcfdiScenario gamma_immediate / T_realloc against brucon
delivered-thrust profile at pwq30.

Background
----------
The cqa WCFDI scenario models the post-failure thrust deficit as

    tau_lost(t) = -(1 - beta(t)) * tau_env       (t >= 0)
    beta(t)     = 1 + (gamma_imm - 1) * exp(-t / T_realloc)

where ``tau_env = b_hat = -OrderTau_pre`` is the mean environmental
load the vessel experiences pre-WCF. beta(0) = gamma_imm = "fraction
of the env load the surviving thrusters can carry immediately after
the failure"; beta(infty) = 1 = "the allocator has fully reallocated
to the surviving thrusters and the deficit has been recovered".

Equivalently, normalising by ``-tau_env`` (so the curve starts at
``1 - gamma_imm`` at t=0 and decays to 0):

    f(t) := tau_lost(t) / (-tau_env) = (1 - gamma_imm) * exp(-t / T_realloc)

The brucon-realised deficit per seed::

    tau_lost_brucon(t) = OrderTau(t) - T(t)       (t >= t_WCF)

is the difference between what the controller asked for and what the
hull actually received -- this includes RPM spool-down, allocator
re-tasking and saturation. The cqa scenario is a 2-parameter
exponential approximation of that.

This script:

  1. Loads the per-seed tau_lost(t) for surge, sway, yaw at pwq30
     (same extraction as check_obs_perseed_taulost.py).
  2. Computes the ensemble-mean tau_lost(t) and the per-seed
     pre-WCF mean OrderTau (= -tau_env).
  3. Fits gamma_imm and T_realloc per axis via least-squares on f(t).
  4. Compares the fitted exponential against the actual ensemble
     mean and reports the RMS residual.
  5. Plots brucon vs scenario fit per axis, with the cqa default
     (gamma_imm=0.5, T_realloc=10 s) overlay for reference.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/calibrate_wcfdi_scenario_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))


WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
DT = 0.1

T_PRE_WIN = (T_WCF - 30.0, T_WCF - 5.0)        # window for tau_env reference
T_FIT_END = 60.0                                # fit window: [0, 60] s after WCF
T_PLOT_END = 100.0                              # plot horizon

CQA_DEFAULT_GAMMA = 0.5
CQA_DEFAULT_T = 10.0

AXES = [("surge", "OrderTauSurge", "Tx"),
        ("sway",  "OrderTauSway",  "Ty"),
        ("yaw",   "OrderTauYaw",   "Tz")]


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
    if cols["t"][-1] < T_WCF + 30.0:
        return None
    return cols


def main():
    seed_data = []
    for s in SEEDS:
        c = _load_seed(s)
        if c is None:
            continue
        seed_data.append((s, c))
    print(f"Loaded {len(seed_data)} seeds")

    # Common time grid relative to t_WCF.
    t_grid = np.arange(0.0, T_PLOT_END + DT / 2, DT)

    # Per-axis: stack per-seed tau_lost(t) (kN units throughout) and
    # per-seed -tau_env (= OrderTau pre-WCF mean, kN).
    fits = {}
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    for k, (axis, ord_col, t_col) in enumerate(AXES):
        tau_env_per_seed = []                       # kN
        tau_lost_arr = np.zeros((len(seed_data), len(t_grid)))   # kN
        for j, (_, c) in enumerate(seed_data):
            t_rel = c["t"] - T_WCF
            order = c[ord_col]      # kN
            T_act = c[t_col]        # kN
            deficit = order - T_act  # kN; positive = controller commanded more than was delivered

            # tau_env_pre = -OrderTau pre = -mean(order in T_PRE_WIN)
            pre_mask = (c["t"] >= T_PRE_WIN[0]) & (c["t"] <= T_PRE_WIN[1])
            order_pre = float(order[pre_mask].mean())
            tau_env_pre = -order_pre  # kN
            tau_env_per_seed.append(tau_env_pre)

            # cqa convention: tau_lost(t) = Order_pre - T(t)
            #   (Order goes back to its pre-WCF value as the allocator
            #   recovers; T tracks. The brucon controller may reissue a
            #   larger Order to fight wave excursions, but the cqa
            #   model abstracts that into a single delivered-fraction
            #   curve. We fit against the realised hull deficit
            #   referenced to the pre-WCF demand.)
            # In the linear scenario, tau_lost = Order_pre - T,
            # because cqa pretends the steady-state nu_dot = 0 was
            # exactly carried by the pre-WCF thrust.
            tau_lost = order_pre - T_act    # kN
            tau_lost_arr[j] = np.interp(t_grid, t_rel, tau_lost)

        tau_env_per_seed = np.array(tau_env_per_seed)
        tau_env_mean = float(tau_env_per_seed.mean())     # kN, signed

        # Ensemble mean and 5-95% bands
        tl_mean = tau_lost_arr.mean(axis=0)
        tl_p5 = np.percentile(tau_lost_arr, 5, axis=0)
        tl_p95 = np.percentile(tau_lost_arr, 95, axis=0)

        # Normalised: f(t) = tau_lost(t) / (-tau_env)
        # Sign convention: f(0+) = 1 - gamma_imm > 0 (deficit positive
        # in the same sign as -tau_env when most thrust is lost).
        # tau_env is signed, so f(t) is signed too; we want the ratio
        # tau_lost / (-tau_env), which gives a curve that starts at
        # (1 - gamma) and decays to 0 regardless of axis sign.
        f_t = tl_mean / (-tau_env_mean)
        # Fit on the post-WCF window [0, T_FIT_END]
        fit_mask = (t_grid >= 0.0) & (t_grid <= T_FIT_END)
        t_fit = t_grid[fit_mask]
        y_fit = f_t[fit_mask]

        def model(t, gamma, T_re):
            return (1.0 - gamma) * np.exp(-t / max(T_re, 1e-3))

        # Initial guess: cqa default
        try:
            p0 = [CQA_DEFAULT_GAMMA, CQA_DEFAULT_T]
            popt, _ = curve_fit(model, t_fit, y_fit, p0=p0,
                                bounds=([-2.0, 0.5], [1.5, 200.0]))
            gamma_fit, T_fit = float(popt[0]), float(popt[1])
        except Exception as exc:
            print(f"  {axis}: fit failed: {exc}")
            gamma_fit, T_fit = CQA_DEFAULT_GAMMA, CQA_DEFAULT_T

        f_fit = model(t_grid, gamma_fit, T_fit)
        f_default = model(t_grid, CQA_DEFAULT_GAMMA, CQA_DEFAULT_T)

        rms_fit = float(np.sqrt(np.mean((f_t[fit_mask] - f_fit[fit_mask]) ** 2)))
        rms_default = float(np.sqrt(np.mean((f_t[fit_mask] - f_default[fit_mask]) ** 2)))

        peak_idx = int(np.argmax(np.abs(tl_mean)))
        print(f"\n--- axis: {axis} ---")
        print(f"  -tau_env (-OrderTau_pre, ensemble mean) : {-tau_env_mean:+.2f} kN  "
              f"(seed std {tau_env_per_seed.std():.2f} kN)")
        print(f"  tau_lost ensemble peak                  : {tl_mean[peak_idx]:+.2f} kN at "
              f"t={t_grid[peak_idx]:+.1f} s")
        print(f"  tau_lost / (-tau_env) at t=0+           : {f_t[1]:.3f}  -> "
              f"gamma_imm_data ~ {1.0 - f_t[1]:.3f}")
        print(f"  fitted gamma_imm = {gamma_fit:+.3f}   T_realloc = {T_fit:6.2f} s   "
              f"RMS residual = {rms_fit:.4f}")
        print(f"  cqa default      = {CQA_DEFAULT_GAMMA:+.3f}   "
              f"           = {CQA_DEFAULT_T:6.2f} s   RMS residual = {rms_default:.4f}")

        fits[axis] = dict(gamma=gamma_fit, T=T_fit, tau_env=tau_env_mean,
                          rms_fit=rms_fit, rms_default=rms_default)

        # ---- plot ----
        ax = axes[k]
        ax.fill_between(t_grid, tl_p5, tl_p95, alpha=0.15, color="C0",
                        label="brucon 5-95%")
        ax.plot(t_grid, tl_mean, color="C0", lw=2,
                label=f"brucon ensemble mean ({len(seed_data)})")
        ax.plot(t_grid, -tau_env_mean * f_fit, color="C3", lw=2, ls="--",
                label=f"fit: gamma={gamma_fit:.2f}, T={T_fit:.1f}s "
                      f"(RMS resid {rms_fit:.3f})")
        ax.plot(t_grid, -tau_env_mean * f_default, color="C2", lw=1.5, ls=":",
                label=f"cqa default: gamma={CQA_DEFAULT_GAMMA:.2f}, "
                      f"T={CQA_DEFAULT_T:.1f}s  (RMS resid {rms_default:.3f})")
        ax.axhline(0, color="k", lw=0.3)
        ax.axhline(-tau_env_mean * (1.0 - gamma_fit), color="C3", lw=0.5, ls=":",
                   label=f"asymptote at t=0: (1-gamma)*(-tau_env) = "
                         f"{-tau_env_mean*(1-gamma_fit):+.1f}")
        ax.set_ylabel(f"tau_lost {axis} [kN" + ("m]" if axis == "yaw" else "]"))
        ax.set_title(f"{axis} thrust deficit (Order_pre - T_actual) at pwq30")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("t - t_WCF [s]")
    axes[-1].set_xlim(-2.0, T_PLOT_END)

    plt.suptitle(f"WcfdiScenario calibration vs brucon delivered thrust at {TAG}",
                 fontsize=12)
    plt.tight_layout()
    out = THIS / "calibrate_wcfdi_scenario_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # Summary table
    print(f"\n=== Summary (axis : gamma_fit, T_fit) ===")
    for axis, f in fits.items():
        print(f"  {axis:5s} : gamma_imm = {f['gamma']:+.3f}   T_realloc = {f['T']:6.2f} s   "
              f"RMS_fit = {f['rms_fit']:.4f}  vs  RMS_default = {f['rms_default']:.4f}")
    # Recommendation: a single global pair? Take the per-axis means or
    # the surge/sway average (yaw is often noisier and less critical).
    g_xy = float(np.mean([fits["surge"]["gamma"], fits["sway"]["gamma"]]))
    T_xy = float(np.mean([fits["surge"]["T"], fits["sway"]["T"]]))
    print(f"\n  surge+sway average : gamma_imm = {g_xy:.3f}   T_realloc = {T_xy:.2f} s")
    print(f"  (compare cqa default 0.500 / 10.00 s)")


if __name__ == "__main__":
    main()
