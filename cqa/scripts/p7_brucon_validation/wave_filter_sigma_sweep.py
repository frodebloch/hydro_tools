"""Quantify how much closed-loop sigma the 2nd-order wave filter (Tp=10.2s) explains.

Hypothesis: with Tp=10.2s the wave-filter notch (omega_w=0.616 rad/s, zeta_n=0.107)
reaches down to slow-band frequencies (omega ~ 0.03 rad/s) and degrades the LF
estimate, mimicking a first-order thrust-lag-like effect on closed-loop sway sigma.

Test: in the sandbox, sweep wave-filter omega_w over a range corresponding to
Tp = 5, 8, 10.2, 14, 20, 30 s, all with thrust_tau=0 and no integrator. Compare
the resulting closed-loop sigma to the brucon-target 0.65 m. If the actual
Tp=10.2 setting alone (without thrust_tau) gets close to 0.65, the wave-filter
hypothesis is confirmed.

Also report the wave-filter transfer eta_w(s)/e(s) magnitude/phase at the
controller bandwidth omega_n_sway = 0.08 rad/s.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox_passive_observer import build_closed_loop  # noqa: E402
from validate_sandbox_timeseries import (  # noqa: E402
    load_seed_csv, project_ned_to_body, simulate_lti,
)

T_START = 300.0
T_END = 555.0


def wave_filter_eta_over_e(omega: np.ndarray, omega_w: float, zeta_n: float) -> np.ndarray:
    """Sælid/Jensen 2nd-order wave filter: eta_w(s) / e(s) = k2*s / (s^2 + 2*zeta_n*omega_w*s + omega_w^2).

    With k2_f = 2*(1-zeta_n)*omega_w.
    """
    k2 = 2.0 * (1.0 - zeta_n) * omega_w
    s = 1j * omega
    num = k2 * s
    den = s * s + 2.0 * zeta_n * omega_w * s + omega_w ** 2
    return num / den


def main() -> None:
    work = Path(__file__).parent / "work"
    seeds = sorted(work.glob("pwo_seed*"))
    print(f"Found {len(seeds)} seeds")

    # Tp values and corresponding wave-filter parameters
    Tp_values = [5.0, 8.0, 10.2, 14.0, 20.0, 30.0]
    cases = []
    for Tp in Tp_values:
        omega_w = 2 * np.pi / Tp
        # Brucon's gain-scheduling: ScaleGainLinear(omega_w, 2pi/18, 2pi/10, 0.25, 0.1)
        # zeta_n linearly between 0.25 (at Tp=18s) and 0.10 (at Tp=10s), clipped
        omega_lo = 2 * np.pi / 18.0
        omega_hi = 2 * np.pi / 10.0
        if omega_w <= omega_lo:
            zeta_n = 0.25
        elif omega_w >= omega_hi:
            zeta_n = 0.10
        else:
            t = (omega_w - omega_lo) / (omega_hi - omega_lo)
            zeta_n = 0.25 + t * (0.10 - 0.25)
        label = f"Tp={Tp:.1f}s_zeta={zeta_n:.2f}"
        cases.append((label, dict(use_observer=True, use_bias_ff=True, use_wave_filter=True,
                                  use_integrator=False, thrust_tau=0.0,
                                  omega_w=omega_w, zeta_n=zeta_n)))
    # Also a no-WF reference and the original tau=5 reference
    cases.append(("no_WF",          dict(use_observer=True, use_bias_ff=True, use_wave_filter=False,
                                         use_integrator=False, thrust_tau=0.0)))
    cases.append(("Tp10.2_tau5",    dict(use_observer=True, use_bias_ff=True, use_wave_filter=True,
                                         use_integrator=False, thrust_tau=5.0,
                                         omega_w=2*np.pi/10.2, zeta_n=0.107)))

    # ---- Frequency-domain: show eta_w/e magnitude & phase for each Tp
    omega = np.logspace(-3, 1, 400)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    omega_n_sway = 0.08
    print(f"\nWave-filter transfer eta_w(j*omega_n_sway={omega_n_sway})/e at controller bandwidth:")
    print(f"{'Tp [s]':>8} {'omega_w':>10} {'zeta_n':>8} {'|H|':>10} {'phase[deg]':>12}")
    for Tp in Tp_values:
        omega_w = 2 * np.pi / Tp
        omega_lo = 2 * np.pi / 18.0
        omega_hi = 2 * np.pi / 10.0
        if omega_w <= omega_lo:
            zeta_n = 0.25
        elif omega_w >= omega_hi:
            zeta_n = 0.10
        else:
            t = (omega_w - omega_lo) / (omega_hi - omega_lo)
            zeta_n = 0.25 + t * (0.10 - 0.25)
        H = wave_filter_eta_over_e(omega, omega_w, zeta_n)
        axes[0].loglog(omega, np.abs(H), label=f"Tp={Tp:.1f}s, ζn={zeta_n:.2f}")
        axes[1].semilogx(omega, np.degrees(np.angle(H)), label=f"Tp={Tp:.1f}s")
        # Value at omega_n
        H_b = wave_filter_eta_over_e(np.array([omega_n_sway]), omega_w, zeta_n)[0]
        print(f"{Tp:>8.1f} {omega_w:>10.4f} {zeta_n:>8.3f} {np.abs(H_b):>10.4f} {np.degrees(np.angle(H_b)):>12.1f}")
    axes[0].axvline(omega_n_sway, color="k", ls=":", alpha=0.4, label="ω_n_sway")
    axes[0].set_ylabel("|η_w/e|"); axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, loc="upper right"); axes[0].set_ylim(1e-3, 5)
    axes[1].set_ylabel("phase [deg]"); axes[1].set_xlabel("omega [rad/s]")
    axes[1].grid(alpha=0.3, which="both"); axes[1].axvline(omega_n_sway, color="k", ls=":", alpha=0.4)
    axes[1].legend(fontsize=7)
    out = Path(__file__).parent / "wave_filter_transfer.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")

    # ---- Time-domain: per-seed sigma for each case (subset of seeds for speed)
    print(f"\nRunning time-domain sandbox over {len(seeds)} seeds, {len(cases)} cases each...")
    sigmas = {label: [] for label, _ in cases}
    sigmas["brucon_LF"] = []
    for sd in seeds:
        try:
            cols = load_seed_csv(sd)
        except Exception as e:
            print(f"  skip {sd.name}: {e}")
            continue
        t = cols["t"]; m = (t >= T_START) & (t <= T_END)
        t_w = t[m] - t[m][0]
        sway_total = project_ned_to_body(cols["x"], cols["y"], cols["heading"])[1]
        sway_lf = sway_total - cols["yHf"]
        sway_lf_w = sway_lf[m] - sway_lf[m].mean()
        F_drift = cols["DriftY"][m] * 1000.0; F_drift = F_drift - F_drift.mean()
        y_wf = cols["yHf"][m] - cols["yHf"][m].mean()
        sigmas["brucon_LF"].append(sway_lf_w.std())
        for label, kw in cases:
            A, Bd, Bw, _, _ = build_closed_loop(**kw)
            x = simulate_lti(A, Bd, Bw, t_w, F_drift, y_wf)
            sigmas[label].append(x[0, :].std())

    print(f"\nResults across {len(sigmas['brucon_LF'])} seeds:")
    print(f"{'case':<32} {'median σ':>10} {'mean σ':>10}  {'% of brucon':>12}")
    print("-" * 72)
    target = np.median(sigmas["brucon_LF"])
    for label in ["brucon_LF"] + [c[0] for c in cases]:
        arr = np.array(sigmas[label])
        med = np.median(arr); mn = arr.mean()
        pct = 100 * med / target if target > 0 else float("nan")
        print(f"{label:<32} {med:>10.3f} {mn:>10.3f}  {pct:>10.0f}%")

    # ---- Plot sigma vs Tp
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    Tp_arr = np.array(Tp_values)
    sig_arr = np.array([np.median(sigmas[c[0]]) for c in cases[:len(Tp_values)]])
    ax.plot(Tp_arr, sig_arr, "o-", label="sandbox: full obs, no thrust lag")
    ax.axhline(target, color="r", ls="--", label=f"brucon target σ_LF = {target:.2f} m")
    ax.axhline(np.median(sigmas["no_WF"]), color="g", ls=":", label=f"no wave filter = {np.median(sigmas['no_WF']):.2f} m")
    ax.axhline(np.median(sigmas["Tp10.2_tau5"]), color="C2", ls=":",
               label=f"Tp=10.2 + tau=5s = {np.median(sigmas['Tp10.2_tau5']):.2f} m")
    ax.axvline(10.2, color="k", ls=":", alpha=0.5, label="actual Tp=10.2s")
    ax.set_xlabel("wave-filter Tp [s]"); ax.set_ylabel("median σ_y_LF [m]")
    ax.set_title("Closed-loop σ_LF vs wave-filter design Tp (P7 waves-only)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out2 = Path(__file__).parent / "sigma_vs_wavefilter_Tp.png"
    plt.tight_layout(); plt.savefig(out2, dpi=120)
    print(f"\nSaved: {out2}")


if __name__ == "__main__":
    main()
