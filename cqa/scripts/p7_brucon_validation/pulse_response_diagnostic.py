"""Closed-loop pulse-response diagnostic.

Question: when we drive the linearised cqa augmented system (Fix 1+2+3,
include_integrator=True) with the exact tau_lost(t) pulse used by the
cqa MC, does the analytical η(t) match (a) the cqa MC ensemble mean
[sanity check] and (b) the brucon truth ensemble mean [the real gap]?

The step response is a controller-design tool but not informative for the
post-WCF transient: the actual disturbance is a short-duration thrust
deficit pulse, not a sustained force step. We use the pulse response
shape that matches what cqa actually injects.

Three competing hypotheses for the post-WCF ensemble-mean recovery gap:

  H1. The cqa MC and the linearised pulse response disagree. Then there
      is a bug or hidden nonlinearity in the MC path itself.

  H2. cqa MC and analytic pulse agree, but both disagree with brucon
      truth. The shared disturbance shape (linear-decay pulse with
      per-seed peak / T_eff) is not what the brucon loop actually sees.
      Likely: the disturbance lasts longer than 9 s, or has a different
      profile (e.g. azimuth slew + spool-up creates a more complex shape).

  H3. cqa MC and analytic pulse agree on shape and peak, but brucon truth
      shows different LOOP DYNAMICS (slower or faster recovery). Then the
      linearised model has the wrong gain / damping / lag for some loop
      element.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from cqa.config import csov_default_config
from cqa.transient import build_augmented_system
from cqa.controller import LinearDpController
from cqa.vessel import LinearVesselModel


def main():
    # --- 1. Build the linearised closed-loop A matrix (Fix 1+2+3) ---
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    cp = cfg.controller
    ctrl = LinearDpController.from_bandwidth(
        M=vessel.M, D=vessel.D, omega_n=cp.omega_n, zeta=cp.zeta,
    )
    aug = build_augmented_system(
        vessel=vessel,
        controller=ctrl,
        T_b=cp.bias_time_constant_s,
        T_thr=cp.thruster_time_constant_s,
        include_integrator=True,
    )
    print(f"A_cl shape = {aug.A.shape}")
    print(f"Eigenvalues of A_cl:")
    eig = np.linalg.eigvals(aug.A)
    for e in sorted(eig, key=lambda z: z.real):
        print(f"  {e.real:+.4e} {e.imag:+.4e}j")
    n_zero = int(np.sum(np.abs(eig) < 1e-10))
    print(f"  -> {n_zero} eigenvalues at 0 (expected: 3, from frozen b_hat block)")

    # --- 2. Pulse response. The cqa post-WCF disturbance model is a
    # tau_lost(t) pulse: linear-decay over T_pulse seconds from the
    # peak amplitude F_pulse_peak to zero. This is what the cqa MC
    # actually injects; comparing the pulse response to (a) the cqa MC
    # ensemble mean confirms the MC is implementing the linearised
    # dynamics correctly; (b) the brucon truth ensemble mean tells us
    # whether the gap is in the loop dynamics or in the disturbance shape.
    #
    # Per-seed median from the brucon ensemble (sec.12.21.7):
    #   sway peak deficit ~ -200 kN (NEGATIVE: lost thrusters were producing
    #     positive sway force, so their loss is a negative force deficit
    #     pushing the vessel in -y), T_eff ~ 8-9 s, linear decay.
    #   yaw  peak deficit ~ -1500 kNm, T_eff ~ 12 s.
    # Use sway-only F_pulse_peak = -200 kN, T_pulse = 9 s.
    F_pulse_peak = np.array([0.0, -200_000.0, 0.0])
    T_pulse = 9.0

    def tau_lost(ti):
        if ti < 0 or ti >= T_pulse:
            return np.zeros(3)
        return F_pulse_peak * (1.0 - ti / T_pulse)

    n = aug.n_state
    t = np.linspace(0, 180, 1801)  # dt = 0.1 s

    # Time-step: x_{k+1} = expm(A dt) x_k + (integral over [t_k, t_k+dt])
    # For dt = 0.1 s and a slowly-varying F over that interval, use
    # trapezoidal rule:
    #   x_{k+1} = Phi_dt @ x_k + 0.5 dt (Phi_dt @ B_d F(t_k) + B_d F(t_k+dt))
    dt = float(t[1] - t[0])
    Phi_dt = expm(aug.A * dt)
    B_d = aug.B_w  # B_d == B_w (both = M^-1 on nu rows)
    x = np.zeros(n)
    eta_pulse = np.zeros((len(t), 3))
    eta_pulse[0] = x[0:3]
    for k in range(len(t) - 1):
        f_k = B_d @ tau_lost(t[k])
        f_kp = B_d @ tau_lost(t[k + 1])
        x = Phi_dt @ x + 0.5 * dt * (Phi_dt @ f_k + f_kp)
        eta_pulse[k + 1] = x[0:3]

    # --- 4. Load brucon ensemble + cqa MC ensemble means for overlay ---
    npz_path = "/tmp/status_ki_on.npz"
    try:
        data = np.load(npz_path)
        t_truth = data["t_truth"]
        truth_dsway = data["truth_dsway_mean"]
        truth_dsurge = data["truth_dsurge_mean"]
        t_cqa = data["t_cqa"]
        cal_dsway = data["cal_dsway_mean"]
        cal_dsurge = data["cal_dsurge_mean"]
        have_overlay = True
    except FileNotFoundError:
        print(f"WARNING: {npz_path} not found; plotting analytical only")
        have_overlay = False

    # --- 4. Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_x, ax_y = axes
    ax_x.plot(t, eta_pulse[:, 0], "k-", lw=2,
              label=f"cqa analytic PULSE (peak {F_pulse_peak[0]/1e3:.0f} kN, T={T_pulse:.0f}s lin-decay)")
    ax_y.plot(t, eta_pulse[:, 1], "k-", lw=2,
              label=f"cqa analytic PULSE (peak {F_pulse_peak[1]/1e3:.0f} kN, T={T_pulse:.0f}s lin-decay)")
    if have_overlay:
        ax_x.plot(t_truth, truth_dsurge, "C0-", lw=1.2, alpha=0.8,
                  label="brucon truth ensemble-mean Δsurge")
        ax_x.plot(t_cqa, cal_dsurge, "C3:", lw=1.5,
                  label="cqa MC ensemble-mean Δsurge (Fix 1+2+3)")
        ax_y.plot(t_truth, truth_dsway, "C0-", lw=1.2, alpha=0.8,
                  label="brucon truth ensemble-mean Δsway")
        ax_y.plot(t_cqa, cal_dsway, "C3:", lw=1.5,
                  label="cqa MC ensemble-mean Δsway (Fix 1+2+3)")
    ax_x.set_ylabel("Δsurge [m]")
    ax_y.set_ylabel("Δsway [m]")
    ax_y.set_xlabel("t since WCF [s]")
    ax_x.legend(loc="best", fontsize=7)
    ax_y.legend(loc="best", fontsize=7)
    ax_x.grid(True, alpha=0.3)
    ax_y.grid(True, alpha=0.3)
    ax_x.set_title("Pulse response: cqa analytic vs cqa MC vs brucon truth ensemble means")

    out_path = "scripts/p7_brucon_validation/pulse_response_diagnostic.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")

    # --- 5. Print summary metrics ---
    print()
    print("--- pulse response summary ---")
    peak_yp = float(np.max(np.abs(eta_pulse[:, 1])))
    t_peak_yp = float(t[np.argmax(np.abs(eta_pulse[:, 1]))])
    print(f"ANALYTIC PULSE sway peak: {peak_yp:.3f} m at t={t_peak_yp:.1f} s")
    if have_overlay:
        cal_peak_y = float(np.max(np.abs(cal_dsway)))
        t_cal_peak = float(t_cqa[np.argmax(np.abs(cal_dsway))])
        truth_peak_y = float(np.max(np.abs(truth_dsway)))
        t_truth_peak = float(t_truth[np.argmax(np.abs(truth_dsway))])
        print(f"CQA MC      sway-mean peak: {cal_peak_y:.3f} m at t={t_cal_peak:.1f} s")
        print(f"BRUCON truth sway-mean peak: {truth_peak_y:.3f} m at t={t_truth_peak:.1f} s")
        print()
        print(f"H1 check (analytic == cqa MC?): "
              f"{abs(peak_yp - cal_peak_y) / max(cal_peak_y, 1e-6):.1%} peak diff")
        print(f"  -> if small, the cqa MC faithfully implements the linearised "
              f"pulse response.")
        print(f"H2/H3 (analytic vs brucon truth): "
              f"{abs(peak_yp - truth_peak_y) / max(truth_peak_y, 1e-6):.1%} peak diff, "
              f"{t_truth_peak - t_peak_yp:+.1f} s peak-time shift")
        print(f"  -> if large, gap is in tau_lost shape (H2) or loop dynamics (H3).")


if __name__ == "__main__":
    main()
