"""Cross-checks for the 21-state observer-augmented system.

Run with:
    PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/check_transient_obs.py

Tests
-----
T1. Zero forcing, zero IC -> state stays at zero (sanity).
T2. Eigenvalues of A: count of zero modes, slowest decay rate, fastest.
T3. Observer convergence in isolation: hold truth state constant, verify
    eta_hat -> eta and nu_hat -> nu (steady) on a relevant time scale.
T4. Intact steady-state under tau_env: A x_ss + B_d tau_env should be
    zero in the closed-form prediction.
T5. Pulse response from a brucon-like τ_lost(0+) = -200 kN sway, 9-s
    linear decay. Compare peak magnitude / timing against the
    brucon truth ensemble (Δsway peak -0.50 m at t = +35 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config
from cqa.vessel import LinearVesselModel
from cqa.controller import LinearDpController
from cqa.transient_obs import (
    build_observer_augmented_system_full,
    intact_mean_steady_state_obs,
    pulse_response,
    csov_observer_gains,
    IDX_ETA, IDX_NU, IDX_ETA_HAT, IDX_NU_HAT, IDX_B_HAT, IDX_TAU_THR, IDX_INT,
    IDX_XI, IDX_ETA_W,
    N_STATE,
)


def build():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    # Brucon CSOV regulator: omega_n = (0.060, 0.080, 0.120), zeta=0.95.
    omega_n = np.array([0.060, 0.080, 0.120])
    zeta = np.array([0.95, 0.95, 0.95])
    ctrl = LinearDpController.from_bandwidth(vessel.M, vessel.D, omega_n=omega_n, zeta=zeta)
    aug = build_observer_augmented_system_full(vessel, ctrl, T_thr=5.0)
    return cfg, vessel, ctrl, aug


def main():
    cfg, vessel, ctrl, aug = build()
    print(f"vessel.M diag = {np.diag(aug.M)}")
    print(f"vessel.D diag = {np.diag(aug.D)}")
    print(f"Kp diag       = {np.diag(aug.Kp)}")
    print(f"Kd diag       = {np.diag(aug.Kd)}")
    print(f"Ki diag       = {np.diag(aug.Ki)}")
    print(f"obs.K_b_pos   = {aug.obs_gains.K_b_pos}")
    print(f"obs.K_a_pos   = {aug.obs_gains.K_a_pos}")
    print(f"obs.omega_c   = {aug.obs_gains.omega_c}")
    print(f"obs.T_b       = {aug.obs_gains.T_b}")
    print(f"T_thr         = {aug.T_thr}")

    # ------------ T1: zero forcing, zero IC ------------
    t = np.linspace(0, 60, 601)
    tau_lost = np.zeros((len(t), 3))
    X = pulse_response(aug, t, tau_lost, x0=np.zeros(N_STATE))
    err = np.max(np.abs(X))
    print(f"\n[T1] zero forcing, zero IC -> max|x| = {err:.3e}  (expect 0)")
    assert err < 1e-12, "T1 failed: state drifted from zero with no forcing"

    # ------------ T2: eigenvalues ------------
    w, _ = np.linalg.eig(aug.A)
    re = np.real(w)
    im = np.imag(w)
    n_zero = int(np.sum(np.abs(w) < 1e-10))
    n_pos = int(np.sum(re > 1e-10))
    print(f"\n[T2] eigenvalues:")
    print(f"  count           : {len(w)}")
    print(f"  near-zero modes : {n_zero}")
    print(f"  positive-real   : {n_pos}    (expect 0 for Hurwitz)")
    print(f"  Re range        : [{re.min():.4f}, {re.max():.4f}] rad/s")
    # Slowest stable mode (excluding numerical zeros)
    re_neg = re[re < -1e-10]
    if len(re_neg) > 0:
        slowest = re_neg.max()  # closest to zero
        print(f"  slowest stable  : Re = {slowest:.4f} rad/s   (tau = {-1/slowest:.1f} s)")
        fastest = re_neg.min()
        print(f"  fastest stable  : Re = {fastest:.4f} rad/s   (tau = {-1/fastest:.2f} s)")
    print(f"  imag range      : [{im.min():.4f}, {im.max():.4f}] rad/s")
    if n_pos > 0:
        print("  WARNING: positive-real eigenvalues -> system is NOT Hurwitz!")

    # ------------ T3: observer convergence in isolation ------------
    # Hold truth eta = (0, 0.5, 0), nu = 0; freeze truth dynamics by zeroing
    # the truth rows of A; let the observer relax.
    A_obs_only = aug.A.copy()
    A_obs_only[IDX_ETA, :] = 0
    A_obs_only[IDX_NU, :] = 0
    A_obs_only[IDX_TAU_THR, :] = 0   # break the controller -> thruster path so it doesn't react
    A_obs_only[IDX_INT, :] = 0
    x = np.zeros(N_STATE)
    x[IDX_ETA] = np.array([0.0, 0.5, 0.0])
    # Time-step
    from scipy.linalg import expm
    dt = 0.1
    Phi = expm(A_obs_only * dt)
    T_test = 200.0
    nT = int(T_test / dt) + 1
    X_test = np.zeros((nT, N_STATE))
    X_test[0] = x
    for k in range(1, nT):
        X_test[k] = Phi @ X_test[k - 1]
    eta_hat_final = X_test[-1, IDX_ETA_HAT]
    eta_truth     = X_test[-1, IDX_ETA]
    err_obs = eta_hat_final - eta_truth
    print(f"\n[T3] observer convergence (truth η held constant):")
    print(f"  truth eta      = {eta_truth}")
    print(f"  eta_hat(T=200s)= {eta_hat_final}")
    print(f"  error          = {err_obs}    (expect ~0 in sway after 5/omega_c ≈ 5 s, but b_hat takes 3*T_b)")
    # observer ν̂ should also have settled to truth ν=0.
    nu_hat_final = X_test[-1, IDX_NU_HAT]
    print(f"  nu_hat(T=200s) = {nu_hat_final}    (expect ~0)")

    # ------------ T4: intact steady state under tau_env ------------
    tau_env = np.array([10e3, -50e3, 200e3])  # arbitrary plausible env force
    # Compare the analytic SS guess (closed-form) against the exact SS
    # obtained by solving A x_ss = -B_d tau_env. The closed-form sets
    # b_hat_ss = 0 and Ki I_ss = tau_env, ignoring the observer's
    # nu_hat_dot residual (see docstring of intact_mean_steady_state_obs).
    # The exact SS has a small η̂ offset that closes the loop.
    x_ss_analytic = intact_mean_steady_state_obs(aug, tau_env)
    rhs_analytic = aug.A @ x_ss_analytic + aug.B_d @ tau_env
    x_ss_exact = np.linalg.solve(aug.A, -aug.B_d @ tau_env)
    rhs_exact = aug.A @ x_ss_exact + aug.B_d @ tau_env
    print(f"\n[T4] intact SS check, tau_env = {tau_env}:")
    print(f"  analytic guess:")
    print(f"    x_ss[I]       = {x_ss_analytic[IDX_INT]}")
    print(f"    x_ss[b_hat]   = {x_ss_analytic[IDX_B_HAT]}")
    print(f"    x_ss[tau_thr] = {x_ss_analytic[IDX_TAU_THR]}")
    print(f"    ||residual||  = {np.linalg.norm(rhs_analytic):.3e}")
    print(f"  exact (solve):")
    print(f"    x_ss[eta]     = {x_ss_exact[IDX_ETA]}")
    print(f"    x_ss[eta_hat] = {x_ss_exact[IDX_ETA_HAT]}")
    print(f"    x_ss[b_hat]   = {x_ss_exact[IDX_B_HAT]}")
    print(f"    x_ss[I]       = {x_ss_exact[IDX_INT]}")
    print(f"    x_ss[tau_thr] = {x_ss_exact[IDX_TAU_THR]}")
    print(f"    x_ss[xi]      = {x_ss_exact[IDX_XI]}")
    print(f"    x_ss[eta_w]   = {x_ss_exact[IDX_ETA_W]}")
    print(f"    ||residual||  = {np.linalg.norm(rhs_exact):.3e}   (expect ~0)")

    # ------------ T5: brucon-like pulse response ------------
    # τ_lost(t) on sway: linear decay from peak at t=0 to zero at t = T_pulse.
    # Use peak = -200 kN, T_pulse = 9 s, applied to sway only.
    # Background: post-WCF reallocation pulls one bus_port group offline,
    # leading to an instantaneous sway thrust deficit that decays as the
    # remaining thrusters re-allocate (see brucon_thrust_deficit.py).
    PEAK_KN = -200.0  # peak of (Order - Delivered) on sway, [kN]
    T_PULSE = 9.0
    T_TOT = 120.0
    t = np.arange(0, T_TOT + 0.05, 0.1)
    tau_lost = np.zeros((len(t), 3))
    pulse = np.where(t <= T_PULSE, PEAK_KN * 1e3 * (1 - t / T_PULSE), 0.0)
    tau_lost[:, 1] = pulse  # sway only
    X = pulse_response(aug, t, tau_lost, x0=np.zeros(N_STATE))
    sway = X[:, 1]  # truth sway
    sway_hat = X[:, 7]  # eta_hat sway
    bhat_sway = X[:, 13]  # b_hat sway
    pulse_peak_idx = int(np.argmin(sway))
    pulse_peak_t = t[pulse_peak_idx]
    pulse_peak = sway[pulse_peak_idx]
    sway_hat_peak = sway_hat[int(np.argmin(sway_hat))]
    print(f"\n[T5] pulse response (sway, peak -200 kN, 9-s decay):")
    print(f"  truth Δsway peak  : {pulse_peak:+.3f} m at t = {pulse_peak_t:.1f} s")
    print(f"  η̂   Δsway peak  : {sway_hat_peak:+.3f} m")
    print(f"  truth - η̂ peak  : {(sway - sway_hat).min():+.3f} m at t = {t[(sway - sway_hat).argmin()]:.1f} s")
    print(f"  b̂_sway @ end     : {bhat_sway[-1]:+.1f} N    (3*T_b = {3*aug.obs_gains.T_b[1]:.0f} s; will be small at t=120)")
    print(f"\n  brucon truth Δsway peak ensemble mean: -0.504 m at t = +34.0 s (use_tau_feedback=true)")
    print(f"                                           -0.558 m at t = +34.0 s (default)")


if __name__ == "__main__":
    main()
