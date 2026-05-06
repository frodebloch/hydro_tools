"""Sandbox: linearised closed loop with brucon nonlinear passive observer.

Goal
----
Resolve the σ_y_LF gap (cqa 0.17 m perfect-FB vs brucon 0.65 m,
P7 waves-only sway, 30-seed median, intact window t ∈ [300, 555] s).

cqa's 6-state PD-with-bias-FF model assumes:
  - perfect bias rejection (b̂ = b instantly)
  - perfect LF position estimate (ŷ_LF = y_LF instantly)
  - no observer dynamics, no wave filter, no integrator
  - infinite thrust bandwidth

Brucon's actual stack (per DOF):
  - Fossen nonlinear passive observer with K_a1 = 0.12, K_b1 = 0.0012,
    T_b = 1000 s, ω_c = 1.04 rad/s
  - Sælid/Jensen 2nd-order wave filter on position innovation
    (ω_w = 2π/Tp, ζ_n from gain schedule, k1_f / k2_f computed)
  - Innovation = y_meas − ŷ_LF − η̂_w (wave-corrected innovation, key!)
  - PID with brucon-realistic Kp = M·ω_n², Kd = 2·ζ·M·ω_n,
    Ki = 0.1·ω_n·Kp, plus controller-side velocity feedforward
    that cancels the vessel's natural damping in software

Modelling decisions documented in analysis.md §12.20:
  - The optional `thrust_tau` parameter is a *phenomenological* 1st-order
    pole on the controller-output → vessel-force path. Empirically
    estimating brucon's actual `OrderTau → Ty` transfer (see
    estimate_thrust_lag.py) finds it flat through the entire slow band
    (|H|² ≈ 1.0, phase ≈ 0°) -- the simulator's RPM rate (10 %/s) and
    azimuth rate (12°/s) limits are essentially inactive in P7
    station-keeping. The τ ≈ 5 s setting that closes the σ-gap to 6 %
    therefore does NOT correspond to a measurable physical actuator
    lag; it is a calibration knob whose physical mechanism is
    unidentified.

This script builds a per-DOF (sway) state-space model:

  States x = [y, ν, ŷ_LF, ν̂, b̂, ξ_w, η̂_w]   (7 states)
                + integrator (1 state, optional)
                + thrust-lag actuator (1 state, optional)

  Vessel:
    y_dot = ν
    ν_dot = (1/M) · (-D·ν + b + u_act + F_drift)     (b = true environmental bias)
    (we treat b as constant slow process with F_drift = white-driven slow drift)

  Innovation:
    e = y_meas − ŷ_LF − η̂_w,   with y_meas = y + y_WF
    (here y_WF is the true wave-frequency motion superimposed on the LF;
     for the LF-only test we set y_WF = 0 and just inject drift force)

  Observer (per DOF, sway):
    ŷ_LF_dot = ν̂ + ω_c · e                          ω_c = 1.04 rad/s
    ν̂_dot   = (1/M)·(b̂ + u + D·ν̂_obs_term) + K_a1·e
              (we use observer's own model; assume D_obs = D for simplicity)
    b̂_dot   = -(1/T_b)·b̂ + K_b1·e                   T_b = 1000 s
    ξ_w_dot = η̂_w + k1_f · e
    η̂_w_dot = -ω_w² · ξ_w − 2ζ_n·ω_w · η̂_w + k2_f · e

  Controller (PID with bias FF):
    u = -Kp · ŷ_LF − Kd · ν̂ − b̂

Brucon controller gains for sway (from tuning_parameters_no_speed_dependency
+ tuning.prototxt: ω_n_sway = 0.08 rad/s, ζ = 0.95):
  Kp = M · ω_n²        Kd = 2·ζ·M·ω_n        Ki = 0.1·ω_n·Kp
  (we enable Ki separately as a 8th state ∫ŷ_LF dt)

Driving force: brucon's pdstrip Newman slow-drift PSD at Hs=4.2, Tp=10.2,
β=90° (head-to-wave, sway-dominant). We approximate it as a flat white-noise
spectrum of strength matching the variance band 0.001–0.1 Hz.

Usage
-----
    .venv/bin/python scripts/p7_brucon_validation/sandbox_passive_observer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output, SIM_DT  # noqa: E402

from cqa.config import csov_default_config  # noqa: E402
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw  # noqa: E402
from cqa.drift import slow_drift_force_psd_newman_pdstrip  # noqa: E402
from cqa.rao import load_pdstrip_rao  # noqa: E402

PDSTRIP_PATH = (
    "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
)

# ---------------------------------------------------------------------------
# Configuration matching the P7 waves-only validation case
# ---------------------------------------------------------------------------
VW_FOR_PM = 14.0
HS = pm_hs_from_vw(VW_FOR_PM)
TP = pm_tp_from_vw(VW_FOR_PM)
WAVE_DIR_NED = 270.0   # waves coming from west
HEADING_NED = 180.0    # vessel pointing south  -> waves on starboard beam (β = 90°)
BETA_REL = 90.0        # for sway dominant

# Brucon CSOV parameters (sway DOF)
cfg = csov_default_config()
M_SWAY = cfg.vessel.displacement_mass * (1.0 + cfg.vessel.sway_added_mass_frac)
print(f"M_sway = {M_SWAY:.3e} kg")

# Brucon damping (effective, per session findings: ~2e4 N/(m/s))
# but cqa controller uses linear_damping_sway = M/40 ~ 4.65e5
# Per session: total damping in closed loop is dominated by Kd, and
# Kd_brucon = 2·ζ·M·ω - 0 (brucon does NOT subtract D from Kd!)
# Let's use D_open = small (true effective brucon damping)
D_SWAY = 2.0e4   # ~CurY_std / SwaySpeed_std from P7 brucon ensemble

# Brucon controller gains (sway): omega_n = 0.08, zeta = 0.95
OMEGA_N = 0.08
ZETA = 0.95
KP = M_SWAY * OMEGA_N ** 2
KD = 2.0 * ZETA * M_SWAY * OMEGA_N           # brucon does NOT subtract D
KI = 0.1 * OMEGA_N * KP
print(f"Kp = {KP:.3e}, Kd = {KD:.3e}, Ki = {KI:.3e}")

# Brucon observer gains (config_csov/observer.prototxt sway block)
KA1 = 0.12      # acceleration gain on position innovation
KB1 = 0.0012    # bias gain on position innovation
T_B = 1000.0    # bias time constant [s]
OMEGA_C = 1.04  # position wave filter cutoff frequency [rad/s]

# Wave filter for sway (peak at Tp = 10.2 s)
OMEGA_W = 2.0 * np.pi / TP
# brucon ScaleGainLinear: ζ_n = ScaleGainLinear(ω_w, 2π/18, 2π/10, 0.25, 0.1)
ZETA_N_LO, ZETA_N_HI = 0.25, 0.10
OMEGA_W_LO, OMEGA_W_HI = 2.0 * np.pi / 18.0, 2.0 * np.pi / 10.0
if OMEGA_W <= OMEGA_W_LO:
    ZETA_N = ZETA_N_LO
elif OMEGA_W >= OMEGA_W_HI:
    ZETA_N = ZETA_N_HI
else:
    f = (OMEGA_W - OMEGA_W_LO) / (OMEGA_W_HI - OMEGA_W_LO)
    ZETA_N = ZETA_N_LO + f * (ZETA_N_HI - ZETA_N_LO)
K1F = -2.0 * (1.0 - ZETA_N) * OMEGA_C / OMEGA_W
K2F = 2.0 * OMEGA_W * (1.0 - ZETA_N)
print(f"Wave filter: ω_w = {OMEGA_W:.3f} rad/s, ζ_n = {ZETA_N:.3f}, "
      f"k1_f = {K1F:.3f}, k2_f = {K2F:.3f}, ω_c = {OMEGA_C:.3f}")


# ---------------------------------------------------------------------------
# Build A_cl, B_w (drift) and B_wf (true wave-frequency motion injected as y_meas)
# ---------------------------------------------------------------------------
# State: x = [y, ν, ŷ_LF, ν̂, b̂, ξ_w, η̂_w]
# Inputs: w_drift (force on vessel), y_wf (true HF motion mixed into y_meas)

def build_closed_loop(use_integrator: bool = False, use_bias_ff: bool = True,
                      use_wave_filter: bool = True, use_observer: bool = True,
                      thrust_tau: float = 0.0,
                      omega_w: float | None = None, zeta_n: float | None = None,
                      k_b1: float | None = None, k_a1: float | None = None,
                      t_b: float | None = None):
    """Return (A_cl, B_w_drift, B_wf, C_yLF, C_y) for the closed loop.

    use_integrator: enable PI integrator state
    use_bias_ff:    feed b̂ forward in u
    use_wave_filter: enable 2nd-order wave filter (else η̂_w ≡ 0, ξ_w ≡ 0)
    use_observer:   if False, controller uses true (y, ν) (perfect feedback)
    thrust_tau:     if > 0, add a 1st-order thrust lag state with
                    τ·u_dot = u_cmd − u (approximates rate limiting)
    omega_w, zeta_n: optionally override the wave-filter peak/damping (defaults
                    to module-level OMEGA_W / ZETA_N from Tp=TP).
    """
    # Wave filter parameters (with optional per-call overrides)
    if omega_w is None:
        omega_w = OMEGA_W
    if zeta_n is None:
        zeta_n = ZETA_N
    # Observer-gain overrides (default = brucon nominal CSOV sway values)
    ka1 = KA1 if k_a1 is None else k_a1
    kb1 = KB1 if k_b1 is None else k_b1
    tb = T_B if t_b is None else t_b
    k1f = -2.0 * (1.0 - zeta_n) * OMEGA_C / omega_w
    k2f = 2.0 * omega_w * (1.0 - zeta_n)
    # Variables for clarity
    extra = (1 if use_integrator else 0) + (1 if thrust_tau > 0 else 0)
    n = 7 + extra

    A = np.zeros((n, n))
    B_w = np.zeros(n)         # drift force input
    B_wf = np.zeros(n)        # true HF motion injected in y_meas

    # State indices
    iy, iv, ily, ivh, ib, ix, ie = 0, 1, 2, 3, 4, 5, 6
    next_idx = 7
    iI = None
    iU = None
    if use_integrator:
        iI = next_idx
        next_idx += 1
    if thrust_tau > 0:
        iU = next_idx
        next_idx += 1

    # Helper to write the controller command u_cmd into row `row` with given scale
    if use_observer:
        def _write_u_cmd(row, scale):
            A[row, ily] += -KP * scale
            A[row, ivh] += -KD * scale
            if use_bias_ff:
                A[row, ib] += -1.0 * scale
            if use_integrator:
                A[row, iI] += -KI * scale
    else:
        def _write_u_cmd(row, scale):
            A[row, iy] += -KP * scale
            A[row, iv] += -KD * scale
            if use_integrator:
                A[row, iI] += -KI * scale

    # If we have a thrust-lag state, the actual force on the vessel is u_th,
    # and u_th_dot = (u_cmd - u_th) / tau. Otherwise u_act == u_cmd directly.
    if thrust_tau > 0:
        # u_th_dot = (1/tau)*(u_cmd - u_th)
        _write_u_cmd(iU, 1.0 / thrust_tau)
        A[iU, iU] += -1.0 / thrust_tau
        # the force entering ν_dot is now state iU
        def add_u_to_v(row, scale):
            A[row, iU] += scale
    else:
        def add_u_to_v(row, scale):
            _write_u_cmd(row, scale)

    # Vessel dynamics:
    A[iy, iv] = 1.0
    A[iv, iv] = -D_SWAY / M_SWAY
    add_u_to_v(iv, 1.0 / M_SWAY)
    B_w[iv] = 1.0 / M_SWAY  # drift force enters here

    if use_observer:
        # Innovation: e = y_meas − ŷ_LF − η̂_w  with  y_meas = y + y_wf_true
        # We build linear coeffs on (y, ŷ_LF, η̂_w) and on input y_wf_true.

        # ŷ_LF_dot = ν̂ + ω_c · e
        A[ily, ivh] += 1.0
        A[ily, iy] += OMEGA_C
        A[ily, ily] += -OMEGA_C
        if use_wave_filter:
            A[ily, ie] += -OMEGA_C
        B_wf[ily] = OMEGA_C

        # ν̂_dot = -(D/M)·ν̂ + (1/M)·b̂ + (1/M)·u_actual_into_observer + K_a1·e
        # The observer's velocity model uses the ACTUAL force on the vessel
        # (u_th if thrust lag, else u_cmd). brucon does not know about
        # rate-limit clipping; the observer model uses commanded thrust.
        # For sandbox, use u_cmd here regardless.
        A[ivh, ivh] += -D_SWAY / M_SWAY
        A[ivh, ib] += 1.0 / M_SWAY
        # observer-vessel-model uses u_cmd (not u_th)
        _write_u_cmd(ivh, 1.0 / M_SWAY)
        A[ivh, iy] += ka1
        A[ivh, ily] += -ka1
        if use_wave_filter:
            A[ivh, ie] += -ka1
        B_wf[ivh] = ka1

        # b̂_dot = -(1/T_b)·b̂ + K_b1·e
        A[ib, ib] += -1.0 / tb
        A[ib, iy] += kb1
        A[ib, ily] += -kb1
        if use_wave_filter:
            A[ib, ie] += -kb1
        B_wf[ib] = kb1

        if use_wave_filter:
            # ξ_w_dot = η̂_w + k1_f · e
            A[ix, ie] += 1.0
            A[ix, iy] += k1f
            A[ix, ily] += -k1f
            A[ix, ie] += -k1f
            B_wf[ix] = k1f

            # η̂_w_dot = -ω_w² · ξ_w − 2ζ_n·ω_w · η̂_w + k2_f · e
            A[ie, ix] += -omega_w ** 2
            A[ie, ie] += -2.0 * zeta_n * omega_w
            A[ie, iy] += k2f
            A[ie, ily] += -k2f
            A[ie, ie] += -k2f
            B_wf[ie] = k2f

    if use_integrator:
        if use_observer:
            A[iI, ily] = 1.0
        else:
            A[iI, iy] = 1.0

    # Output mappings
    C_y = np.zeros(n);    C_y[iy] = 1.0
    C_yLF = np.zeros(n);  C_yLF[ily] = 1.0 if use_observer else 0.0

    return A, B_w, B_wf, C_y, C_yLF


def stable_eigvals(A: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, bool]:
    """Stability of *active* dynamics: ignore exact-zero eigenvalues that
    correspond to disconnected (all-zero) rows/cols (inactive observer
    states when use_observer=False). Such states do not couple to any
    output and are integrand-irrelevant."""
    eig = np.linalg.eigvals(A)
    # active stability: any negative-real eigenvalue is fine; any positive
    # real eigenvalue = unstable; tiny zero eigenvalues from inactive
    # disconnected states are tolerated.
    stab = np.all(eig.real < tol)
    # but reject if there are positive real parts > tol
    bad = np.any(eig.real > tol)
    return eig, (not bad)


# ---------------------------------------------------------------------------
# Drive: brucon Newman slow-drift PSD at the test sea state
# ---------------------------------------------------------------------------
def drift_psd_omega(omega: np.ndarray) -> np.ndarray:
    """One-sided sway slow-drift PSD [N²·s] vs ω [rad/s] at HS, TP, β=90°."""
    rao = load_pdstrip_rao(PDSTRIP_PATH)
    theta_rel = np.deg2rad(WAVE_DIR_NED - HEADING_NED)  # = +π/2 for beam-on
    S_F_callable = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao, Hs=HS, Tp=TP, theta_wave_rel=theta_rel,
    )
    S_F = np.array([S_F_callable(w) for w in omega])    # (n, 3, 3)
    S_yy = S_F[:, 1, 1]                                  # sway diagonal
    return S_yy


def state_variance_freqdomain(A: np.ndarray, B: np.ndarray, S_omega,
                              omega: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Return (σ², ω_peak) of output y = c·x driven by an input with
    one-sided rad/s-native PSD S(ω) acting through B.

    σ² = ∫₀^∞ |H(jω)|² S(ω) dω,  H(jω) = c·(jωI − A)⁻¹·B
    Convention: one-sided PSD in rad/s, matching cqa.psd /
    cqa.drift output (verified: ∫ S_eta(ω) dω = Hs²/16 with no /π).

    Earlier versions of this function carried a spurious /π factor
    that under-predicted σ by √π = 1.77. Removed 2026-05-06 after
    cross-check against ∫ S_eta = Hs²/16 and against time-domain
    Welch on a long brucon realisation (nperseg ≥ 2000 → match to
    0.69 m vs prior buggy 0.40 m).
    """
    n = A.shape[0]
    Hsq = np.zeros_like(omega)
    I = np.eye(n)
    for k, w in enumerate(omega):
        try:
            H = c @ np.linalg.solve(1j * w * I - A, B)
            Hsq[k] = np.abs(H) ** 2
        except np.linalg.LinAlgError:
            Hsq[k] = np.nan
    integrand = Hsq * S_omega
    sigma2 = np.trapezoid(integrand, omega)
    ω_peak = omega[np.nanargmax(integrand)]
    return sigma2, ω_peak


# ---------------------------------------------------------------------------
# Main: scan model variants
# ---------------------------------------------------------------------------
def main() -> None:
    omega = np.logspace(-3.5, 0.0, 2048)
    print(f"\nDrift PSD at HS={HS:.2f}, TP={TP:.2f}, β={BETA_REL:.0f}°...")
    S_drift = drift_psd_omega(omega)
    sigma_F = np.sqrt(np.trapezoid(S_drift, omega))
    print(f"  σ_F_drift_y = {sigma_F/1000:.1f} kN  (one-sided rad/s PSD ∫ S dω)")

    print("\n=== Closed-loop sway σ_y under various models ===")
    print(f"{'model':<55} {'σ_y [m]':>10} {'ω_peak [rad/s]':>16} {'stable':>8}")
    print("-" * 92)

    cases = [
        ("perfect FB, no integrator (cqa equivalent)",
         dict(use_integrator=False, use_bias_ff=False,
              use_wave_filter=False, use_observer=False)),
        ("perfect FB, with PI integrator",
         dict(use_integrator=True, use_bias_ff=False,
              use_wave_filter=False, use_observer=False)),
        ("observer no-WF, no bias-FF, no integrator",
         dict(use_integrator=False, use_bias_ff=False,
              use_wave_filter=False, use_observer=True)),
        ("observer no-WF, bias-FF",
         dict(use_integrator=False, use_bias_ff=True,
              use_wave_filter=False, use_observer=True)),
        ("observer with WF, no bias-FF",
         dict(use_integrator=False, use_bias_ff=False,
              use_wave_filter=True, use_observer=True)),
        ("observer with WF + bias-FF (full brucon-like)",
         dict(use_integrator=False, use_bias_ff=True,
              use_wave_filter=True, use_observer=True)),
        ("observer with WF + bias-FF + integrator",
         dict(use_integrator=True, use_bias_ff=True,
              use_wave_filter=True, use_observer=True)),
        ("full obs + thrust lag τ=2s",
         dict(use_integrator=False, use_bias_ff=True,
              use_wave_filter=True, use_observer=True, thrust_tau=2.0)),
        ("full obs + thrust lag τ=5s",
         dict(use_integrator=False, use_bias_ff=True,
              use_wave_filter=True, use_observer=True, thrust_tau=5.0)),
        ("full obs + thrust lag τ=10s",
         dict(use_integrator=False, use_bias_ff=True,
              use_wave_filter=True, use_observer=True, thrust_tau=10.0)),
        ("full obs + thrust lag τ=10s + integrator",
         dict(use_integrator=True, use_bias_ff=True,
              use_wave_filter=True, use_observer=True, thrust_tau=10.0)),
    ]

    for label, kw in cases:
        A, B_w, _, C_y, _ = build_closed_loop(**kw)
        eig, stab = stable_eigvals(A)
        if not stab:
            print(f"{label:<55} {'-':>10} {'-':>16} {'NO':>8}  "
                  f"max Re(eig) = {eig.real.max():+.3e}")
            continue
        sigma2, w_peak = state_variance_freqdomain(A, B_w, S_drift, omega, C_y)
        sigma = np.sqrt(sigma2)
        print(f"{label:<55} {sigma:>10.3f} {w_peak:>16.4f} {'yes':>8}")

    print(f"\nBrucon target σ_y_total = 0.79 m (LF+WF), σ_y_LF only = 0.59 m")
    print(f"cqa current value (PD only, perfect FB, no int) = ~0.255 m")


if __name__ == "__main__":
    main()
