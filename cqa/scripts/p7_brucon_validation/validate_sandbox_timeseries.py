"""Time-domain sandbox validation: drive sandbox closed-loop with brucon's
actual wave-drift force from one seed and compare predicted vs measured
sway position.

The sandbox closed loop (vessel + Fossen passive observer + 2nd-order
wave filter + optional 1st-order thrust lag + optional integrator) is
exercised on the same `DriftY` time series brucon used, then compared
against brucon's body-frame LF sway in the intact window
[T_START, T_END].

See `multi_seed_sandbox_validation.py` for the 30-seed batch wrapper
and `analysis.md §12.20` for the full diagnostic narrative.

Usage
-----
    .venv/bin/python scripts/p7_brucon_validation/validate_sandbox_timeseries.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse build_closed_loop and constants from sandbox script
from sandbox_passive_observer import (  # noqa: E402
    build_closed_loop, M_SWAY, D_SWAY, KP, KD, KI, KA1, KB1, T_B,
    OMEGA_C, OMEGA_W, ZETA_N, K1F, K2F, HEADING_NED,
)


def load_seed_csv(seed_dir: Path) -> dict[str, np.ndarray]:
    """Load brucon .out file (tab-separated, single header row)."""
    out_path = next(seed_dir.glob("*.out"))
    if "estimator" in out_path.name:
        # pick the main one
        out_path = next(p for p in seed_dir.glob("*.out") if "estimator" not in p.name)
    with open(out_path) as f:
        header = f.readline().strip().split("\t")
    data = np.loadtxt(out_path, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(header)}


def project_ned_to_body(north: np.ndarray, east: np.ndarray,
                        heading_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NED (north,east) -> body (surge,sway) given heading [deg, NED frame].
    surge = +cos(h)·N + sin(h)·E
    sway  = -sin(h)·N + cos(h)·E
    """
    h_rad = np.deg2rad(heading_deg)
    cos_h = np.cos(h_rad)
    sin_h = np.sin(h_rad)
    surge =  cos_h * north + sin_h * east
    sway  = -sin_h * north + cos_h * east
    return surge, sway


def simulate_lti(A: np.ndarray, B_drift: np.ndarray, B_wf: np.ndarray,
                 t: np.ndarray, F_drift: np.ndarray, y_wf_meas: np.ndarray,
                 x0: np.ndarray | None = None) -> np.ndarray:
    """Simulate dx/dt = A x + B_drift·F_drift(t) + B_wf·y_wf_meas(t).

    Returns x(t) of shape (n, len(t)).
    Uses scipy.signal.lsim with combined input.
    """
    n = A.shape[0]
    # System with two inputs stacked: u = [F_drift; y_wf_meas]
    B = np.column_stack([B_drift, B_wf])
    C = np.eye(n)
    D = np.zeros((n, 2))
    sys = scipy.signal.StateSpace(A, B, C, D)
    U = np.column_stack([F_drift, y_wf_meas])
    if x0 is None:
        x0 = np.zeros(n)
    _, y_out, x_out = scipy.signal.lsim(sys, U=U, T=t, X0=x0)
    return x_out.T  # (n, len(t))


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed_dir = Path(__file__).parent / "work" / f"pwo_seed{seed}"
    print(f"Loading {seed_dir} ...")
    cols = load_seed_csv(seed_dir)
    print(f"  {len(cols)} columns, {len(cols['t'])} rows, dt={cols['t'][1]-cols['t'][0]:.3f}s")

    t = cols["t"]
    dt = t[1] - t[0]

    # Brucon outputs:
    #   x, y         : true NED position deltas from start [m]   (LF + WF combined)
    #   xHf, yHf     : BODY-frame HF/wave-frequency motion [m]   (already body)
    #   heading      : true heading [deg, NED]
    #   DriftY       : body-frame slow-drift force [kN]  (sway component)
    #   SwayDev      : DP estimator's LF body-frame sway estimate [m]
    # Project (x,y) NED to body using true heading; xHf/yHf already body.
    surge_total_body, sway_total_body = project_ned_to_body(cols["x"], cols["y"], cols["heading"])
    sway_hf_body = cols["yHf"]                  # already body-frame
    sway_lf_body = sway_total_body - sway_hf_body

    # Subtract initial transient: the lua script holds the vessel fixed up to
    # t=60s, then activates station keeping. WCFDI failure is injected at
    # t=560s (i_fail=5600 in the lua, 10Hz). So the *intact* window is
    # t in [60, 560]; we discard the first 240s of station-keeping settling
    # (~3 closed-loop time constants at ω_n=0.08) and stop before the failure.
    T_START = 300.0
    T_END = 555.0
    mask = (t >= T_START) & (t <= T_END)
    t_w = t[mask] - t[mask][0]
    sway_total_w = sway_total_body[mask] - sway_total_body[mask].mean()
    sway_lf_w = sway_lf_body[mask] - sway_lf_body[mask].mean()
    sway_dev_w = cols["SwayDev"][mask] - cols["SwayDev"][mask].mean()
    sway_hf_w = sway_hf_body[mask] - sway_hf_body[mask].mean()

    # Driving force: brucon DriftY [kN] -> N
    F_drift_N = cols["DriftY"][mask] * 1000.0
    # Demean to remove static offset (the controller's bias FF would absorb it
    # in steady state, but our linear model with finite T_b handles it correctly).
    F_drift_N = F_drift_N - F_drift_N.mean()

    # HF measurement signal injected into y_meas (so observer's wave filter
    # sees something). We use brucon's true HF sway as a proxy.
    y_wf_meas = sway_hf_w

    print(f"\nBrucon stats over t > 200s ({len(t_w)} samples):")
    print(f"  σ(sway_total_body) = {sway_total_w.std():.3f} m")
    print(f"  σ(sway_LF_body)    = {sway_lf_w.std():.3f} m")
    print(f"  σ(SwayDev/DP-LF)   = {sway_dev_w.std():.3f} m")
    print(f"  σ(sway_HF_body)    = {sway_hf_w.std():.3f} m")
    print(f"  σ(F_drift)         = {F_drift_N.std()/1000:.1f} kN")

    # Build sandbox: full observer + thrust lag (try a few tau values)
    print("\n=== Sandbox model variants (driven by brucon DriftY) ===")
    print(f"{'model':<55} {'σ_y_pred [m]':>14} {'σ_yLF_pred [m]':>16}")
    print("-" * 88)

    cases = [
        ("perfect FB, no integrator", dict(use_observer=False, use_bias_ff=False,
                                            use_wave_filter=False, use_integrator=False,
                                            thrust_tau=0.0)),
        ("full observer, no thrust lag", dict(use_observer=True, use_bias_ff=True,
                                                use_wave_filter=True, use_integrator=False,
                                                thrust_tau=0.0)),
        ("full observer, thrust lag τ=5s", dict(use_observer=True, use_bias_ff=True,
                                                  use_wave_filter=True, use_integrator=False,
                                                  thrust_tau=5.0)),
        ("full observer, thrust lag τ=10s", dict(use_observer=True, use_bias_ff=True,
                                                   use_wave_filter=True, use_integrator=False,
                                                   thrust_tau=10.0)),
        ("full obs + integrator + thrust lag τ=10s", dict(use_observer=True, use_bias_ff=True,
                                                            use_wave_filter=True, use_integrator=True,
                                                            thrust_tau=10.0)),
    ]

    fig, axes = plt.subplots(len(cases) + 1, 1, figsize=(11, 2.0 * (len(cases) + 1)),
                              sharex=True)
    # First subplot: brucon truth
    ax = axes[0]
    ax.plot(t_w, sway_total_w, "k-", lw=0.8, label="brucon sway_total (LF+WF, body)")
    ax.plot(t_w, sway_lf_w, "b-", lw=0.7, label="brucon sway_LF (body, true)")
    ax.plot(t_w, sway_dev_w, "r--", lw=0.7, label="brucon SwayDev (DP LF est)")
    ax.set_ylabel("sway [m]"); ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)
    ax.set_title(f"Seed {seed}: brucon truth vs sandbox predictions, β=90°")

    sim_results = {}
    for i, (label, kw) in enumerate(cases):
        A, B_w, B_wf, _, _ = build_closed_loop(**kw)
        x = simulate_lti(A, B_w, B_wf, t_w, F_drift_N, y_wf_meas)
        y_pred = x[0, :]                    # true vessel y
        y_pred_lf_obs = x[2, :] if kw["use_observer"] else np.zeros_like(t_w)
        sigma_y = y_pred.std()
        sigma_y_lf = y_pred_lf_obs.std() if kw["use_observer"] else 0.0
        print(f"{label:<55} {sigma_y:>14.3f} {sigma_y_lf:>16.3f}")
        sim_results[label] = (y_pred, y_pred_lf_obs)

        ax = axes[i + 1]
        ax.plot(t_w, sway_lf_w, "b-", lw=0.7, alpha=0.6, label="brucon sway_LF (truth)")
        ax.plot(t_w, y_pred, "C0-", lw=0.7,
                label=f"sandbox y (σ={sigma_y:.3f})")
        if kw["use_observer"]:
            ax.plot(t_w, y_pred_lf_obs, "C3--", lw=0.6,
                    label=f"sandbox ŷ_LF (σ={sigma_y_lf:.3f})")
        ax.set_ylabel("sway [m]")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=9, loc="left")

    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    out = Path(__file__).parent / f"sandbox_validation_seed{seed}.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
