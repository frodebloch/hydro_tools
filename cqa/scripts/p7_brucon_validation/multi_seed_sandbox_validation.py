"""Multi-seed sandbox cross-validation.

Aggregates time-series predictions from `validate_sandbox_timeseries.py`
across all 30 P7 waves-only seeds (pwo_seed1000..1029) to produce robust
sigma statistics in the intact window. Compares sandbox model variants
against brucon ensemble.

Aggregation: per-seed sigma over the intact window [T_START, T_END], then
report median, IQR, and a long-record-equivalent σ obtained by treating
all seeds as concatenated samples.

Usage
-----
    .venv/bin/python scripts/p7_brucon_validation/multi_seed_sandbox_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sandbox_passive_observer import build_closed_loop  # noqa: E402
from validate_sandbox_timeseries import (  # noqa: E402
    load_seed_csv, project_ned_to_body, simulate_lti,
)

T_START = 300.0
T_END = 555.0


def per_seed_results(seed_dir: Path, cases: list[tuple[str, dict]]):
    """Return dict {key -> (sigma_value, residual_array_for_pooling)}."""
    cols = load_seed_csv(seed_dir)
    t = cols["t"]
    mask = (t >= T_START) & (t <= T_END)
    t_w = t[mask] - t[mask][0]

    sway_total = project_ned_to_body(cols["x"], cols["y"], cols["heading"])[1]
    sway_lf = sway_total - cols["yHf"]
    sway_lf_w = sway_lf[mask] - sway_lf[mask].mean()
    sway_dev_w = cols["SwayDev"][mask] - cols["SwayDev"][mask].mean()
    sway_total_w = sway_total[mask] - sway_total[mask].mean()

    F_drift = cols["DriftY"][mask] * 1000.0
    F_drift = F_drift - F_drift.mean()
    y_wf_meas = cols["yHf"][mask] - cols["yHf"][mask].mean()

    out = {
        "brucon_sway_total": sway_total_w,
        "brucon_sway_LF":    sway_lf_w,
        "brucon_SwayDev":    sway_dev_w,
        "F_drift":           F_drift,
    }
    for label, kw in cases:
        A, B_w, B_wf, _, _ = build_closed_loop(**kw)
        x = simulate_lti(A, B_w, B_wf, t_w, F_drift, y_wf_meas)
        y_pred = x[0, :]
        out[label] = y_pred
    return t_w, out


def main() -> None:
    work = Path(__file__).parent / "work"
    seed_dirs = sorted(work.glob("pwo_seed*"))
    print(f"Found {len(seed_dirs)} seeds in {work}")

    cases = [
        ("perfect_FB",            dict(use_observer=False, use_bias_ff=False, use_wave_filter=False,
                                       use_integrator=False, thrust_tau=0.0)),
        ("full_obs_no_lag",       dict(use_observer=True,  use_bias_ff=True,  use_wave_filter=True,
                                       use_integrator=False, thrust_tau=0.0)),
        ("full_obs_tau5",         dict(use_observer=True,  use_bias_ff=True,  use_wave_filter=True,
                                       use_integrator=False, thrust_tau=5.0)),
        ("full_obs_tau10",        dict(use_observer=True,  use_bias_ff=True,  use_wave_filter=True,
                                       use_integrator=False, thrust_tau=10.0)),
    ]
    keys_brucon = ["brucon_sway_total", "brucon_sway_LF", "brucon_SwayDev"]
    keys_sandbox = [c[0] for c in cases]

    # Per-seed sigmas
    sigmas = {k: [] for k in keys_brucon + keys_sandbox + ["F_drift_kN"]}
    pooled = {k: [] for k in keys_brucon + keys_sandbox}

    for sd in seed_dirs:
        try:
            t_w, results = per_seed_results(sd, cases)
        except Exception as e:
            print(f"  skip {sd.name}: {e}")
            continue
        for k in keys_brucon + keys_sandbox:
            arr = results[k]
            sigmas[k].append(arr.std())
            pooled[k].append(arr)
        sigmas["F_drift_kN"].append(results["F_drift"].std() / 1000.0)
        if sd.name == "pwo_seed1000":
            print(f"  {sd.name}: σ_LF_brucon={results['brucon_sway_LF'].std():.3f}, "
                  f"σ_LF_sand_full={results['full_obs_no_lag'].std():.3f}")

    print(f"\nUsable seeds: {len(sigmas['brucon_sway_LF'])}")
    print(f"\n{'channel':<25} {'median':>10} {'IQR':>14} {'pooled σ':>12} {'mean σ':>10}")
    print("-" * 75)
    for k in keys_brucon + keys_sandbox + ["F_drift_kN"]:
        arr = np.array(sigmas[k])
        med = np.median(arr)
        q25 = np.quantile(arr, 0.25)
        q75 = np.quantile(arr, 0.75)
        if k != "F_drift_kN":
            pool = np.concatenate(pooled[k])
            psig = pool.std()
        else:
            psig = float("nan")
        print(f"{k:<25} {med:>10.3f} [{q25:>5.3f},{q75:>5.3f}]  {psig:>12.3f} {arr.mean():>10.3f}")

    # Plot 1: per-seed sigma scatter (brucon vs sandbox cases)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    brucon_sig = np.array(sigmas["brucon_sway_LF"])
    colors = ["C0", "C1", "C2", "C3"]
    for k, c in zip(keys_sandbox, colors):
        s = np.array(sigmas[k])
        ax.scatter(brucon_sig, s, alpha=0.7, c=c, label=f"{k} (mean σ_pred/σ_brucon = {s.mean()/brucon_sig.mean():.2f})")
    lim = [0, max(brucon_sig.max(), 1.5)]
    ax.plot(lim, lim, "k--", alpha=0.4, label="1:1")
    ax.set_xlabel("brucon σ(sway_LF) [m]"); ax.set_ylabel("sandbox σ(y) [m]")
    ax.set_title(f"Per-seed σ comparison (P7 waves-only, intact window {T_START:.0f}-{T_END:.0f}s, n={len(brucon_sig)})")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax.set_aspect("equal"); ax.set_xlim(lim); ax.set_ylim(lim)
    out = Path(__file__).parent / "multi_seed_sandbox_sigma_scatter.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")

    # Plot 2: pooled time-domain PSDs
    import scipy.signal
    dt = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for k in ["brucon_sway_LF", "perfect_FB", "full_obs_no_lag", "full_obs_tau5", "full_obs_tau10"]:
        # PSD per seed, then average
        Ps = []
        for arr in pooled[k]:
            f, P = scipy.signal.welch(arr, fs=1/dt, nperseg=1024)
            Ps.append(P)
        P_mean = np.mean(Ps, axis=0)
        ax.loglog(f, P_mean, label=k, lw=1.5 if k == "brucon_sway_LF" else 1.0)
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("PSD [m²/Hz]"); ax.grid(alpha=0.3, which="both")
    ax.set_xlim(1e-3, 0.1); ax.set_ylim(1e-3, 1e3)
    ax.legend(fontsize=8)
    ax.set_title(f"Ensemble-mean PSD (n={len(brucon_sig)} seeds, intact window)")
    out2 = Path(__file__).parent / "multi_seed_sandbox_psd.png"
    plt.tight_layout(); plt.savefig(out2, dpi=120)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
