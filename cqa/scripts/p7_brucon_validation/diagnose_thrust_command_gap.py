"""Compare commanded thrust variance: brucon vs sandbox.

If brucon commands more thrust than sandbox but still has higher sway sigma,
the missing physics is on the disturbance/observer side. If brucon commands
less thrust, it's a controller-tuning mismatch.

Also computes the actual ratio of sigma(F_drift) vs sigma(OrderTau) in brucon
vs sandbox -- if brucon has additional unmodelled disturbances entering the
vessel (beyond the logged DriftY), this should show as larger sigma(thrust)
that cannot be explained by sigma(F_drift) alone.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox_passive_observer import build_closed_loop, M_SWAY, KP, KD  # noqa: E402
from validate_sandbox_timeseries import (  # noqa: E402
    load_seed_csv, project_ned_to_body, simulate_lti,
)

T_START = 300.0
T_END = 555.0


def main() -> None:
    work = Path(__file__).parent / "work"
    seeds = sorted(work.glob("pwo_seed*"))
    print(f"Found {len(seeds)} seeds")

    sig = {k: [] for k in [
        "brucon_F_drift_kN", "brucon_OrderTauSway_kN", "brucon_Ty_kN",
        "brucon_sway_LF_m", "brucon_sway_total_m",
        "sand_u_cmd_full_kN", "sand_u_cmd_tau5_kN",
        "sand_y_full_m", "sand_y_tau5_m",
    ]}

    cases = {
        "full":  dict(use_observer=True, use_bias_ff=True, use_wave_filter=True,
                      use_integrator=False, thrust_tau=0.0),
        "tau5":  dict(use_observer=True, use_bias_ff=True, use_wave_filter=True,
                      use_integrator=False, thrust_tau=5.0),
    }

    for sd in seeds:
        try:
            cols = load_seed_csv(sd)
        except Exception:
            continue
        t = cols["t"]; m = (t >= T_START) & (t <= T_END)
        t_w = t[m] - t[m][0]
        sway_total = project_ned_to_body(cols["x"], cols["y"], cols["heading"])[1]
        sway_lf = sway_total - cols["yHf"]
        F_drift_N = cols["DriftY"][m] * 1000.0; F_drift_N -= F_drift_N.mean()
        OrderTau_N = cols["OrderTauSway"][m] * 1000.0; OrderTau_N -= OrderTau_N.mean()
        Ty_N = cols["Ty"][m] * 1000.0; Ty_N -= Ty_N.mean()
        y_wf = cols["yHf"][m] - cols["yHf"][m].mean()

        sig["brucon_F_drift_kN"].append(F_drift_N.std() / 1000.0)
        sig["brucon_OrderTauSway_kN"].append(OrderTau_N.std() / 1000.0)
        sig["brucon_Ty_kN"].append(Ty_N.std() / 1000.0)
        sig["brucon_sway_LF_m"].append((sway_lf[m] - sway_lf[m].mean()).std())
        sig["brucon_sway_total_m"].append((sway_total[m] - sway_total[m].mean()).std())

        for label, kw in cases.items():
            A, Bd, Bw, _, _ = build_closed_loop(**kw)
            x = simulate_lti(A, Bd, Bw, t_w, F_drift_N, y_wf)
            y_pred = x[0, :]
            yLF_pred = x[2, :]
            vh_pred = x[3, :]
            b_pred = x[4, :]
            # Reconstruct u_cmd = -KP*yLF - KD*vh - b
            u_cmd = -KP * yLF_pred - KD * vh_pred - b_pred
            sig[f"sand_u_cmd_{label}_kN"].append(u_cmd.std() / 1000.0)
            sig[f"sand_y_{label}_m"].append(y_pred.std())

    print(f"\nMedian over {len(sig['brucon_F_drift_kN'])} seeds:\n")
    print(f"  brucon F_drift:        {np.median(sig['brucon_F_drift_kN']):>6.1f} kN")
    print(f"  brucon OrderTauSway:   {np.median(sig['brucon_OrderTauSway_kN']):>6.1f} kN  (controller command)")
    print(f"  brucon Ty (applied):   {np.median(sig['brucon_Ty_kN']):>6.1f} kN  (force on vessel)")
    print(f"  brucon sway_LF:        {np.median(sig['brucon_sway_LF_m']):>6.3f} m")
    print(f"  brucon sway_total:     {np.median(sig['brucon_sway_total_m']):>6.3f} m")
    print()
    print(f"  sandbox u_cmd full:    {np.median(sig['sand_u_cmd_full_kN']):>6.1f} kN  (no thrust lag)")
    print(f"  sandbox y full:        {np.median(sig['sand_y_full_m']):>6.3f} m")
    print(f"  sandbox u_cmd tau5:    {np.median(sig['sand_u_cmd_tau5_kN']):>6.1f} kN  (with thrust lag)")
    print(f"  sandbox y tau5:        {np.median(sig['sand_y_tau5_m']):>6.3f} m")
    print()
    # Key ratios
    print("RATIOS (median):")
    print(f"  brucon  σ(OrderTau) / σ(F_drift) = {np.median(sig['brucon_OrderTauSway_kN'])/np.median(sig['brucon_F_drift_kN']):.2f}")
    print(f"  sandbox σ(u_cmd_full) / σ(F_drift) = {np.median(sig['sand_u_cmd_full_kN'])/np.median(sig['brucon_F_drift_kN']):.2f}")
    print(f"  brucon  σ(Ty - OrderTau)... computing per-seed:")
    diffs = []
    for sd in seeds:
        try:
            cols = load_seed_csv(sd)
        except Exception:
            continue
        t = cols["t"]; m = (t >= T_START) & (t <= T_END)
        d = (cols["Ty"][m] - cols["OrderTauSway"][m])
        d -= d.mean()
        diffs.append(d.std())
    print(f"     median σ(Ty-OrderTau) = {np.median(diffs):.1f} kN  (would be ~0 for ideal actuator)")


if __name__ == "__main__":
    main()
