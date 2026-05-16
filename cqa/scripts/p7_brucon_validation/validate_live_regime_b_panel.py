"""End-to-end validation of the regime-B bar in
``summarise_for_operator_live`` against the brucon ensemble
(sec.12.21.21.22).

For each of the 11 calibration cells, for each seed, take a pre-WCF
window of delivered thrust (Tx/Ty/Tz, brucon cols 7/8/9 in kN),
synthesise an otherwise-minimal LiveObserverState, and call the panel
with cap_residual = brucon CSOV bus_port-lost residual polytope. The
per-cell tail-shape diagnostic (sec.12.21.21.21) predicts all cells
should produce regime_b_traffic == "green" with severity < 0.01, so
this script is a sanity / wiring check, not a discriminative test.

Reports per cell: median, 90th-pct, max severity across seeds, and
the fraction of seeds in each traffic light bucket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa import summarise_for_operator_live  # noqa: E402
from cqa.live_decision import LiveObserverState, LiveSigmaPosterior  # noqa: E402
from cqa.online_estimator import SigmaPosterior, RadialPosterior, ValidityBadge  # noqa: E402
from cqa import csov_default_config  # noqa: E402

WORK = THIS / "work"
T_WCF_S = 1560.0
PRE_LO, PRE_HI = T_WCF_S - 600.0, T_WCF_S - 60.0

CAP_RESIDUAL_N_NM = (838.0e3, 1104.0e3, 47929.0e3)
COL_T = 0
COL_TX, COL_TY, COL_TZ = 7, 8, 9  # delivered thrust, kN/kN/kNm

CELLS = [
    "pwq30", "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_h0_w45", "bf6_q10", "bf6_q10_w45",
    "bf8_h0", "bf8_h0_w45", "bf8_q10", "bf8_q10_w45",
]


def _make_minimal_sigma_post() -> LiveSigmaPosterior:
    s = 0.3
    s2 = s * s
    sp = SigmaPosterior(
        sigma2_mean=s2, sigma2_median=s2, sigma2_lo=s2, sigma2_hi=s2,
        sigma_mean=s, sigma_median=s, sigma_lo=s, sigma_hi=s,
        n_raw=100, n_eff=50.0, alpha=10.0, beta=s2 * 9.0,
        prior_sigma2=s2, prior_strength_n0=2.0, credible=0.90,
    )
    rp = RadialPosterior(
        sigma_R_median=s * np.sqrt(2), sigma_R_mean=s * np.sqrt(2),
        sigma_R_lo=s * np.sqrt(2), sigma_R_hi=s * np.sqrt(2),
        expected_R_median=s * np.sqrt(np.pi / 2),
        expected_R_lo=s, expected_R_hi=s,
        n_mc=4000, n_eff_min=10.0, n_eff_x=10.0, n_eff_y=10.0,
        is_warm=True, credible=0.90,
        radial_mean_offset_m=0.0, radial_mean_offset_over_sigma=0.0,
    )
    badge = ValidityBadge(level="OK", reasons=())
    return LiveSigmaPosterior(
        posterior_lf_x=sp, posterior_lf_y=sp, posterior_lf_yaw=sp,
        radial_lf=rp, validity_lf=badge,
        posterior_wf_x=sp, posterior_wf_y=sp, posterior_wf_yaw=sp,
        radial_wf=rp, validity_wf=badge,
        sigma_R_b_hat_m=0.1,
    )


def _load_tau_buffer(cell: str, seed: int) -> tuple[np.ndarray, float] | None:
    f = WORK / f"{cell}_seed{seed}" / f"{cell}_seed{seed}.out"
    if not f.exists():
        return None
    d = np.loadtxt(f, skiprows=1)
    t = d[:, COL_T]
    mask = (t >= PRE_LO) & (t <= PRE_HI)
    if mask.sum() < 100:
        return None
    # brucon log in kN/kN.m -> N/N.m
    tau = 1e3 * d[mask][:, [COL_TX, COL_TY, COL_TZ]]
    # Sample rate from time-step.
    fs_hz = 1.0 / float(np.median(np.diff(t[mask])))
    return tau, fs_hz


def main() -> None:
    cfg = csov_default_config()
    sigma_post = _make_minimal_sigma_post()

    print(f"{'cell':<13} | {'N':>3} | {'sev_med':>9} {'sev_p90':>9} {'sev_max':>9} | "
          f"{'green':>5} {'amber':>5} {'red':>5}")
    print("-" * 92)
    for cell in CELLS:
        severities = []
        traffic_counts = {"green": 0, "amber": 0, "red": 0}
        for seed in range(1000, 1030):
            buf = _load_tau_buffer(cell, seed)
            if buf is None:
                continue
            tau, fs = buf
            obs = LiveObserverState(
                eta_hat=np.zeros(3), nu_hat=np.zeros(3),
                b_hat=np.zeros(3), eta_wave=np.zeros(3),
                heading_compass=0.0,
                tau_buffer=tau, tau_buffer_fs_hz=fs,
            )
            s = summarise_for_operator_live(
                cfg, obs, sigma_post, cap_residual_N_Nm=CAP_RESIDUAL_N_NM,
            )
            severities.append(s.regime_b_severity)
            traffic_counts[s.regime_b_traffic] += 1
        if not severities:
            continue
        sev = np.array(severities)
        print(f"{cell:<13} | {len(sev):>3} | "
              f"{np.median(sev):9.2e} {np.percentile(sev, 90):9.2e} {sev.max():9.2e} | "
              f"{traffic_counts['green']:>5d} {traffic_counts['amber']:>5d} "
              f"{traffic_counts['red']:>5d}")

    print("\nExpectation (per sec.12.21.21.21): all green, severity < 1e-2.")
    print("Min operating headroom is 6.24 sigma (bf8_q10_w45 sway), giving")
    print("P_sat ~ 2e-10; the panel will show this as deep green.")


if __name__ == "__main__":
    main()
