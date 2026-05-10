"""Brucon roll-up of the gangway-telescope bar in the live operator panel.

|dL| comparator (mirrors the WCF position-bar comparator)
---------------------------------------------------------
After an initial signed-dL prototype showed structural under-
prediction in bf6/bf8, the live operator panel was reverted to the
|dL| metric (folded-normal halo about the deterministic |dL| peak
under WCFDI) with a single worst-margin red threshold
``min(L_max - L0, L0 - L_min)``. This script validates that
|dL| metric against brucon truth in the same apples-to-apples
fashion as ``roll_up_live_operator_panel.py`` does for the WCF
radial position bar:

    pred  : panel ``gangway_dL_excursion_p50`` /
            ``gangway_dL_excursion_p95``. Two regimes (gated on
            whether ``sigma_post.sigma_dL_wf_measured`` and
            ``sigma_post.T_zc_dL_wf_measured`` are populated):

            (a) "window-max" -- with both measured fields present,
                the panel returns the LF-peak + Gumbel-WF-max
                statistic::

                    pred_pq = |c . delta_eta_mean(t_peak)|
                              + a_q(N_eff) * sigma_dL_wf_measured

                with N_eff = tau_LF / T_zc_dL_wf_measured, tau_LF =
                duration the |LF transient| stays >= 0.8*peak, a_q
                from the Gumbel/Rice peak factor (a_50 = sqrt(2 ln
                N_eff) - gamma/sqrt(2 ln N_eff), gamma = Euler-
                Mascheroni). This is the apples-to-apples
                companion to a max-over-window truth statistic.

            (b) "folded-normal fallback" -- single-instant folded-
                normal halo at the deterministic peak. Structurally
                low for the window-max truth comparator.

    truth : per seed, ``s_max_abs = max |dL_truth(t) - dL_pre|``
            over the post-WCF window, where ``dL_truth(t) = c6 .
            eta_full_6(t)`` and the demean is performed channel-
            wise on the pre-WCF window. Quantiles **across seeds**
            of single-realisation s_max_abs give the truth P50 / P95.

Direct gangway-channel WF posterior
-----------------------------------
The loader synthesizes a pre-WCF ``dL_wf_samples = c6 . eta_wf_full``
series by projecting the per-DOF WF samples through the 6-DOF
telescope sensitivity, demeans, and reports back
``sigma_dL_wf_measured`` (m, std of the demeaned series) and
``T_zc_dL_wf_measured`` (s, mean zero-up-crossing period). This is
WF-only on purpose: the Gumbel formula assumes a narrow-band Gaussian
process where the peak factor multiplies sigma by sqrt(2 ln N_eff),
which would massively overcount the LF tail if the LF channel were
included (the LF has an effective N_eff of ~1 over a 30 s near-peak
window). The LF deterministic peak is already accounted for additively
via |c . delta_eta_mean(t_peak)|, so combining LF-deterministic +
WF-stochastic is the right decomposition.

Earlier comparator-shape mistakes (now fixed)
---------------------------------------------
Two predecessors of the current pred formulation were considered and
rejected:

  * ``pred = gangway_dL_p* - gangway_dL_intact_offset`` (magnitude of
    magnitudes subtraction). NOT equivalent to the apples-to-apples
    excursion: folds the WF noise around a non-zero ``c3 .
    eta_hat_lf + c3 . delta_eta_mean`` mean, then subtracts ``|c3 .
    eta_hat_lf|``, biasing the magnitude and double-counting the
    live offset when its sign differs from the WCFDI excursion.

  * ``pred = folded_normal(c . delta_eta_mean(t_peak), sigma_dL_wcf)``
    at the deterministic peak instant (per-instant statistic).
    Structurally low when truth is max-over-window: for Gaussian
    noise over N=20 crests, max/median ratio ~2.5 -- exactly the
    P50 gap the user flagged.

The current LF + Gumbel-WF formulation eliminates both issues. The
operator-facing ``gangway_dL_*`` (which include the live LF baseline)
remain the right thing for the panel UI.

Residual physics gap
--------------------
After the comparator fix, P50 bias improves on every cell (e.g.
bf6_h0 -52% -> -42%, bf8_h0 -47% -> -41%). P95 bias on energetic
cells (bf8 -32%, pwo -59%) is now cleanly attributable to the LF
transient model itself: on bf8_h0 seed 1000 the predicted LF surge
peak is 1.13 m at t=32s but truth peaks at 2.05 m at t=76s -- 1.8x
under and ~45 s mis-located in time. This is the documented
dF/dpsi mirror term physics gap (analysis.md sec.12) which the
position bar inherits too; the position bar's larger sigma_R
envelope masks it more effectively in cell-aggregate, but the
gangway's tighter Gumbel halo exposes it directly. Resolving the
LF transient model will improve both bars at once.

Coverage (full 6-DOF; matches the loader)
-----------------------------------------
``live_cell_per_seed_pwq30.build_live_sigma_posterior`` now plumbs
WF roll/pitch/heave posteriors derived from brucon's ``Roll``,
``Pitch`` and ``Heave`` channels (windowed-demeaned -- the LF
component is effectively zero on a DP CSOV, and there is no
``RollHf``/``PitchHf``/``HeaveHf`` split in the brucon log
header), so the panel reports
``gangway_wf_coverage = "full_6dof"`` for every seed and the
truth comparator below uses the matching c6 6-DOF projection of
brucon's full motion.

Joint geometry (held fixed across all 12 cells)
----------------------------------------------
Forward-pointing horizontal gangway, mid-stroke length::

    h        = 15.0  m  (rotation centre 15 m above the gangway base)
    alpha_g  = 0.0   rad
    beta_g   = 0.0   rad
    L0       = 25.0  m  (stroke = min(7, 7) = 7 m)

This is a "hypothetical landing setpoint" -- the brucon scenarios
are sea-trial DP runs without an actual W2W landing target, so the
joint state was chosen to expose the in-plane variance (forward
gangway -> surge dominates dL via c6[0]=-1, contributes via yaw
lever-arm c6[5]=-9 m/rad, and now via pitch lever-arm c6[4]=+23
m/rad as the dominant out-of-plane contributor).

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/roll_up_gangway_bar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

import live_cell_per_seed_pwq30 as live_cell                     # noqa: E402
from live_cell_per_seed_pwq30 import (                            # noqa: E402
    _load_tsv,
    load_seed,
    build_live_sigma_posterior,
)

from cqa.config import csov_default_config                       # noqa: E402
from cqa.live_decision import LiveObserverState                  # noqa: E402
from cqa.live_operator_view import summarise_for_operator_live   # noqa: E402
from cqa.gangway import (                                         # noqa: E402
    GangwayJointState,
    telescope_sensitivity,
    telescope_sensitivity_6dof,
)


CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0",    "bf6_q10",    "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0",    "bf8_q10",    "bf8_h0_w45", "bf8_q10_w45",
    "pwo",       "pwq30",
]
T_WCF = 560.0
SEED_LO, SEED_HI = 1000, 1030

# Fixed joint geometry (see module docstring).
JOINT = GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)


def _set_cell(tag: str) -> None:
    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_LO, SEED_HI))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"


def _validate_cell(tag: str, cfg, c6: np.ndarray) -> dict | None:
    _set_cell(tag)
    calib_npz = live_cell.CALIB_NPZ
    if not calib_npz.exists():
        return None
    sigma_R_b_hat_m = float(np.load(calib_npz, allow_pickle=True)["sigma_R_b_hat_m"])

    rows = []
    for seed in live_cell.SEEDS:
        d = load_seed(seed)
        if d is None:
            continue
        sigma_post = build_live_sigma_posterior(
            d, sigma_R_b_hat_m=sigma_R_b_hat_m, joint=JOINT, cfg=cfg,
        )
        obs = LiveObserverState(
            eta_hat=d["eta_hat"], nu_hat=d["nu_hat"], b_hat=d["b_hat"],
            eta_wave=d["eta_wave"],
            heading_compass=float(d.get("heading_compass", 0.0)),
        )
        s = summarise_for_operator_live(cfg, obs, sigma_post, joint=JOINT)
        # With the loader plumbing roll/pitch/heave WF posteriors
        # always-on, the panel must now report full_6dof coverage.
        assert s.gangway_wf_coverage == "full_6dof"

        # Reload the brucon main log to project full 6-DOF motion
        # onto the telescope axis. Roll/Pitch/Heave channels are
        # TOTAL motion (no HF split); we rely on the same windowed-
        # demean used inside the loader to remove the LF component
        # (which is effectively zero on a DP CSOV anyway -- the
        # mean-trim shift is what gets removed).
        seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out")
                       if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t_main = M["t"]

        eta_full_6 = np.column_stack([
            M["SurgeDev"] + M["xHf"],
            M["SwayDev"] + M["yHf"],
            M["Heave"],
            np.deg2rad(M["Roll"]),
            np.deg2rad(M["Pitch"]),
            np.deg2rad(M["HeadingDev"] + M["headingHf"]),
        ])
        dL_truth = eta_full_6 @ c6

        win_m = (t_main >= live_cell.WIN_START) & (t_main <= live_cell.WIN_END)
        post_mask = ((t_main > live_cell.T_EVAL + 5.0)
                     & (t_main <= live_cell.T_EVAL + 120.0))
        dL_pre = float(dL_truth[win_m].mean())
        # |dL| max-over-window of vector-demeaned dL_truth; analogous
        # to the WCF position-bar truth statistic
        # max |hypot(SurgeDev-sd_pre, SwayDev-wd_pre)| over post-WCF.
        s_max_abs = float(np.max(np.abs(dL_truth[post_mask] - dL_pre)))

        # Apples-to-apples comparator: truth is max|dL_truth - dL_pre|
        # (vector-demeaned post-WCF excursion). The matching prediction
        # is the folded-normal of the WCFDI-induced *change* in dL with
        # the live LF baseline removed -- exposed by the panel as
        # gangway_dL_excursion_p50/p95. The operator-facing
        # gangway_dL_p50/p95 (which include the live LF baseline) are
        # the right thing for the UI but NOT for this comparator.
        pred_p50 = s.gangway_dL_excursion_p50
        pred_p95 = s.gangway_dL_excursion_p95
        pred_det = (s.gangway_dL_wcf_offset_at_peak
                    - s.gangway_dL_intact_offset)

        rows.append(dict(
            seed=seed,
            pred_p50=pred_p50,
            pred_p95=pred_p95,
            pred_det=pred_det,
            sigma_dL_wcf=s.gangway_sigma_dL_wcf_m,
            t_peak=s.gangway_t_peak_s,
            traffic=s.gangway_traffic,
            s_max_abs=s_max_abs,
        ))
    if not rows:
        return None

    p50_pred = np.array([r["pred_p50"] for r in rows])
    p95_pred = np.array([r["pred_p95"] for r in rows])
    s_abs = np.array([r["s_max_abs"] for r in rows])

    # Truth quantiles across the seed ensemble: distribution of
    # single-realisation max-|dL| post-WCF excursion.
    truth_p50 = float(np.quantile(s_abs, 0.50))
    truth_p95 = float(np.quantile(s_abs, 0.95))

    pred_p50_mean = float(p50_pred.mean())
    pred_p95_mean = float(p95_pred.mean())

    # Coverage of truth |dL| by the predicted P95 envelope, per seed.
    cov_p95 = float(np.mean(s_abs <= p95_pred))

    return dict(
        tag=tag,
        n_seeds=len(rows),
        pred_p50_mean=pred_p50_mean,
        pred_p95_mean=pred_p95_mean,
        truth_p50=truth_p50,
        truth_p95=truth_p95,
        # Bias on the matched magnitude quantiles. Use a denominator
        # floor (0.05 m, ~5 cm) to keep near-zero cells from blowing
        # up the percent.
        bias_p50_pct=100 * (pred_p50_mean - truth_p50)
                          / max(truth_p50, 0.05),
        bias_p95_pct=100 * (pred_p95_mean - truth_p95)
                          / max(truth_p95, 0.05),
        cov_p95=cov_p95,
        sigma_dL_wcf_mean=float(np.mean([r["sigma_dL_wcf"] for r in rows])),
        n_red=sum(r["traffic"] == "red" for r in rows),
        n_amber=sum(r["traffic"] == "amber" for r in rows),
        n_green=sum(r["traffic"] == "green" for r in rows),
    )


def main() -> int:
    cfg = csov_default_config()
    c3 = telescope_sensitivity(JOINT, cfg.gangway)
    c6 = telescope_sensitivity_6dof(JOINT, cfg.gangway)

    print(f"Joint:      h={JOINT.h:.1f} m, alpha={JOINT.alpha_g:.2f} rad, "
          f"beta={JOINT.beta_g:.2f} rad, L0={JOINT.L:.1f} m")
    print(f"c3:         {c3}  (m/m, m/m, m/rad)")
    print(f"c6:         {c6}  "
          f"(surge, sway, heave, roll, pitch, yaw)")
    print(f"Coverage:   full_6dof "
          f"(WF roll/pitch/heave posteriors plumbed in loader)\n")

    results = []
    for tag in CELLS:
        m = _validate_cell(tag, cfg, c6)
        if m is None:
            print(f"  {tag}: skipped (no data)")
            continue
        results.append(m)
        print(f"  {tag}: validated {m['n_seeds']} seeds")
    print()

    print("Gangway-telescope bar vs brucon truth (full 6-DOF projection)")
    print("Pred is |dL|_p* relative to the live LF \"now\" |dL| offset.")
    print("Truth is per-seed max |dL_truth - dL_pre| over post-WCF window")
    print("(c6 . eta_full_6 with eta_full_6 = surge,sway,heave,roll,pitch,yaw),")
    print("then quantiles across seeds.\n")

    hdr = (f"{'cell':<14} {'N':>3}  "
           f"{'p50.pr':>7} {'p50.tr':>7} {'b50%':>6}  "
           f"{'p95.pr':>7} {'p95.tr':>7} {'b95%':>6} {'cov%':>5}  "
           f"{'sig_dL':>7}  {'g/a/r':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(f"{m['tag']:<14} {m['n_seeds']:>3}  "
              f"{m['pred_p50_mean']:>7.3f} {m['truth_p50']:>7.3f} "
              f"{m['bias_p50_pct']:>+5.0f}%  "
              f"{m['pred_p95_mean']:>7.3f} {m['truth_p95']:>7.3f} "
              f"{m['bias_p95_pct']:>+5.0f}% {100*m['cov_p95']:>4.0f}%  "
              f"{m['sigma_dL_wcf_mean']:>7.3f}  "
              f"{m['n_green']}/{m['n_amber']}/{m['n_red']:<5}")

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.0))
    tags = [m["tag"] for m in results]
    x = np.arange(len(tags))
    width = 0.20

    ax.bar(x - 1.5 * width, [m["pred_p50_mean"] for m in results],
           width=width, color="#1f77b4", alpha=0.85, label="pred P50")
    ax.bar(x - 0.5 * width, [m["truth_p50"] for m in results],
           width=width, color="#1f77b4", alpha=0.45, edgecolor="#1f77b4",
           hatch="//", label="brucon P50")
    ax.bar(x + 0.5 * width, [m["pred_p95_mean"] for m in results],
           width=width, color="#d62728", alpha=0.85, label="pred P95")
    ax.bar(x + 1.5 * width, [m["truth_p95"] for m in results],
           width=width, color="#d62728", alpha=0.45, edgecolor="#d62728",
           hatch="//", label="brucon P95")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|dL| relative to pre-WCF mean [m]")
    ax.set_title(
        f"Gangway-telescope bar  -  12-cell brucon roll-up  "
        f"|dL| pred vs truth\n"
        f"(joint forward, h=15 m, L0=25 m; full 6-DOF)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = THIS / "roll_up_gangway_bar.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
