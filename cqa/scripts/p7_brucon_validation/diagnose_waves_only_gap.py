"""Diagnose the 2x P50/P90 |pos| gap in waves-only P7 cross-validation.

Hypothesis under test
---------------------
brucon ensemble (waves only, Vw=0, Vc=0) shows P50/P90 |pos| ~ 1.74 / 2.26 m
with per-seed sigma_sway ~ 0.59 m, while cqa quantiles are 0.85 / 1.03 m.
Force-level (drift) is matched to ~6% (§12.18). So the closed-loop transfer
must differ. Two candidate explanations:

  (A) cqa's closed-loop sigma is correct (~0.6 m, matching brucon's sample
      sigma) but the position autocorrelation time T_decorr is much shorter
      in cqa than in brucon -> cqa sees more effective independent samples
      in a 200 s window -> Cartwright-Longuet-Higgins running-max under-
      predicts.

  (B) cqa's closed-loop sigma itself is wrong. The brucon sample sigma
      (0.59 m) implies running-max ~ sigma * sqrt(2 ln(T/Tz)) ~ 1.7 m for
      T=200s, Tz~1/nu0 ~ 12-30s -- which matches brucon. cqa would need
      sigma ~ 0.4 m to give P50 ~ 0.85 m, i.e. much smaller than 0.59 m.

This script prints both: (i) cqa's predicted sigma_x, sigma_y, nu_0_max,
T_decorr_var_x/y, q from the prior; (ii) brucon's empirical per-axis sigma
and autocorrelation T_decorr from the existing pwo_seed1000..1029 ensemble.

Run from cqa root with:
    .venv/bin/python scripts/p7_brucon_validation/diagnose_waves_only_gap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output, SIM_DT  # noqa: E402

from cqa.decision_matrix import _build_intact_prior_at_forecast  # noqa: E402
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw  # noqa: E402
from cqa.rao import load_pdstrip_rao  # noqa: E402
from run_comparison_waves_only import setup_cqa  # noqa: E402

PDSTRIP_PATH = (
    "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
)

VW_FOR_PM = 14.0
HS = pm_hs_from_vw(VW_FOR_PM)
TP = pm_tp_from_vw(VW_FOR_PM)
WAVE_DIR = 270.0
HEADING = 180.0
T_OP_S = 30.0 * 60.0


def empirical_decorr_time(x: np.ndarray, dt: float) -> tuple[float, float]:
    """Return (T_decorr_int, T_decorr_zerocross) of zero-mean signal x.

    T_decorr_int : integral of normalised autocorr from lag=0 up to first
                   crossing of 0 (in s). This is the variance-of-mean
                   decorrelation time; for a CT-stationary process it
                   matches the cqa convention (∫ rho(tau) d tau).

    T_decorr_zerocross : 2 * (first lag at which rho(tau) <= 1/e) [s].
                         A more local metric, gives a feel for the high-
                         frequency content.
    """
    x = x - x.mean()
    n = len(x)
    # FFT-based autocorr (biased, normalised).
    npad = 1
    while npad < 2 * n:
        npad *= 2
    X = np.fft.rfft(x, n=npad)
    acf = np.fft.irfft(X * np.conj(X), n=npad)[:n]
    acf /= acf[0]
    lags = np.arange(n) * dt
    # Integral up to first zero crossing.
    zc = np.argmax(acf <= 0)
    if zc == 0:
        zc = n // 4  # fallback if never crosses
    T_int = float(np.trapezoid(acf[:zc], lags[:zc]))
    # 1/e width (one-sided), times 2 to get a "T_decorr" comparable
    # to the integral measure for a roughly exponential ACF.
    inv_e = np.argmax(acf <= 1.0 / np.e)
    if inv_e == 0:
        T_e = float("nan")
    else:
        T_e = 2.0 * float(lags[inv_e])
    return T_int, T_e


def predicted_runmax(sigma: float, nu0: float, T: float, q: float = 1.0,
                     p: float = 0.5) -> float:
    """Predict |pos| running-max quantile via Vanmarcke (CLH+clustering).

    For a zero-mean Gaussian process with std sigma, mean zero-up-crossing
    rate nu0 [Hz], over duration T [s]:

      max(|x(t)|) ~ sigma * sqrt( -2 ln( -ln(p) / (2 nu0 T q) ) )

    where q in (0,1] is Vanmarcke's spectral bandwidth correction (q=1
    means narrow-band, q->0 wide-band). This is the same expression used
    in cqa's running_max quantile mapping.
    """
    arg = -np.log(p) / (2.0 * nu0 * T * q)
    return float(sigma * np.sqrt(-2.0 * np.log(arg)))


def main() -> None:
    work = Path(__file__).resolve().parent / "work"

    print("=" * 70)
    print("WAVES-ONLY GAP DIAGNOSIS")
    print("=" * 70)
    print(f"  Hs = {HS:.3f} m,  Tp = {TP:.3f} s,  Vw = 0,  Vc = 0")
    print(f"  T_op (cqa quantile horizon) = {T_OP_S:.0f} s")

    # ------------------------------------------------------------------
    # 1. cqa side: introspect prior
    # ------------------------------------------------------------------
    print("\n[cqa] building intact prior ...")
    cfg, joint = setup_cqa()
    rao = load_pdstrip_rao(PDSTRIP_PATH)
    rel_deg = (WAVE_DIR - HEADING + 540) % 360 - 180
    # Boundary: negate compass-CW bearing -> cqa-internal theta_rel
    # (analysis.md sec.12.21.19).
    theta_rel = np.radians(-rel_deg)

    prior = _build_intact_prior_at_forecast(
        cfg, joint,
        Vw=0.0, Hs=HS, Tp=TP, Vc=0.0,
        theta_rel=theta_rel,
        rao_table=rao,
        sigma_Vc=0.1, tau_Vc=600.0,
        T_op_s=T_OP_S, quantile_p=0.90,
        omega_grid=None, use_pm_for_drift=False,
    )

    print("\n[cqa]  prior at forecast (waves-only):")
    print(f"  pos_sigma_x_m         = {prior.pos_sigma_x_m:.3f} m")
    print(f"  pos_sigma_y_m         = {prior.pos_sigma_y_m:.3f} m")
    print(f"  pos_sigma_m (radial)  = {prior.pos_sigma_m:.3f} m")
    print(f"  pos_nu0_max           = {prior.pos_nu0_max:.4f} Hz "
          f"(Tz_min ~ {1.0 / prior.pos_nu0_max:.1f} s)")
    print(f"  pos_q (Vanmarcke)     = {prior.pos_q:.3f}")
    print(f"  pos_T_decorr_var_x_s  = {prior.pos_T_decorr_var_x_s:.1f} s")
    print(f"  pos_T_decorr_var_y_s  = {prior.pos_T_decorr_var_y_s:.1f} s")
    print(f"  pos_a_p50             = {prior.pos_a_p50:.3f} m")
    print(f"  pos_a_p90             = {prior.pos_a_p90:.3f} m")

    # ------------------------------------------------------------------
    # 2. brucon side: load ensemble, measure sigma & T_decorr
    # ------------------------------------------------------------------
    seeds = list(range(1000, 1030))
    print(f"\n[brucon] loading {len(seeds)} pwo seeds ...")
    surge_list = []
    sway_list = []
    t_grid = None
    for s in seeds:
        run_dir = work / f"pwo_seed{s:04d}"
        out = parse_output(run_dir / f"pwo_seed{s:04d}.out")
        if t_grid is None:
            t_grid = out["t"]
        surge_list.append(out["SurgeDev"])
        sway_list.append(out["SwayDev"])
    n_min = min(len(a) for a in surge_list)
    surge_arr = np.array([a[:n_min] for a in surge_list])
    sway_arr = np.array([a[:n_min] for a in sway_list])
    t_grid = t_grid[:n_min]

    # Use the same intact-stats window as run_comparison_waves_only.py:
    # 200 s ending 1 s before the failure event (failure_time_s = 560 s).
    failure_time_s = 60.0 + 500.0
    INTACT_S = 200.0
    intact_mask = (
        (t_grid >= failure_time_s - INTACT_S)
        & (t_grid < failure_time_s - 1.0)
    )
    t_intact = t_grid[intact_mask]
    surge_intact = surge_arr[:, intact_mask]
    sway_intact = sway_arr[:, intact_mask]
    n_samp = surge_intact.shape[1]
    win_dur = n_samp * SIM_DT
    print(f"  intact window: [{t_intact[0]:.1f}, {t_intact[-1]:.1f}] s "
          f"= {win_dur:.0f} s, {n_samp} samples")

    sigma_x_emp = float(np.median(surge_intact.std(axis=1)))
    sigma_y_emp = float(np.median(sway_intact.std(axis=1)))
    mean_x_emp = float(np.median(surge_intact.mean(axis=1)))
    mean_y_emp = float(np.median(sway_intact.mean(axis=1)))
    print(f"\n[brucon] sample stats (median across {len(seeds)} seeds):")
    print(f"  sigma_x (surge)       = {sigma_x_emp:.3f} m  (cqa: {prior.pos_sigma_x_m:.3f})")
    print(f"  sigma_y (sway)        = {sigma_y_emp:.3f} m  (cqa: {prior.pos_sigma_y_m:.3f})")
    print(f"  mean_x  (surge)       = {mean_x_emp:+.3f} m")
    print(f"  mean_y  (sway)        = {mean_y_emp:+.3f} m")

    # Per-seed T_decorr, then median.
    Tx_int = []
    Tx_e = []
    Ty_int = []
    Ty_e = []
    for k in range(len(seeds)):
        ti, te = empirical_decorr_time(surge_intact[k], SIM_DT)
        Tx_int.append(ti)
        Tx_e.append(te)
        ti, te = empirical_decorr_time(sway_intact[k], SIM_DT)
        Ty_int.append(ti)
        Ty_e.append(te)
    Tx_int = np.array(Tx_int)
    Tx_e = np.array(Tx_e)
    Ty_int = np.array(Ty_int)
    Ty_e = np.array(Ty_e)

    print(f"\n[brucon] empirical position decorrelation times "
          f"(median across seeds):")
    print(f"  surge: T_int (∫ρ to first zc) = {np.median(Tx_int):6.1f} s   "
          f"T_e (2 * 1/e width) = {np.median(Tx_e):6.1f} s")
    print(f"  sway : T_int (∫ρ to first zc) = {np.median(Ty_int):6.1f} s   "
          f"T_e (2 * 1/e width) = {np.median(Ty_e):6.1f} s")
    print(f"  cqa  pos_T_decorr_var_x_s = {prior.pos_T_decorr_var_x_s:.1f} s")
    print(f"  cqa  pos_T_decorr_var_y_s = {prior.pos_T_decorr_var_y_s:.1f} s")
    print("  NOTE: cqa T_decorr_var is the variance-estimator decorrelation")
    print("        time (used by Bayes σ updater); the running-max statistic")
    print("        depends on nu_0 (zero-up-crossing rate) and q (Vanmarcke).")

    # ------------------------------------------------------------------
    # 3. Predict running-max from each side's σ + brucon's empirical Tz
    # ------------------------------------------------------------------
    # For brucon: estimate nu_0 from the actual ensemble by counting
    # zero-up-crossings of (sway - mean_sway) across the intact window.
    nu0_y_list = []
    for k in range(len(seeds)):
        s = sway_intact[k] - sway_intact[k].mean()
        crossings = np.sum((s[:-1] < 0) & (s[1:] >= 0))
        nu0_y_list.append(crossings / win_dur)
    nu0_y_emp = float(np.median(nu0_y_list))
    Tz_y_emp = 1.0 / nu0_y_emp if nu0_y_emp > 0 else float("nan")
    print(f"\n[brucon] empirical sway zero-up-crossing rate:")
    print(f"  nu_0_y = {nu0_y_emp:.4f} Hz  (Tz_y ~ {Tz_y_emp:.1f} s)")
    print(f"  cqa    pos_nu0_max = {prior.pos_nu0_max:.4f} Hz  "
          f"(Tz ~ {1/prior.pos_nu0_max:.1f} s)")

    # CLH/Vanmarcke prediction: assume sway dominates radial (mean_x ~ 0,
    # sigma_x small relative to sigma_y at beam-on).
    print("\n[predict] running-max |pos| over 200 s, assuming sway-dominated:")
    print("  using Vanmarcke with q=cqa's q={:.3f}, T=200s".format(prior.pos_q))
    for label, sig, nu0 in [
        ("cqa σ_y, cqa nu0",        prior.pos_sigma_y_m, prior.pos_nu0_max),
        ("brucon σ_y, brucon nu0",  sigma_y_emp,         nu0_y_emp),
        ("brucon σ_y, cqa nu0",     sigma_y_emp,         prior.pos_nu0_max),
        ("cqa σ_y, brucon nu0",     prior.pos_sigma_y_m, nu0_y_emp),
    ]:
        p50 = predicted_runmax(sig, nu0, 200.0, q=prior.pos_q, p=0.5)
        p90 = predicted_runmax(sig, nu0, 200.0, q=prior.pos_q, p=0.9)
        print(f"  {label:32s}  P50={p50:.2f} m  P90={p90:.2f} m")

    # Empirical brucon ensemble running-max from the actual data.
    pos_intact = np.hypot(surge_intact, sway_intact)
    runmax = pos_intact.max(axis=1)
    print("\n[brucon] empirical running-max |pos| over 200 s window:")
    print(f"  P50 = {np.median(runmax):.2f} m")
    print(f"  P90 = {np.quantile(runmax, 0.9):.2f} m")
    print(f"  mean / std = {runmax.mean():.2f} / {runmax.std():.2f} m")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
- If brucon σ_y >> cqa σ_y (e.g. 0.6 vs 0.4 m): the gap is in σ. cqa is
  missing closed-loop variance. Look at the closed-loop transfer
  (controller gains, dynamics) -- maybe cqa's controller is too stiff.

- If brucon σ_y ≈ cqa σ_y but brucon nu_0 << cqa nu_0: the gap is in
  spectral shape. brucon has more low-frequency power for the same
  variance, so fewer crossings, larger characteristic peak.
  (predicted_runmax with brucon σ_y + brucon nu_0 should match the
  empirical brucon running-max; cqa σ_y + cqa nu_0 should match cqa
  pos_a_p50/p90.)

- If σ AND nu_0 match but T_decorr differs: the variance is correct but
  the spectral *bandwidth* (q) differs -- brucon has a narrower/clumpier
  spectrum, so the running-max sees fewer effective independent peaks.
""")


if __name__ == "__main__":
    main()
