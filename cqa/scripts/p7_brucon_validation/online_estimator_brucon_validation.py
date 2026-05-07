"""G1: brucon-ensemble end-to-end validation of BayesianSigmaEstimator.

Goal
----
Close the Option-B loop end-to-end on simulator-grade data: take the
existing 30-seed ``pwo_lockedTp`` brucon ensemble (HS=4.20 m,
Tp=10.22 s, β=90°), feed each seed's body-frame total position
(x_body, y_body) — the operationally-measurable signal — through two
``cqa.online_estimator.BayesianSigmaEstimator`` instances (one per
axis), and check that the InvGamma posterior recovers brucon's
empirical std within the noise implied by the Bartlett effective
sample size.

This is gap **G1** from analysis.md §12.21.3. It is the foundation
for the live operator nowcast: until we know the existing 1182-line
estimator stack actually recovers σ from real time-domain data, we
cannot wire the posterior into ``summarise_intact_prior`` for live
operations.

Channels and conventions
------------------------
brucon writes ``(x, y, heading)`` as **total NED CG state** (LF + WF;
see §12.20.15, §12.21.2). Body-frame projection follows
``long_run_locked_tp_validation.get_brucon_sway_lf``:

    x_body =  cos(ψ) · x_ned + sin(ψ) · y_ned
    y_body = -sin(ψ) · x_ned + cos(ψ) · y_ned

The brucon truth at this sea state (validated across 30 seeds in
the late window [1500, 3000] s, §12.20.10):

    σ_y_body_total ≈ 0.893 m
    σ_y_body_LF    ≈ 0.691 m   (= std(y_body − yHf))
    σ_yHf          ≈ 0.536 m

We validate σ_y_body_total because (a) it is what the operator
measures from GPS/INS, (b) it feeds the IMCA M254 Fig 8 footprint,
(c) it exercises the full LF+WF model prior built via
``cqa.observer.total_position_psd``.

Estimator setup per axis
------------------------
* ``prior_sigma2``     : σ_x_model² or σ_y_model² from
                        ``total_position_psd`` integrated over ω, at
                        the same (HS, Tp, β). This is the +13 %
                        biased model prior of §12.20.15. Choosing
                        the model prior (not the brucon truth) lets
                        the A5 health diagnostic fire WARMING when
                        the data disagrees with the prior — exactly
                        what A5 is designed to catch.
* ``T_decorr_s``       : variance-estimator decorrelation time
                        ``T_var = π · ∫ S_X(ω)² dω / m₀²`` from the
                        same model PSD via
                        ``variance_decorrelation_time_from_psd``.
                        NOT the legacy ``closed_loop_decorrelation_time``
                        (the docstring at online_estimator.py:78-98
                        cites a 5× error on the canonical CSOV).
* ``window_s``         : 1500 s (the §12.20 late window).
* ``dt_s``             : 0.1 s (10 Hz brucon output cadence).
* ``prior_strength_n0``: 2.0 (default; weak — data dominates by
                        the end of the window).
* ``assume_zero_mean`` : True; we subtract the per-window mean
                        before pushing samples (the DP regulates
                        to zero offset, but the late window mean
                        is not exactly zero due to drift bias).

Acceptance
----------
Per seed: posterior σ_median should recover empirical std(window)
to within ~3 %. The 90 % credible interval should cover the
empirical std at ≥85 % of seeds (binomial coverage check; with
30 seeds and true coverage 0.90, the 5th percentile of the binomial
is ⌊30·0.85⌋ ≈ 25, so we require ≥25/30).

Health ladder expectations:
  A1 (stationarity)     -- clean (late window is settled)
  A2 (zero-mean)        -- clean (window-mean subtracted)
  A3 (Gaussian)         -- clean (linear vessel response to Gaussian wave forcing)
  A4 (n_eff warm)       -- warm (n_eff ≈ window_s / T_var)
  A5 (prior in CI)      -- WARMING (model prior is +13 % biased)

A5 firing WARMING is the **expected** correct behaviour: the
posterior is data-driven and the model prior is conservative, so
the data sigma should sit below the prior credible interval.

Output
------
* console table per seed: posterior σ_x, σ_y, brucon truth σ_x,
  σ_y, error %, A1-A5 verdicts.
* ensemble summary: median rel error per axis, coverage rate,
  health ladder histogram.
* PNG: scatter (posterior σ vs brucon truth) per axis with 90 % CI
  bars, plus a histogram of A1-A5 levels across the ensemble.
  Saved to the script directory (gitignored).

Companion test in ``tests/test_online_estimator_brucon.py`` runs
seed 1000 only and asserts the per-seed acceptance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))                       # for `harness`
sys.path.insert(0, str(THIS.parent.parent))         # for `cqa` package

from harness import parse_output  # noqa: E402

# ---- ensemble + sea state (matches long_run_locked_tp_validation.py) ----
ENSEMBLE_DIR = THIS / "work_long_lockedTp"
TAG = "pwo_lockedTp"
N_SEEDS = 30
SEED_FIRST = 1000
HS = 4.19571865443425
TP = 10.22443464601827
THETA_REL = np.pi / 2          # β = 90° (beam-on)
WINDOW = (1500.0, 3000.0)
DT_S = 0.1                     # 10 Hz brucon output cadence

# ---- estimator config ----
PRIOR_N0 = 2.0
CREDIBLE = 0.90

# ---- pdstrip RAO file (only needed if we have a copy on this host) ----
PDSTRIP_PATH = Path("/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat")


def project_body_frame(result, t_start: float, t_end: float):
    """Project brucon (x_ned, y_ned, heading) into body frame and slice to window."""
    t = result.columns["t"]
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        raise RuntimeError(f"window {t_start}..{t_end} contains no samples")
    x_ned = result.columns["x"][mask]
    y_ned = result.columns["y"][mask]
    psi = np.deg2rad(result.columns["heading"][mask])
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    x_body = cos_p * x_ned + sin_p * y_ned
    y_body = -sin_p * x_ned + cos_p * y_ned
    return t[mask], x_body, y_body


def build_model_prior(hs: float, tp: float, theta_rel: float, omega: np.ndarray):
    """Compute (σ²_x_model, σ²_y_model, T_var_x, T_var_y) for the body-frame
    CG total position channels at (Hs, Tp, theta_rel).

    Uses the full LF+WF observer-augmented pipeline of §12.20.14:
    cqa.observer.total_position_psd with
    cqa.drift.slow_drift_force_psd_newman_pdstrip on the LF channel
    and pdstrip RAO × wave_elevation_psd on the WF channel.

    Returns None if the pdstrip RAO file isn't available — caller
    should fall back to an empirical prior derived from a small
    bootstrap subset of seeds.
    """
    if not PDSTRIP_PATH.exists():
        return None

    from cqa.config import csov_default_config
    from cqa.controller import LinearDpController
    from cqa.vessel import LinearVesselModel
    from cqa.observer import build_observer_augmented_system, total_position_psd
    from cqa.rao import load_pdstrip_rao, evaluate_rao
    from cqa.psd import wave_elevation_psd
    from cqa.drift import slow_drift_force_psd_newman_pdstrip
    from cqa.wave_response import cqa_theta_rel_to_pdstrip_beta_deg
    from cqa.extreme_value import variance_decorrelation_time_from_psd
    from scipy.integrate import trapezoid

    cfg = csov_default_config()
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    rao = load_pdstrip_rao(str(PDSTRIP_PATH))
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao, Hs=hs, Tp=tp, theta_wave_rel=theta_rel,
    )
    beta_deg = cqa_theta_rel_to_pdstrip_beta_deg(theta_rel)

    def S_eta_w(w):
        H = evaluate_rao(rao, np.array([w]), beta_deg)[0]
        H_pos = np.array([H[0], H[1], H[5]])
        S_eta_scalar = wave_elevation_psd(np.array([w]), hs, tp)[0]
        return np.diag(np.abs(H_pos) ** 2 * S_eta_scalar)

    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=tp, T_thr=2.0, include_integrator=True,
    )
    S_y = total_position_psd(aug, [S_drift], S_eta_w, omega)
    S_xx = S_y[:, 0, 0].real
    S_yy = S_y[:, 1, 1].real
    sigma2_x = float(trapezoid(S_xx, omega))
    sigma2_y = float(trapezoid(S_yy, omega))
    T_var_x = float(variance_decorrelation_time_from_psd(S_xx, omega))
    T_var_y = float(variance_decorrelation_time_from_psd(S_yy, omega))
    return sigma2_x, sigma2_y, T_var_x, T_var_y


def stream_through_estimator(
    samples: np.ndarray,
    prior_sigma2: float,
    T_decorr_s: float,
    window_s: float,
    dt_s: float,
    prior_strength_n0: float = PRIOR_N0,
    credible: float = CREDIBLE,
):
    """Construct a fresh BayesianSigmaEstimator, stream the (already
    mean-subtracted) samples through ``update``, return (posterior, health)
    after the final sample."""
    from cqa.online_estimator import BayesianSigmaEstimator
    est = BayesianSigmaEstimator(
        prior_sigma2=prior_sigma2,
        T_decorr_s=T_decorr_s,
        dt_s=dt_s,
        prior_strength_n0=prior_strength_n0,
        window_s=window_s,
        assume_zero_mean=True,
    )
    for x in samples:
        est.update(float(x))
    post = est.posterior(credible=credible)
    health = est.health(credible=credible)
    return post, health


def evaluate_seed(
    seed: int,
    prior_sigma2_x: float,
    prior_sigma2_y: float,
    T_decorr_x_s: float,
    T_decorr_y_s: float,
):
    """Run the per-seed validation. Returns a dict of results, or None if
    the seed directory is missing."""
    seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{seed:04d}"
    out_path = seed_dir / f"{TAG}_seed{seed:04d}.out"
    if not out_path.exists():
        return None
    result = parse_output(out_path)
    _, x_body, y_body = project_body_frame(result, *WINDOW)
    # Subtract per-window mean (the DP regulates to zero, but at the
    # end of a 1500 s window there is residual drift bias).
    x_centred = x_body - x_body.mean()
    y_centred = y_body - y_body.mean()
    truth_x = float(x_centred.std())
    truth_y = float(y_centred.std())

    # Run the InvGamma estimator on each axis. We use the FULL late
    # window as the sliding window so the posterior is computed on the
    # same samples used to compute brucon truth — this is a clean
    # apples-to-apples comparison.
    window_s = WINDOW[1] - WINDOW[0]
    post_x, health_x = stream_through_estimator(
        x_centred, prior_sigma2_x, T_decorr_x_s, window_s, DT_S,
    )
    post_y, health_y = stream_through_estimator(
        y_centred, prior_sigma2_y, T_decorr_y_s, window_s, DT_S,
    )

    # Radial composition.
    from cqa.online_estimator import combine_radial_posterior
    rng = np.random.default_rng(seed)
    radial = combine_radial_posterior(
        post_x, post_y, credible=CREDIBLE, n_mc=4000, rng=rng,
        sample_mean_x=float(x_body.mean()),
        sample_mean_y=float(y_body.mean()),
    )
    truth_R = float(np.sqrt(truth_x ** 2 + truth_y ** 2))

    return {
        "seed": seed,
        "truth_x": truth_x,
        "truth_y": truth_y,
        "truth_R": truth_R,
        "post_x": post_x,
        "post_y": post_y,
        "radial": radial,
        "health_x": health_x,
        "health_y": health_y,
    }


def covered(post, truth: float) -> bool:
    """Is the empirical truth inside the posterior 90 % CI on σ?"""
    return post.sigma_lo <= truth <= post.sigma_hi


def health_summary(health) -> str:
    """Compose a per-channel ValidityBadge from a PosteriorHealth."""
    from cqa.online_estimator import compose_validity_badge
    badge = compose_validity_badge(health)
    return badge.level


def main() -> None:
    print("=" * 78)
    print("G1: BayesianSigmaEstimator brucon-ensemble validation")
    print(f"  ensemble: {ENSEMBLE_DIR}")
    print(f"  seeds: {SEED_FIRST}..{SEED_FIRST + N_SEEDS - 1}")
    print(f"  window: {WINDOW} s, dt = {DT_S} s")
    print(f"  sea state: HS = {HS:.4f} m, Tp = {TP:.4f} s, β = 90°")
    print("=" * 78)

    if not ENSEMBLE_DIR.exists():
        print(f"\n  ENSEMBLE NOT FOUND at {ENSEMBLE_DIR}")
        print("  Run long_run_locked_tp_validation.py first to populate it.")
        sys.exit(1)

    # ---- model prior ----
    print("\n--- building model prior σ² and T_var via total_position_psd ---")
    omega = np.unique(np.concatenate([
        np.logspace(-4, -1, 256),
        np.linspace(0.1, 2.5, 1024),
    ]))
    prior = build_model_prior(HS, TP, THETA_REL, omega)
    if prior is None:
        print(f"  pdstrip RAO not available at {PDSTRIP_PATH}")
        print("  falling back to bootstrap prior from seed 1000.")
        # Bootstrap: use seed 1000 std as a 1.2× scaled prior (so the
        # prior is mildly mis-calibrated and A5 still has something
        # to flag).
        seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{SEED_FIRST:04d}"
        result = parse_output(seed_dir / f"{TAG}_seed{SEED_FIRST:04d}.out")
        _, x_body, y_body = project_body_frame(result, *WINDOW)
        sigma2_x = (x_body.std() * 1.2) ** 2
        sigma2_y = (y_body.std() * 1.2) ** 2
        # Heuristic decorrelation time: dominant peak at ~Tp ⇒ T_var ~
        # (Tp / 2π) · q^(-1) · π ~ a few × Tp; conservative 100 s.
        T_var_x = T_var_y = 100.0
        prior_source = "BOOTSTRAP (seed 1000 × 1.2)"
    else:
        sigma2_x, sigma2_y, T_var_x, T_var_y = prior
        prior_source = "MODEL (cqa.observer.total_position_psd)"

    print(f"  source: {prior_source}")
    print(f"  prior σ_x = {np.sqrt(sigma2_x):.3f} m, T_var_x = {T_var_x:.1f} s")
    print(f"  prior σ_y = {np.sqrt(sigma2_y):.3f} m, T_var_y = {T_var_y:.1f} s")

    # ---- per-seed ----
    print("\n--- per-seed posterior recovery ---")
    print(f"  {'seed':>5} {'truth_x':>8} {'σ_x_post':>9} {'err%':>7} "
          f"{'cov_x':>5} | {'truth_y':>8} {'σ_y_post':>9} {'err%':>7} "
          f"{'cov_y':>5} | {'A_x':>9} {'A_y':>9}")
    rows = []
    t0 = time.time()
    for k in range(N_SEEDS):
        seed = SEED_FIRST + k
        r = evaluate_seed(seed, sigma2_x, sigma2_y, T_var_x, T_var_y)
        if r is None:
            print(f"  {seed:>5} (missing)")
            continue
        cov_x = covered(r["post_x"], r["truth_x"])
        cov_y = covered(r["post_y"], r["truth_y"])
        err_x = (r["post_x"].sigma_median - r["truth_x"]) / r["truth_x"] * 100
        err_y = (r["post_y"].sigma_median - r["truth_y"]) / r["truth_y"] * 100
        ax = health_summary(r["health_x"])
        ay = health_summary(r["health_y"])
        print(
            f"  {seed:>5} {r['truth_x']:>8.3f} {r['post_x'].sigma_median:>9.3f} "
            f"{err_x:>+7.1f} {'Y' if cov_x else 'N':>5} | "
            f"{r['truth_y']:>8.3f} {r['post_y'].sigma_median:>9.3f} "
            f"{err_y:>+7.1f} {'Y' if cov_y else 'N':>5} | "
            f"{ax:>9} {ay:>9}"
        )
        rows.append({
            "seed": seed, "truth_x": r["truth_x"], "truth_y": r["truth_y"],
            "truth_R": r["truth_R"],
            "post_x_med": r["post_x"].sigma_median,
            "post_x_lo": r["post_x"].sigma_lo,
            "post_x_hi": r["post_x"].sigma_hi,
            "post_y_med": r["post_y"].sigma_median,
            "post_y_lo": r["post_y"].sigma_lo,
            "post_y_hi": r["post_y"].sigma_hi,
            "post_R_med": r["radial"].sigma_R_median,
            "post_R_lo": r["radial"].sigma_R_lo,
            "post_R_hi": r["radial"].sigma_R_hi,
            "cov_x": cov_x, "cov_y": cov_y,
            "n_eff_x": r["health_x"].n_eff,
            "n_eff_y": r["health_y"].n_eff,
            "badge_x": ax, "badge_y": ay,
        })
    print(f"  ({time.time() - t0:.1f} s)")

    if not rows:
        print("\n  NO ROWS PROCESSED")
        sys.exit(1)

    # ---- aggregate ----
    truth_x = np.array([r["truth_x"] for r in rows])
    truth_y = np.array([r["truth_y"] for r in rows])
    truth_R = np.array([r["truth_R"] for r in rows])
    post_x_med = np.array([r["post_x_med"] for r in rows])
    post_y_med = np.array([r["post_y_med"] for r in rows])
    post_R_med = np.array([r["post_R_med"] for r in rows])
    err_x_pct = (post_x_med - truth_x) / truth_x * 100
    err_y_pct = (post_y_med - truth_y) / truth_y * 100
    err_R_pct = (post_R_med - truth_R) / truth_R * 100
    cov_x = sum(r["cov_x"] for r in rows)
    cov_y = sum(r["cov_y"] for r in rows)
    n = len(rows)

    print("\n--- ensemble summary ---")
    print(f"  brucon truth     : σ_x = {np.median(truth_x):.3f} m   σ_y = {np.median(truth_y):.3f} m   σ_R = {np.median(truth_R):.3f} m")
    print(f"  posterior median : σ_x = {np.median(post_x_med):.3f} m   σ_y = {np.median(post_y_med):.3f} m   σ_R = {np.median(post_R_med):.3f} m")
    print(f"  median error (%) : x = {np.median(err_x_pct):+.2f}   y = {np.median(err_y_pct):+.2f}   R = {np.median(err_R_pct):+.2f}")
    print(f"  90% coverage     : x = {cov_x}/{n} = {cov_x / n * 100:.0f}%   y = {cov_y}/{n} = {cov_y / n * 100:.0f}%   (target ≥ 85%)")
    print(f"  n_eff (median)   : x = {np.median([r['n_eff_x'] for r in rows]):.1f}   y = {np.median([r['n_eff_y'] for r in rows]):.1f}")

    # Health ladder.
    from collections import Counter
    bx = Counter(r["badge_x"] for r in rows)
    by = Counter(r["badge_y"] for r in rows)
    print(f"  badge x          : " + ", ".join(f"{k}={v}" for k, v in sorted(bx.items())))
    print(f"  badge y          : " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, truth, post_med, post_lo, post_hi, label in [
            (axes[0], truth_x, post_x_med,
             np.array([r["post_x_lo"] for r in rows]),
             np.array([r["post_x_hi"] for r in rows]), "σ_x"),
            (axes[1], truth_y, post_y_med,
             np.array([r["post_y_lo"] for r in rows]),
             np.array([r["post_y_hi"] for r in rows]), "σ_y"),
            (axes[2], truth_R, post_R_med,
             np.array([r["post_R_lo"] for r in rows]),
             np.array([r["post_R_hi"] for r in rows]), "σ_R"),
        ]:
            yerr = np.stack([post_med - post_lo, post_hi - post_med])
            ax.errorbar(truth, post_med, yerr=yerr, fmt="o", alpha=0.7,
                        capsize=2, label="posterior median (90% CI)")
            lim = [min(truth.min(), post_lo.min()) * 0.95,
                   max(truth.max(), post_hi.max()) * 1.05]
            ax.plot(lim, lim, "k--", alpha=0.5, label="y=x (perfect recovery)")
            # Prior σ for reference.
            if label == "σ_x":
                ax.axhline(np.sqrt(sigma2_x), color="C2", ls=":", alpha=0.7,
                           label=f"model prior σ = {np.sqrt(sigma2_x):.2f}")
            elif label == "σ_y":
                ax.axhline(np.sqrt(sigma2_y), color="C2", ls=":", alpha=0.7,
                           label=f"model prior σ = {np.sqrt(sigma2_y):.2f}")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"brucon truth {label} [m]")
            ax.set_ylabel(f"posterior {label} [m]")
            ax.set_title(f"{label} recovery (n = {n} seeds)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(
            f"G1: BayesianSigmaEstimator vs brucon truth, "
            f"HS={HS:.2f} m, Tp={TP:.2f} s, β=90°, late window {WINDOW} s",
            y=1.02
        )
        fig.tight_layout()
        out_png = THIS / "online_estimator_brucon_validation.png"
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        print(f"\nSaved: {out_png}")
    except Exception as e:
        print(f"\nplot failed: {e}")


if __name__ == "__main__":
    main()
