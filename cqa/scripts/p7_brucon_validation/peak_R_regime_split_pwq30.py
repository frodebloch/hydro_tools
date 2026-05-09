"""Split per-seed peak |R| into two regimes and quantify the live-cell ceiling.

Driving question (from the regression study):
  - Regime A (LF-transient): peak |R| lands within ~15 s of WCF.
    Should be predictable from IC at t_WCF (cqa-27 deterministic forward).
  - Regime B (wave-realisation): peak |R| lands 20-60 s after WCF.
    Should be predictable only from post-WCF wave timing, which the live
    cell cannot observe.

This script:
  1. Loads the same 30 pwq30 seeds.
  2. Computes peak_R, t_peak_R, IC features, post-WCF wave features.
  3. Splits into A (t_peak_R < 15 s) and B (t_peak_R >= 15 s).
  4. For each subset, runs a SHORT-list regression (top-k features only,
     to avoid n ~ 15 overfit).
  5. Reports residual std (the irreducible component within each regime).
  6. Plots: peak_R distributions per regime, IC-fit residuals (A), wave-fit
     residuals (B).

Output: peak_R_regime_split_pwq30.png and printed table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_SPLIT = 15.0   # seconds after WCF: peaks earlier than this are regime A.


def _load_seed(seed):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = seed_dir / f"{TAG}_seed{seed:04d}_estimator.out"
    if main_p is None or not est_p.exists():
        return None
    with open(main_p) as f:
        hdr_m = f.readline().strip().split("\t")
    M = {h: data for h, data in zip(hdr_m, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: data for h, data in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    if M["t"][-1] < T_WCF + 60.0:
        return None
    return M, E


def _at(t_arr, val_arr, t_target):
    return float(np.interp(t_target, t_arr, val_arr))


def _ddt(arr, t_arr, t_target, h=0.5):
    return (np.interp(t_target + h, t_arr, arr)
            - np.interp(t_target - h, t_arr, arr)) / (2.0 * h)


def _build_features(M, E):
    t = M["t"]
    tE = E["Time"]
    t0 = T_WCF

    ic = dict(
        eta_x=_at(t, M["SurgeDev"], t0),
        eta_y=_at(t, M["SwayDev"], t0),
        eta_z=_at(t, M["HeadingDev"], t0),
        nu_x=_at(t, M["SurgeSpeed"], t0),
        nu_y=_at(t, M["SwaySpeed"], t0),
        nu_z=np.deg2rad(_at(t, M["RateOfTurn"], t0)) / 60.0,
        etaw_x=_at(t, M["xHf"], t0),
        etaw_y=_at(t, M["yHf"], t0),
        etaw_z=_at(t, M["headingHf"], t0),
        nuw_x=_ddt(M["xHf"], t, t0),
        nuw_y=_ddt(M["yHf"], t, t0),
        nuw_z=_ddt(M["headingHf"], t, t0),
        b_x=_at(tE, E["EstBiasSurge"], t0) * 1e3,
        b_y=_at(tE, E["EstBiasSway"], t0) * 1e3,
        b_z=_at(tE, E["EstBiasYaw"], t0) * 1e3,
    )

    # Truth peak in [T_WCF, T_WCF + 60] s, body-frame, baseline-corrected.
    post_mask = (t >= T_WCF) & (t <= T_WCF + 60.0)
    pre_mask = (t >= T_WCF - 30.0) & (t <= T_WCF - 5.0)
    h_pre = np.deg2rad(M["heading"][pre_mask])
    s_pre_mean = float((np.cos(h_pre) * M["x"][pre_mask] + np.sin(h_pre) * M["y"][pre_mask]).mean())
    w_pre_mean = float((-np.sin(h_pre) * M["x"][pre_mask] + np.cos(h_pre) * M["y"][pre_mask]).mean())
    h = np.deg2rad(M["heading"][post_mask])
    x = M["x"][post_mask]
    y = M["y"][post_mask]
    s_b = np.cos(h) * x + np.sin(h) * y - s_pre_mean
    w_b = -np.sin(h) * x + np.cos(h) * y - w_pre_mean
    R_truth = np.hypot(s_b, w_b)
    peak_R = float(R_truth.max())
    t_peak_R = float(t[post_mask][int(np.argmax(R_truth))] - T_WCF)

    # Post-WCF wave features in same 60 s window.
    xHf_w = M["xHf"][post_mask]
    yHf_w = M["yHf"][post_mask]
    R_wave = np.hypot(xHf_w, yHf_w)
    t_w_local = t[post_mask] - T_WCF
    pk = int(np.argmax(R_wave))

    wave = dict(
        rms_wave_x=float(np.sqrt(np.mean(xHf_w ** 2))),
        rms_wave_y=float(np.sqrt(np.mean(yHf_w ** 2))),
        peak_wave_R=float(R_wave.max()),
        t_peak_wave=float(t_w_local[pk]),
    )
    return ic, wave, peak_R, t_peak_R


def _standardise(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd_safe, mu, sd_safe


def _ridge_fit(Xs, ys, lam=1e-3):
    XtX = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    beta = np.linalg.solve(XtX, Xs.T @ ys)
    y_pred = Xs @ beta
    resid = ys - y_pred
    R2 = 1.0 - np.var(resid) / np.var(ys) if np.var(ys) > 0 else 0.0
    return beta, y_pred, R2


def _univariate_table(X_dict, y, label):
    print(f"\n--- {label}   univariate Pearson r (sorted) ---")
    rs = []
    for k, v in X_dict.items():
        v = np.asarray(v)
        if np.std(v) < 1e-12:
            r = 0.0
        else:
            r = float(np.corrcoef(v, y)[0, 1])
        rs.append((k, r))
    rs.sort(key=lambda t: -abs(t[1]))
    for k, r in rs[:8]:
        print(f"  {k:14s} r = {r:+.3f}   r^2 = {r*r:.3f}")
    return rs


def main():
    rows = []
    for s in SEEDS:
        out = _load_seed(s)
        if out is None:
            continue
        ic, wave, peak_R, t_peak_R = _build_features(*out)
        d = dict(seed=s, peak_R=peak_R, t_peak_R=t_peak_R)
        d.update(ic)
        d.update(wave)
        rows.append(d)

    print(f"Loaded {len(rows)} seeds")
    n = len(rows)
    y = np.array([r["peak_R"] for r in rows])
    t_pk = np.array([r["t_peak_R"] for r in rows])

    print(f"\nFull sample: peak_R mean = {y.mean():.3f}  std = {y.std():.3f}")
    print(f"             t_peak_R mean = {t_pk.mean():.2f} s  std = {t_pk.std():.2f} s")

    mask_A = t_pk < T_SPLIT
    mask_B = ~mask_A
    nA, nB = int(mask_A.sum()), int(mask_B.sum())
    yA, yB = y[mask_A], y[mask_B]
    print(f"\nRegime A (t_peak_R < {T_SPLIT:.0f} s): n = {nA}")
    print(f"  peak_R mean = {yA.mean():.3f}  std = {yA.std():.3f}  range = [{yA.min():.2f}, {yA.max():.2f}]")
    print(f"Regime B (t_peak_R >= {T_SPLIT:.0f} s): n = {nB}")
    print(f"  peak_R mean = {yB.mean():.3f}  std = {yB.std():.3f}  range = [{yB.min():.2f}, {yB.max():.2f}]")

    # IC features and wave features.
    ic_keys = ["eta_x", "eta_y", "eta_z", "nu_x", "nu_y", "nu_z",
               "etaw_x", "etaw_y", "etaw_z", "nuw_x", "nuw_y", "nuw_z",
               "b_x", "b_y", "b_z"]
    wave_keys = ["rms_wave_x", "rms_wave_y", "peak_wave_R", "t_peak_wave"]

    def slice_dict(rows_subset, keys):
        return {k: np.array([r[k] for r in rows_subset]) for k in keys}

    rows_A = [r for r, m in zip(rows, mask_A) if m]
    rows_B = [r for r, m in zip(rows, mask_B) if m]

    ic_A = slice_dict(rows_A, ic_keys)
    wv_A = slice_dict(rows_A, wave_keys)
    ic_B = slice_dict(rows_B, ic_keys)
    wv_B = slice_dict(rows_B, wave_keys)

    rs_ic_A = _univariate_table(ic_A, yA, f"Regime A IC features (n={nA})")
    rs_wv_A = _univariate_table(wv_A, yA, f"Regime A wave features (n={nA})")
    rs_ic_B = _univariate_table(ic_B, yB, f"Regime B IC features (n={nB})")
    rs_wv_B = _univariate_table(wv_B, yB, f"Regime B wave features (n={nB})")

    # Short-list regression per regime: top-3 IC + top-2 wave features by |r|.
    def _topk(rs, k):
        return [name for name, _ in rs[:k]]

    feat_A = _topk(rs_ic_A, 3) + _topk(rs_wv_A, 2)
    feat_B = _topk(rs_ic_B, 3) + _topk(rs_wv_B, 2)
    print(f"\nRegime A short-list features: {feat_A}")
    print(f"Regime B short-list features: {feat_B}")

    XA = np.array([[r[k] for k in feat_A] for r in rows_A])
    XB = np.array([[r[k] for k in feat_B] for r in rows_B])
    XAs, *_ = _standardise(XA)
    XBs, *_ = _standardise(XB)
    yAs = (yA - yA.mean()) / max(yA.std(), 1e-12)
    yBs = (yB - yB.mean()) / max(yB.std(), 1e-12)

    betaA, ypA, R2A = _ridge_fit(XAs, yAs)
    betaB, ypB, R2B = _ridge_fit(XBs, yBs)

    # Convert prediction back to physical units for residual std.
    ypA_phys = ypA * yA.std() + yA.mean()
    ypB_phys = ypB * yB.std() + yB.mean()
    res_A = yA - ypA_phys
    res_B = yB - ypB_phys

    print(f"\n--- Regime A short-list regression: R^2 = {R2A:.3f} ---")
    for k, b in zip(feat_A, betaA):
        print(f"  {k:14s} beta_std = {b:+.3f}")
    print(f"  residual std = {res_A.std():.3f} m  (over n={nA} seeds)")

    print(f"\n--- Regime B short-list regression: R^2 = {R2B:.3f} ---")
    for k, b in zip(feat_B, betaB):
        print(f"  {k:14s} beta_std = {b:+.3f}")
    print(f"  residual std = {res_B.std():.3f} m  (over n={nB} seeds)")

    # Live-cell ceiling interpretation:
    #   Regime A residual std = irreducible noise within IC-explained regime.
    #   Regime B residual std = irreducible noise within wave-explained regime.
    #   The live cell can ONLY use IC. So its prediction error on regime-B
    #   peaks is at minimum yB.std() (no IC predictor helps), or more
    #   precisely the std of yB conditional on IC.
    #
    # Compute: live-cell-equivalent error = std of yB after regressing yB on IC only.
    XB_ic_full = np.array([[r[k] for k in ic_keys] for r in rows_B])
    if XB_ic_full.shape[0] >= 5:
        XBic_s, *_ = _standardise(XB_ic_full)
        # Use top-3 IC features by |r| in regime B to avoid overfit.
        feat_ic_B = _topk(rs_ic_B, 3)
        XB_top = np.array([[r[k] for k in feat_ic_B] for r in rows_B])
        XB_top_s, *_ = _standardise(XB_top)
        beta_ic_B, ypic_B, R2_ic_B = _ridge_fit(XB_top_s, yBs)
        res_ic_B = yB - (ypic_B * yB.std() + yB.mean())
        print(f"\n--- Live-cell equivalent (Regime B, IC-only top3): R^2 = {R2_ic_B:.3f} ---")
        print(f"  residual std = {res_ic_B.std():.3f} m  (= live-cell prediction floor on regime B)")
        ceiling = res_ic_B.std()
    else:
        ceiling = float("nan")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(t_pk[mask_A], y[mask_A], color="C0", label=f"Regime A (n={nA})")
    ax.scatter(t_pk[mask_B], y[mask_B], color="C3", label=f"Regime B (n={nB})")
    ax.axvline(T_SPLIT, color="k", lw=0.5, ls=":")
    ax.set_xlabel("t_peak_R [s after WCF]")
    ax.set_ylabel("peak_R [m]")
    ax.set_title("Regime split by peak timing")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(yA, ypA_phys, color="C0", label=f"A: R^2 = {R2A:.2f}")
    ax.scatter(yB, ypB_phys, color="C3", label=f"B: R^2 = {R2B:.2f}")
    lim = max(y.max(), max(ypA_phys.max(), ypB_phys.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.5)
    ax.set_xlabel("realised peak_R [m]")
    ax.set_ylabel("short-list prediction [m]")
    ax.set_title("Per-regime regression")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(res_A, bins=8, alpha=0.7, color="C0", label=f"A residual (std={res_A.std():.2f} m)")
    ax.hist(res_B, bins=8, alpha=0.7, color="C3", label=f"B residual (std={res_B.std():.2f} m)")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("residual peak_R [m]")
    ax.set_ylabel("count")
    ax.set_title("Per-regime regression residuals")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(["A: short-list",
            "B: short-list",
            "B: IC-only (live-cell floor)"],
           [res_A.std(),
            res_B.std(),
            ceiling],
           color=["C0", "C3", "C5"])
    ax.axhline(y.std(), color="k", lw=0.5, ls="--", label=f"full-sample std = {y.std():.2f}")
    ax.set_ylabel("residual std [m]")
    ax.set_title("Per-regime irreducible component")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f"Per-seed peak |R| regime split at {TAG} (n={n})", fontsize=12)
    plt.tight_layout()
    out = THIS / "peak_R_regime_split_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
