"""What drives the per-seed variation in the post-WCF excursion at pwq30?

User question (paraphrased): The validation script showed a 3x spread
in realised peak |R| across 30 seeds (1.1 m to 3.3 m). Is this
variation first-order in the initial-condition variables (eta_hat,
nu_hat, eta_wave, b_hat at t=t_WCF) and/or in the post-WCF wave
realisation? Are these drivers correlated across seeds?

If the variation IS first-order in IC + post-WCF waves, then a linear
regression of peak |R| against those candidate drivers should explain
most of the variance. If it's NOT, then there's something nonlinear
(saturation, allocator topology change, etc.) that the linear cqa-27
model misses.

This is also relevant to scenario design: if peak |R| is dominated by
initial conditions at t=t_WCF, then the live cell's correctness rests
on getting eta_hat(0), nu_hat(0), eta_wave(0), b_hat correct -- which
the live observer does. If post-WCF wave realisation is also
first-order, the live cell will systematically under-predict by the
fraction of variance attributable to the future waves (which it has
no access to without a wave radar).

Candidate first-order drivers per seed at t = t_WCF:

  ic_eta    : (eta_hat_x, eta_hat_y, eta_hat_yaw)              [m, m, rad]
  ic_nu     : (nu_hat_x, nu_hat_y, nu_hat_yaw)                 [m/s, ..., rad/s]
  ic_etaw   : (xHf, yHf, headingHf)                            [m, m, rad]
  ic_bhat   : (b_hat_surge, b_hat_sway, b_hat_yaw)             [N, N, Nm]

Post-WCF integral drivers (computed from the brucon log):

  rms_wave_x : RMS wave-frequency surge motion in [t_WCF, t_WCF+30] s
  rms_wave_y : same for sway
  peak_wave_R: peak |R| in the WF channel over [t_WCF, t_WCF+30]

Regression target: per-seed truth radial peak |R(t)| in
[t_WCF, t_WCF+60].

Output: linear regression beta-coefficients (standardised) showing
which drivers contribute the most variance. Plus a correlation matrix
across drivers to show which are coupled.
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
DT = 0.1


def _load_seed(seed: int):
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


def main():
    rows = []
    for s in SEEDS:
        out = _load_seed(s)
        if out is None:
            continue
        M, E = out
        t = M["t"]
        tE = E["Time"]

        # Initial conditions at t = T_WCF (use t = T_WCF - 0.1 to be just-pre).
        t0 = T_WCF
        eta_x = _at(t, M["SurgeDev"], t0)
        eta_y = _at(t, M["SwayDev"], t0)
        eta_z = _at(t, M["HeadingDev"], t0)
        nu_x = _at(t, M["SurgeSpeed"], t0)
        nu_y = _at(t, M["SwaySpeed"], t0)
        nu_z = np.deg2rad(_at(t, M["RateOfTurn"], t0)) / 60.0
        etaw_x = _at(t, M["xHf"], t0)
        etaw_y = _at(t, M["yHf"], t0)
        etaw_z = _at(t, M["headingHf"], t0)
        # WF velocity at t_WCF: numerical central difference on the WF channels.
        # Pairs with etaw_* (position) to fully specify the WF oscillator phase.
        def _ddt(arr, t_arr, t_target, h=0.5):
            return (np.interp(t_target + h, t_arr, arr)
                    - np.interp(t_target - h, t_arr, arr)) / (2.0 * h)
        nuw_x = _ddt(M["xHf"], t, t0)
        nuw_y = _ddt(M["yHf"], t, t0)
        nuw_z = _ddt(M["headingHf"], t, t0)
        b_x = _at(tE, E["EstBiasSurge"], t0) * 1e3   # N
        b_y = _at(tE, E["EstBiasSway"], t0) * 1e3
        b_z = _at(tE, E["EstBiasYaw"], t0) * 1e3     # Nm

        # Post-WCF realised body-frame peak |R| in [t_WCF, t_WCF+60].
        post_mask = (t >= T_WCF) & (t <= T_WCF + 60.0)
        h = np.deg2rad(M["heading"][post_mask])
        x = M["x"][post_mask]
        y = M["y"][post_mask]
        # Subtract pre-WCF mean (consistent with the validation script)
        pre_mask = (t >= T_WCF - 30.0) & (t <= T_WCF - 5.0)
        h_pre = np.deg2rad(M["heading"][pre_mask])
        s_b_pre = np.cos(h_pre) * M["x"][pre_mask] + np.sin(h_pre) * M["y"][pre_mask]
        w_b_pre = -np.sin(h_pre) * M["x"][pre_mask] + np.cos(h_pre) * M["y"][pre_mask]
        s_pre_mean = float(s_b_pre.mean())
        w_pre_mean = float(w_b_pre.mean())
        s_b = np.cos(h) * x + np.sin(h) * y - s_pre_mean
        w_b = -np.sin(h) * x + np.cos(h) * y - w_pre_mean
        R_truth = np.hypot(s_b, w_b)
        peak_R = float(R_truth.max())
        t_peak_R = float(t[post_mask][int(np.argmax(R_truth))] - T_WCF)

        # Post-WCF wave-frequency drivers.
        # Use 60 s window to match the target window.
        wave_mask = (t >= T_WCF) & (t <= T_WCF + 60.0)
        t_w = t[wave_mask]
        xHf_w = M["xHf"][wave_mask]
        yHf_w = M["yHf"][wave_mask]
        rms_wave_x = float(np.sqrt(np.mean(xHf_w ** 2)))
        rms_wave_y = float(np.sqrt(np.mean(yHf_w ** 2)))
        R_wave = np.hypot(xHf_w, yHf_w)
        peak_wave_R = float(R_wave.max())
        idx_pk = int(np.argmax(R_wave))
        t_peak_wave = float(t_w[idx_pk] - T_WCF)  # seconds after WCF
        # Direction of WF excursion at the peak, body frame, in env-relative terms.
        # At pwq30, env_rel = +30 deg (env from forward-starboard).
        # Recovery direction (LF response) is roughly along the env-rel direction.
        # Project wave-peak vector onto unit env-rel direction:
        env_rel_rad = np.deg2rad(30.0)
        u_env = np.array([np.cos(env_rel_rad), np.sin(env_rel_rad)])
        wave_proj_env = float(xHf_w[idx_pk] * u_env[0] + yHf_w[idx_pk] * u_env[1])
        # Time-of-truth-peak |R(t)| within the 60s window (for diagnostic, not feature).
        # (We will plot this in the timing diagnostic.)

        rows.append(dict(
            seed=s,
            eta_x=eta_x, eta_y=eta_y, eta_z=eta_z,
            nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
            etaw_x=etaw_x, etaw_y=etaw_y, etaw_z=etaw_z,
            nuw_x=nuw_x, nuw_y=nuw_y, nuw_z=nuw_z,
            b_x=b_x, b_y=b_y, b_z=b_z,
            rms_wave_x=rms_wave_x, rms_wave_y=rms_wave_y,
            peak_wave_R=peak_wave_R,
            t_peak_wave=t_peak_wave,
            wave_proj_env=wave_proj_env,
            peak_R=peak_R,
            t_peak_R=t_peak_R,
        ))

    print(f"Loaded {len(rows)} seeds")
    if not rows:
        sys.exit("no data")

    keys = list(rows[0].keys())
    keys.remove("seed")
    n = len(rows)

    # Build matrix of features and target.
    # t_peak_R is a diagnostic of the target, not a predictor; exclude it.
    X_keys = [k for k in keys if k not in ("peak_R", "t_peak_R")]
    X = np.array([[r[k] for k in X_keys] for r in rows])
    y = np.array([r["peak_R"] for r in rows])
    t_peak_R_arr = np.array([r["t_peak_R"] for r in rows])
    t_peak_wave_arr = np.array([r["t_peak_wave"] for r in rows])

    print(f"\nTarget : peak_R [m]   mean = {y.mean():.3f}   std = {y.std():.3f}   "
          f"range = [{y.min():.3f}, {y.max():.3f}]")

    # Pearson correlations of each feature with target (univariate).
    print(f"\n--- univariate Pearson r with peak_R (sorted by |r|) ---")
    rs = []
    for j, k in enumerate(X_keys):
        x = X[:, j]
        if np.std(x) < 1e-12:
            r = 0.0
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rs.append((k, r))
    rs.sort(key=lambda t: -abs(t[1]))
    for k, r in rs:
        print(f"  {k:12s} r = {r:+.3f}   r^2 = {r * r:.3f}")

    # Standardised multiple linear regression: peak_R = X_std @ beta + b
    # Use ridge with very small lambda for numerical stability.
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    ys = (y - y.mean()) / max(y.std(), 1e-12)
    lam = 1e-4
    XtX = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    beta_std = np.linalg.solve(XtX, Xs.T @ ys)
    y_pred = Xs @ beta_std
    resid = ys - y_pred
    R2 = 1.0 - np.var(resid) / np.var(ys)
    print(f"\n--- standardised multiple regression (R^2 = {R2:.3f}) ---")
    print(f"  feature       beta_std   |beta|*sign(r)   contribution to var")
    sorted_b = sorted(zip(X_keys, beta_std, [r for _, r in rs]),
                      key=lambda t: -abs(t[1]))
    for k, b, r in sorted_b:
        print(f"  {k:12s} {b:+.3f}      {abs(b):.3f}            {abs(b * r):.3f}")

    # Correlation matrix among features (just the IC group)
    ic_keys = [k for k in X_keys if k.startswith(("eta_", "nu_", "etaw_", "nuw_", "b_"))]
    idx = [X_keys.index(k) for k in ic_keys]
    Xic = X[:, idx]
    Xic_s = (Xic - Xic.mean(0)) / np.where(Xic.std(0) > 0, Xic.std(0), 1.0)
    Cic = Xic_s.T @ Xic_s / (n - 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) IC correlation matrix
    ax = axes[0, 0]
    im = ax.imshow(Cic, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(ic_keys)))
    ax.set_xticklabels(ic_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ic_keys)))
    ax.set_yticklabels(ic_keys, fontsize=8)
    for i in range(len(ic_keys)):
        for j in range(len(ic_keys)):
            ax.text(j, i, f"{Cic[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6, color="k" if abs(Cic[i, j]) < 0.5 else "w")
    ax.set_title("Initial-condition correlation matrix (across seeds)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (0,1) bar chart of beta_std for full regression
    ax = axes[0, 1]
    feats, bs, _ = zip(*sorted_b)
    colors = ["C3" if b < 0 else "C0" for b in bs]
    ax.barh(range(len(feats)), [abs(b) for b in bs], color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|beta_std| (standardised regression coeff)")
    ax.set_title(f"Multiple regression on peak_R   R^2 = {R2:.2f}")
    ax.grid(alpha=0.3)

    # (1,0) scatter peak_R vs the strongest univariate driver
    top_key, top_r = rs[0]
    ax = axes[1, 0]
    j = X_keys.index(top_key)
    ax.scatter(X[:, j], y, color="C0")
    z = np.polyfit(X[:, j], y, 1)
    xx = np.linspace(X[:, j].min(), X[:, j].max(), 50)
    ax.plot(xx, z[0] * xx + z[1], color="C3", lw=1, ls="--",
            label=f"slope={z[0]:.3f}")
    ax.set_xlabel(top_key)
    ax.set_ylabel("peak_R [m]")
    ax.set_title(f"Top univariate driver: {top_key}  (r = {top_r:+.2f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (1,1) timing diagnostic: t_peak_R vs t_peak_wave, coloured by peak_R.
    # If peaks track wave timing, the post-WCF wave realisation drives peak_R.
    # If peaks cluster at fixed time regardless of wave, IC drives the timing.
    ax = axes[1, 1]
    sc = ax.scatter(t_peak_wave_arr, t_peak_R_arr, c=y, cmap="viridis", s=50)
    lim_t = max(t_peak_wave_arr.max(), t_peak_R_arr.max()) * 1.05
    ax.plot([0, lim_t], [0, lim_t], color="k", lw=0.5, ls="--", label="t_peak_R = t_peak_wave")
    ax.set_xlabel("t_peak_wave [s after WCF]")
    ax.set_ylabel("t_peak_R [s after WCF]")
    ax.set_title("Timing: truth peak vs WF peak (colour = peak_R)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="peak_R [m]")
    # Print timing correlation
    r_timing = float(np.corrcoef(t_peak_wave_arr, t_peak_R_arr)[0, 1])
    print(f"\n--- timing correlation: r(t_peak_R, t_peak_wave) = {r_timing:+.3f} ---")
    print(f"  t_peak_R     mean = {t_peak_R_arr.mean():.2f} s   std = {t_peak_R_arr.std():.2f} s")
    print(f"  t_peak_wave  mean = {t_peak_wave_arr.mean():.2f} s   std = {t_peak_wave_arr.std():.2f} s")

    plt.suptitle(f"What drives per-seed peak |R| variation at {TAG}? (n={n} seeds)",
                 fontsize=12)
    plt.tight_layout()
    out = THIS / "peak_R_drivers_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
