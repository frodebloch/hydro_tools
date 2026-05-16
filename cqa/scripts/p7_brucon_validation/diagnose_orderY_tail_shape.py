"""Pre-WCF OrderY tail-shape diagnostic for the bf8_q10_w45 ensemble
(sec.12.21.21.20).

Motivation (revised after sec.12.21.21.19 retraction): the pre->post WCF
sigma ratio is ~1.0 (no closed-loop amplification), so the v1 "no gamma"
design is correct.  However, the v1 Gaussian-tail predictor still under-
predicts cap exceedance by ~5-6 orders of magnitude (P_sat ~ 5e-10 vs
observed ~ 1e-4).  Hypothesis: pre-WCF OrderY has heavier-than-Gaussian
tails because LF wave-drift force is the squared envelope of a narrow-
band WF process (chi-square-like), and slow groups (Tp=10s implies
group periods ~80-150s) introduce additional non-Gaussian structure.

This script:

  1. Pools per-seed pre-WCF OrderY from a stationary window
     [T_WCF-600, T_WCF-60] s across 30 seeds, *each de-meaned and
     scaled by its own sigma* to a unit zero-mean ensemble.

  2. Plots:
     - Histogram with Gaussian overlay
     - QQ plot vs Normal
     - Survival function (CCDF) on log-y, with Gaussian and Student-t
       reference curves
     - Per-seed (mu, sigma, skew, kurtosis) scatter

  3. Computes z = (cap - mu) / sigma per seed at the residual polytope
     (1104 kN), then prints Gaussian Phi(-z) vs the empirical exceedance
     rate from the pooled tail.

  4. Estimates df for the best-fit Student-t (method of moments via
     excess kurtosis: df = 6/kurt + 4) and reports the Student-t tail
     probability at the same z.

Output: diagnose_orderY_tail_shape.png + console table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

THIS = Path(__file__).resolve().parent
WORK = THIS / "work"

T_WCF_S = 1560.0
TAG = "bf8_q10_w45"
PRE_LO, PRE_HI = T_WCF_S - 600.0, T_WCF_S - 60.0
CAP_Y_RES_KN = 1104.0  # residual polytope sway cap

COL_T = 0
COL_ORDER_Y = 35


def load_pre_wcf_orderY(seed: int) -> np.ndarray | None:
    f = WORK / f"{TAG}_seed{seed}" / f"{TAG}_seed{seed}.out"
    if not f.exists():
        return None
    d = np.loadtxt(f, skiprows=1)
    t = d[:, COL_T]
    mask = (t >= PRE_LO) & (t <= PRE_HI)
    if mask.sum() < 100:
        return None
    return d[mask, COL_ORDER_Y]


def main() -> None:
    per_seed: dict[int, np.ndarray] = {}
    for s in range(1000, 1051):
        y = load_pre_wcf_orderY(s)
        if y is not None:
            per_seed[s] = y
    print(f"Loaded {len(per_seed)} seeds; pre-WCF window [{PRE_LO:.0f}, {PRE_HI:.0f}]s")

    seeds = sorted(per_seed.keys())
    mus = np.array([per_seed[s].mean() for s in seeds])
    sigs = np.array([per_seed[s].std() for s in seeds])
    skews = np.array([stats.skew(per_seed[s]) for s in seeds])
    kurts = np.array([stats.kurtosis(per_seed[s], fisher=True) for s in seeds])  # excess

    # Pool standardised residuals across seeds.
    z_pool = np.concatenate([
        (per_seed[s] - mus[i]) / sigs[i] for i, s in enumerate(seeds)
    ])
    print(f"\nPooled standardised sample: N = {z_pool.size}")
    print(f"  mean  = {z_pool.mean():+.4f}  (expect 0)")
    print(f"  std   = {z_pool.std():+.4f}  (expect 1)")
    print(f"  skew  = {stats.skew(z_pool):+.4f}  (expect 0 if Gaussian)")
    print(f"  kurt  = {stats.kurtosis(z_pool, fisher=True):+.4f}  (excess; 0 if Gaussian)")

    print(f"\nPer-seed dispersion (across {len(seeds)} seeds):")
    print(f"  mu:    mean = {mus.mean():+7.1f} kN   sd = {mus.std():6.1f} kN")
    print(f"  sigma: mean = {sigs.mean():7.1f} kN   sd = {sigs.std():6.1f} kN")
    print(f"  skew:  mean = {skews.mean():+.3f}   sd = {skews.std():.3f}")
    print(f"  kurt:  mean = {kurts.mean():+.3f}   sd = {kurts.std():.3f}")

    # Tail-probability comparison at z = (cap - mu)/sigma per seed.
    z_cap = (CAP_Y_RES_KN - mus) / sigs
    print(f"\nPer-seed z to upper cap (residual_y={CAP_Y_RES_KN:.0f} kN):")
    print(f"  z:     mean = {z_cap.mean():.2f}  min = {z_cap.min():.2f}  max = {z_cap.max():.2f}")

    # Method-of-moments Student-t df (Fisher excess kurt; t dof = 6/k + 4).
    pooled_excess_kurt = stats.kurtosis(z_pool, fisher=True)
    if pooled_excess_kurt > 0.01:
        t_df = 6.0 / pooled_excess_kurt + 4.0
    else:
        t_df = np.inf
    print(f"\nMethod-of-moments Student-t df = {t_df:.2f}")

    # Empirical upper-tail exceedance at typical z_cap values.
    print(f"\nUpper-tail exceedance comparison at z = z_cap:")
    z_test_vals = [3.0, 4.0, 5.0, z_cap.mean(), 6.0]
    print(f"  {'z':>6} | {'P_gauss':>10} | {'P_student_t':>12} | {'P_emp_pool':>11}")
    for zt in z_test_vals:
        p_g = stats.norm.sf(zt)
        if np.isfinite(t_df):
            # Standardise Student-t to unit variance: t-distrib has var = df/(df-2)
            # so to compare at standardised z, scale arg by sqrt((df-2)/df).
            scale_to_unit = np.sqrt((t_df - 2.0) / t_df) if t_df > 2 else 1.0
            p_t = stats.t.sf(zt / scale_to_unit, df=t_df)
        else:
            p_t = p_g
        p_emp = (z_pool > zt).mean()
        print(f"  {zt:6.2f} | {p_g:10.2e} | {p_t:12.2e} | {p_emp:11.2e}")

    # Empirical exceedance at the actual cap per seed: count fraction of
    # samples with OrderY > cap.
    n_above_cap = sum((per_seed[s] > CAP_Y_RES_KN).sum() for s in seeds)
    n_total = sum(per_seed[s].size for s in seeds)
    print(f"\nPre-WCF empirical: OrderY > cap_residual = {n_above_cap}/{n_total} = {n_above_cap/n_total:.2e}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Histogram of standardised pooled OrderY + Gaussian + Student-t.
    ax = axes[0, 0]
    bins = np.linspace(-6, 6, 121)
    ax.hist(z_pool, bins=bins, density=True, alpha=0.55, color="steelblue",
            label=f"pooled (N={z_pool.size})")
    zz = np.linspace(-6, 6, 600)
    ax.plot(zz, stats.norm.pdf(zz), "k--", lw=1.4, label="N(0,1)")
    if np.isfinite(t_df):
        scale = np.sqrt((t_df - 2.0) / t_df) if t_df > 2 else 1.0
        ax.plot(zz, stats.t.pdf(zz / scale, df=t_df) / scale, "r-", lw=1.4,
                label=f"Student-t (df={t_df:.1f})")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("standardised OrderY z = (y - mu)/sigma")
    ax.set_ylabel("density (log)")
    ax.set_title("Pre-WCF OrderY distribution (pooled, log-density)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (0,1) QQ plot vs Normal.
    ax = axes[0, 1]
    n = z_pool.size
    qs = np.linspace(0.5 / n, 1 - 0.5 / n, min(n, 4000))
    sample_q = np.quantile(z_pool, qs)
    theor_q = stats.norm.ppf(qs)
    ax.plot(theor_q, sample_q, ".", ms=2.0, alpha=0.5)
    lo, hi = -6, 6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("normal quantile")
    ax.set_ylabel("sample quantile")
    ax.set_title("QQ vs Normal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # (1,0) Survival function (CCDF) on log-y, upper tail only.
    ax = axes[1, 0]
    z_sorted = np.sort(z_pool)
    surv = 1.0 - (np.arange(z_sorted.size) + 1) / z_sorted.size
    pos = z_sorted > 0
    ax.plot(z_sorted[pos], surv[pos], "b-", lw=1.4,
            label=f"empirical (N={z_pool.size})")
    zz_pos = np.linspace(0.1, 6, 200)
    ax.plot(zz_pos, stats.norm.sf(zz_pos), "k--", lw=1.2, label="N(0,1)")
    if np.isfinite(t_df):
        scale = np.sqrt((t_df - 2.0) / t_df) if t_df > 2 else 1.0
        ax.plot(zz_pos, stats.t.sf(zz_pos / scale, df=t_df), "r-", lw=1.2,
                label=f"Student-t (df={t_df:.1f})")
    # Mark z_cap as vertical band.
    ax.axvspan(z_cap.min(), z_cap.max(), color="orange", alpha=0.18,
               label=f"z_cap range [{z_cap.min():.1f}, {z_cap.max():.1f}]")
    ax.set_yscale("log")
    ax.set_xlim(0, 7)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("z")
    ax.set_ylabel("P(Z > z)  (log)")
    ax.set_title("Upper-tail survival function")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (1,1) Per-seed (sigma, kurt) scatter.
    ax = axes[1, 1]
    sc = ax.scatter(sigs, kurts, c=skews, cmap="coolwarm", s=40,
                    vmin=-0.5, vmax=0.5, edgecolor="black", lw=0.4)
    ax.axhline(0, color="black", ls="--", lw=0.8, label="Gaussian (kurt=0)")
    ax.set_xlabel("per-seed sigma [kN]")
    ax.set_ylabel("per-seed excess kurtosis")
    ax.set_title("Per-seed moments (color = skewness)")
    plt.colorbar(sc, ax=ax, label="skew")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"bf8_q10_w45 (CSOV) -- pre-WCF OrderY tail shape, {len(seeds)} seeds, "
        f"window [{PRE_LO:.0f},{PRE_HI:.0f}]s",
        fontsize=12,
    )
    fig.tight_layout()
    out = THIS / "diagnose_orderY_tail_shape.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
