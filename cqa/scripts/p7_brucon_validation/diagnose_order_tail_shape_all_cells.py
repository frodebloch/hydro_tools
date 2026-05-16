"""Cross-cell pre-WCF Order(Surge/Sway/Yaw) tail-shape diagnostic
(sec.12.21.21.21).

Generalises diagnose_orderY_tail_shape.py across all 11 cells and all 3
DOFs.  Question: is the v1 Gaussian-tail regime-B severity estimator
defensible across the whole operating envelope, or are there cells/DOFs
where Order is heavy-tailed enough that Gaussian Phi underestimates
cap-exceedance probability?

Decision rule:
  - excess kurtosis < +0.5: Gaussian is fine (sub- or near-Gaussian)
  - excess kurtosis in [+0.5, +2.0]: mildly heavy-tailed, document
  - excess kurtosis > +2.0: Gaussian materially under-conservative;
    need heavier-tailed family (Student-t, GEV)

Outputs:
  - Console summary table (per cell x DOF: mu, sigma, skew, kurt,
    z_cap, P_gauss_at_z_cap, empirical_exc_in_window)
  - One 11x3 grid PNG showing log-density histograms with Gaussian
    overlay, per (cell, DOF).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

THIS = Path(__file__).resolve().parent
WORK = THIS / "work"

T_WCF_S = 1560.0
PRE_LO, PRE_HI = T_WCF_S - 600.0, T_WCF_S - 60.0

CELLS = [
    "pwq30", "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_h0_w45", "bf6_q10", "bf6_q10_w45",
    "bf8_h0", "bf8_h0_w45", "bf8_q10", "bf8_q10_w45",
]

# DOF spec: (display name, column index in .out, residual cap in kN/kN.m)
DOFS = [
    ("Surge", 34,   838.0),   # OrderTauSurge,  residual_x
    ("Sway",  35,  1104.0),   # OrderTauSway,   residual_y
    ("Yaw",   36, 47929.0),   # OrderTauYaw,    residual_psi
]

COL_T = 0


def load_dof(cell: str, seed: int, col: int) -> np.ndarray | None:
    f = WORK / f"{cell}_seed{seed}" / f"{cell}_seed{seed}.out"
    if not f.exists():
        return None
    d = np.loadtxt(f, skiprows=1)
    t = d[:, COL_T]
    mask = (t >= PRE_LO) & (t <= PRE_HI)
    if mask.sum() < 100:
        return None
    return d[mask, col]


def analyse(cell: str, dof_name: str, col: int, cap: float) -> dict:
    per_seed = []
    for s in range(1000, 1030):
        y = load_dof(cell, s, col)
        if y is not None:
            per_seed.append(y)
    if not per_seed:
        return {}

    mus = np.array([y.mean() for y in per_seed])
    sigs = np.array([y.std() for y in per_seed])
    # Pool standardised residuals.
    z_pool = np.concatenate([(y - mu) / sg for y, mu, sg in zip(per_seed, mus, sigs)])
    pooled_kurt = stats.kurtosis(z_pool, fisher=True)
    pooled_skew = stats.skew(z_pool)

    # z_cap relative to per-seed (mu, sigma). Use upper or lower tail
    # depending on sign of mean.
    # We're interested in saturation in EITHER direction; compute distance
    # to nearer cap.
    z_cap_upper = (cap - mus) / sigs        # to +cap
    z_cap_lower = (cap - (-mus)) / sigs     # to -cap (i.e. (-cap - mu)/-sigma if mu<0)
    # Symmetric: z = (cap - |mu|)/sigma  (always positive if vessel is
    # operating below cap)
    z_cap = (cap - np.abs(mus)) / sigs
    p_gauss = stats.norm.sf(z_cap.mean())
    # Empirical exceedance |Order| > cap in the pre-WCF window.
    n_above = sum((np.abs(y) > cap).sum() for y in per_seed)
    n_total = sum(y.size for y in per_seed)

    return {
        "cell": cell,
        "dof": dof_name,
        "n_seeds": len(per_seed),
        "n_samples": z_pool.size,
        "mu_kN": mus.mean(),
        "mu_std_kN": mus.std(),
        "sigma_kN": sigs.mean(),
        "sigma_std_kN": sigs.std(),
        "skew": pooled_skew,
        "kurt": pooled_kurt,
        "z_cap_mean": z_cap.mean(),
        "z_cap_min": z_cap.min(),
        "z_cap_max": z_cap.max(),
        "p_gauss": p_gauss,
        "empirical_exc": n_above / n_total if n_total else np.nan,
        "z_pool": z_pool,
    }


def main() -> None:
    results: dict[tuple[str, str], dict] = {}
    for cell in CELLS:
        for dof_name, col, cap in DOFS:
            r = analyse(cell, dof_name, col, cap)
            if r:
                results[(cell, dof_name)] = r

    # ----- Console table -----
    print(f"\nPre-WCF window [{PRE_LO:.0f}, {PRE_HI:.0f}]s, 30 seeds per cell")
    print(f"{'cell':<13} {'DOF':<6} | {'mu':>9} {'sigma':>8} | "
          f"{'skew':>7} {'kurt':>7} | {'z_cap':>7} {'P_gauss':>10} {'emp_exc':>10}")
    print("-" * 100)
    for cell in CELLS:
        for dof_name, _, _ in DOFS:
            r = results.get((cell, dof_name))
            if not r:
                continue
            unit = "kN.m" if dof_name == "Yaw" else "kN"
            print(f"{cell:<13} {dof_name:<6} | "
                  f"{r['mu_kN']:+9.1f} {r['sigma_kN']:8.1f} | "
                  f"{r['skew']:+7.3f} {r['kurt']:+7.3f} | "
                  f"{r['z_cap_mean']:7.2f} {r['p_gauss']:10.2e} "
                  f"{r['empirical_exc']:10.2e}")
        print()

    # ----- Heavy-tail flag -----
    print("\n=== Cells/DOFs with excess kurtosis > +0.5 (heavier than Gaussian) ===")
    heavy = [(k, r) for k, r in results.items() if r["kurt"] > 0.5]
    if heavy:
        for (cell, dof), r in heavy:
            print(f"  {cell:<14} {dof:<6}  kurt = {r['kurt']:+.3f}  skew = {r['skew']:+.3f}")
    else:
        print("  None.")

    print("\n=== Cells/DOFs with excess kurtosis > +2.0 (Gaussian Phi materially under-conservative) ===")
    very_heavy = [(k, r) for k, r in results.items() if r["kurt"] > 2.0]
    if very_heavy:
        for (cell, dof), r in very_heavy:
            print(f"  {cell:<14} {dof:<6}  kurt = {r['kurt']:+.3f}")
    else:
        print("  None.")

    # ----- 11 x 3 grid plot -----
    fig, axes = plt.subplots(len(CELLS), 3, figsize=(13, 2.0 * len(CELLS)),
                             sharex=True)
    bins = np.linspace(-6, 6, 121)
    zz = np.linspace(-6, 6, 600)
    gauss_pdf = stats.norm.pdf(zz)

    for i, cell in enumerate(CELLS):
        for j, (dof_name, _, _) in enumerate(DOFS):
            ax = axes[i, j]
            r = results.get((cell, dof_name))
            if not r:
                ax.set_visible(False)
                continue
            ax.hist(r["z_pool"], bins=bins, density=True, alpha=0.55,
                    color="steelblue")
            ax.plot(zz, gauss_pdf, "k--", lw=1.0)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            kurt_marker = "*" if r["kurt"] > 0.5 else ""
            ax.text(0.02, 0.97, f"k={r['kurt']:+.2f}{kurt_marker}",
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top",
                    color="red" if r["kurt"] > 0.5 else "black")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(dof_name)
            if j == 0:
                ax.set_ylabel(cell, fontsize=8)
            if i == len(CELLS) - 1:
                ax.set_xlabel("z = (y - mu)/sigma")

    fig.suptitle("Pre-WCF Order(Surge/Sway/Yaw) standardised log-density vs N(0,1)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = THIS / "diagnose_order_tail_shape_all_cells.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
