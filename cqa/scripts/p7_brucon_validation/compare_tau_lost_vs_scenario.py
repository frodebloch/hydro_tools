"""Compare brucon's actual tau_lost (Tx/Ty/Tz - Order) vs cqa scenario tau_lost.

Tests Mechanism 1 for the bf8-oblique residual P95 under-prediction
(see analysis.md sec. 12.21.11 verdict and sec. 12.21.12 scoping).

Brucon column semantics (verified against brucon source + behavioural test
2026-05, see analysis.md sec. 12.21.12):

  Order{Surge,Sway,Yaw}      controller's commanded body-frame thrust
                              (controller_wrapper_.tau()).
  AllocTau{Surge,Sway,Yaw}   allocated tau the DP allocator can assign
                              given its (possibly stale) view of thruster
                              state (allocator_wrapper_.allocated_tau()).
  Tx, Ty, Tz                 simulator-side total body-frame thrust applied
                              to the rigid-body solver
                              (thruster_simulator_wrapper_.total_thrust()).
                              In brucon's WCFDI mode (SetThrusterActive(idx,
                              false)) this drops correctly because the
                              simulator stops integrating the failed
                              thrusters' contribution.
  FbTau{Surge,Sway,Yaw}      DP allocator's tau_feedback, reconstructed
                              from per-thruster feedback signals
                              (allocator_wrapper_.tau_feedback()).
                              In brucon's WCFDI mode the feedback path is
                              NOT cut; the failed thrusters' feedback
                              still echoes orders, so Fb does NOT see the
                              loss. Fb is what the DP CONTROLLER thinks
                              it's delivering, not what the hull actually
                              receives.

Therefore the truest hull-experienced tau_lost during the WCFDI is:

    tau_lost_truth(t) = (Tx(t) - Order(t)) * 1e3   [N/Nm]

Plotted alongside (Fb - Order) and (Alloc - Order) for diagnostic context
(both should sit near zero in brucon's WCFDI mode).

cqa scenario formula (live_decision.py line 443):
    tau_lost(t) = -(1 - beta(t)) * tau_env
    beta(t) = 1 + (gamma_imm - 1) * exp(-t/T_realloc)
where tau_env = +b_hat(t_eval) (frozen).

If brucon's (Tx - Order) shape matches the cqa scenario tau_lost ->
scenario representation is faithful. If not (in magnitude, asymmetry,
or shape) -> Mechanism 1 is in play and WcfdiScenario needs an
extension that goes beyond the symmetric per-DOF
(alpha, gamma_imm, T_realloc) parameterisation.

Usage:
  .venv/bin/python scripts/p7_brucon_validation/compare_tau_lost_vs_scenario.py \
      --tags bf8_h0,bf8_h0_w45,bf8_q10,bf8_q10_w45
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

from harness import parse_output  # noqa: E402

ENSEMBLE_DIR = THIS / "work"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_EVAL = T_WCF - 5.0
T_PRE = 30.0
T_POST = 120.0
DT = 0.1

# cqa scenario knobs, matching what live_decision.py uses for CSOV bus_port.
GAMMA_IMM = 0.5
T_REALLOC = 10.0


def load_seed(tag: str, seed: int):
    seed_dir = ENSEMBLE_DIR / f"{tag}_seed{seed:04d}"
    out_path = seed_dir / f"{tag}_seed{seed:04d}.out"
    est_path = seed_dir / f"{tag}_seed{seed:04d}_estimator.out"
    if not out_path.exists() or not est_path.exists():
        return None
    main = parse_output(out_path)
    est = parse_output(est_path)
    t = main.columns["t"]
    if t[-1] < T_WCF + T_POST:
        return None

    # Three definitions of tau_lost (kN -> N):
    #   _t     : (Tx/Ty/Tz - Order)  hull-experienced truth (simulator-side)
    #   _fb    : (Fb       - Order)  DP controller's view (misses the loss
    #                                in brucon WCFDI mode by design)
    #   _alloc : (Alloc    - Order)  allocator's view (also misses the loss)
    def lost(prefix, ord_col):
        return (main.columns[prefix] - main.columns[ord_col]) * 1e3

    pairs_t     = [("Tx", "OrderTauSurge"), ("Ty", "OrderTauSway"),
                   ("Tz", "OrderTauYaw")]
    pairs_fb    = [("FbTauSurge", "OrderTauSurge"),
                   ("FbTauSway",  "OrderTauSway"),
                   ("FbTauYaw",   "OrderTauYaw")]
    pairs_alloc = [("AllocTauSurge", "OrderTauSurge"),
                   ("AllocTauSway",  "OrderTauSway"),
                   ("AllocTauYaw",   "OrderTauYaw")]

    t_rel = t - T_WCF
    t_grid = np.arange(-T_PRE, T_POST + DT / 2, DT)

    def stack(pairs):
        cols = [np.interp(t_grid, t_rel, lost(a, b)) for a, b in pairs]
        return np.column_stack(cols)

    tau_lost       = stack(pairs_t)      # truth (hull-applied)
    tau_lost_fb    = stack(pairs_fb)
    tau_lost_alloc = stack(pairs_alloc)

    # b_hat at t_eval: estimator log column EstBias{Surge,Sway,Yaw} in kN, kN, kNm.
    t_est = est.columns["Time"]
    i_eval = int(np.argmin(np.abs(t_est - T_EVAL)))
    b_hat = 1e3 * np.array([
        est.columns["EstBiasSurge"][i_eval],
        est.columns["EstBiasSway"][i_eval],
        est.columns["EstBiasYaw"][i_eval],
    ])
    return dict(t_grid=t_grid, tau_lost=tau_lost,
                tau_lost_fb=tau_lost_fb, tau_lost_alloc=tau_lost_alloc,
                b_hat=b_hat)


def cqa_tau_lost_t(b_hat: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """cqa scenario tau_lost(t) using tau_env = +b_hat."""
    tau_env = b_hat                          # (3,)
    tau_lost = np.zeros((len(t_grid), 3))
    post = t_grid >= 0.0
    beta_t = 1.0 + (GAMMA_IMM - 1.0) * np.exp(-t_grid[post] / T_REALLOC)
    # cqa code (live_decision.py:443):
    #     tau_lost = (beta_t - 1.0) * (-tau_env)  with tau_env = +b_hat
    # i.e. tau_lost = (1 - beta) * b_hat. At t=0: -(1 - gamma_imm) is the
    # negative immediate-loss fraction times tau_env_thrust (= -b_hat in
    # steady state), so tau_lost has the SAME sign as b_hat (the env load).
    tau_lost[post] = (beta_t - 1.0)[:, None] * (-tau_env[None, :])
    return tau_lost


def cell_summary(tag: str):
    data = []
    for s in SEEDS:
        d = load_seed(tag, s)
        if d is not None:
            data.append(d)
    if not data:
        print(f"[{tag}] no data"); return
    n = len(data)
    t_grid = data[0]["t_grid"]
    tau_lost_truth = np.array([d["tau_lost"] for d in data])  # (n, Nt, 3) -- T-Order (hull truth)
    tau_lost_fb_arr = np.array([d["tau_lost_fb"] for d in data])
    tau_lost_a_arr = np.array([d["tau_lost_alloc"] for d in data])
    b_hat_per_seed = np.array([d["b_hat"] for d in data])     # (n, 3)
    b_hat_mean = b_hat_per_seed.mean(axis=0)                  # (3,)

    # cqa scenario tau_lost using ensemble-mean b_hat
    tau_lost_scenario = cqa_tau_lost_t(b_hat_mean, t_grid)    # (Nt, 3)

    # Per-seed cqa tau_lost (per-seed b_hat) for comparison spread
    tau_lost_scenario_per_seed = np.array([
        cqa_tau_lost_t(b_hat_per_seed[k], t_grid) for k in range(n)
    ])

    # Plot 3-row (per-DOF) figure
    labels = ("surge", "sway", "yaw")
    units = ("kN", "kN", "kNm")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c, (lab, u) in enumerate(zip(labels, units)):
        ax = axes[c]
        # brucon truth (T - Order): per-seed grey + ensemble mean blue
        for k in range(n):
            ax.plot(t_grid, tau_lost_truth[k, :, c] / 1e3,
                    color="grey", alpha=0.18, lw=0.5)
        truth_mean = tau_lost_truth.mean(axis=0)[:, c] / 1e3
        ax.plot(t_grid, truth_mean, "C0-", lw=2,
                label=f"truth (Tx/Ty/Tz - Order) ens-mean (n={n})")
        # diagnostic: ens-mean Fb-Order and Alloc-Order (both miss the
        # loss in brucon's WCFDI mode by design -- documented in 12.21.12)
        ax.plot(t_grid, tau_lost_fb_arr.mean(axis=0)[:, c] / 1e3,
                "C2--", lw=1.2, alpha=0.8,
                label="(Fb - Order) DP-view ens-mean")
        ax.plot(t_grid, tau_lost_a_arr.mean(axis=0)[:, c] / 1e3,
                "C4:", lw=1.2, alpha=0.8, label="(Alloc - Order) ens-mean")
        # cqa scenario per-seed thin orange + ens-mean
        for k in range(n):
            ax.plot(t_grid, tau_lost_scenario_per_seed[k, :, c] / 1e3,
                    color="C1", alpha=0.18, lw=0.5)
        ax.plot(t_grid, tau_lost_scenario[:, c] / 1e3, "C1-", lw=2,
                label=f"cqa scenario -(1-beta(t))*b_hat (gamma={GAMMA_IMM}, "
                f"T_realloc={T_REALLOC}s)")
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel(f"tau_lost {lab} [{u}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("t since WCF [s]")
    fig.suptitle(f"tau_lost: brucon truth (T-Order) vs cqa scenario  [{tag}]", y=1.00)
    plt.tight_layout()
    out_png = THIS / f"compare_tau_lost_vs_scenario_{tag}.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Numerical summary at key moments.
    def at(t0):
        i = int(np.argmin(np.abs(t_grid - t0)))
        return tau_lost_truth.mean(axis=0)[i] / 1e3, tau_lost_scenario[i] / 1e3

    print(f"\n[{tag}]  n={n}")
    print(f"  ensemble-mean b_hat at t_eval = (S {b_hat_mean[0]/1e3:+7.1f}, "
          f"W {b_hat_mean[1]/1e3:+7.1f} kN, "
          f"Y {b_hat_mean[2]/1e3:+8.1f} kNm)")
    print(f"  channel       brucon truth (T-Order) ens-mean        cqa scenario (kN/kNm)")
    for tt in (0.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0):
        truth_v, scen_v = at(tt)
        print(f"  t={tt:6.1f}s : "
              f"S={truth_v[0]:+8.1f}  W={truth_v[1]:+8.1f}  Y={truth_v[2]:+9.1f}  | "
              f"S={scen_v[0]:+8.1f}  W={scen_v[1]:+8.1f}  Y={scen_v[2]:+9.1f}")
    print(f"  saved {out_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bf8_h0,bf8_h0_w45,bf8_q10,bf8_q10_w45")
    args = ap.parse_args()
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        cell_summary(tag)


if __name__ == "__main__":
    main()
