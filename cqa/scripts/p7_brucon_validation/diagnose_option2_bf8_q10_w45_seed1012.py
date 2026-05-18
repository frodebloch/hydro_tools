"""Diagnostic: probe Option-2 internals on bf8_q10_w45 seed 1012 (known outlier)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("scripts/p7_brucon_validation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, ".")

import live_cell_per_seed_pwq30 as live_cell
from live_cell_per_seed_pwq30 import _load_tsv, load_seed, build_live_sigma_posterior
from cqa.config import csov_default_config
from cqa.live_decision import LiveObserverState
from cqa.live_operator_view import summarise_for_operator_live
from cqa.live_regime_b import (
    DEFAULT_WINDOW_S as REGB_WIN_S,
    estimate_post_wcf_excursion_distribution,
    estimate_regime_b_severity,
)
import roll_up_live_operator_panel as roll
from cqa.transient_obs import build_observer_augmented_system_full
from cqa.vessel import LinearVesselModel
from cqa.controller import LinearDpController

tag = "bf8_q10_w45"
roll._set_cell(tag) if hasattr(roll, "_set_cell") else None
# Replicate cell setup
from _constants import T_WCF_S as T_WCF
live_cell.TAG = tag
live_cell.T_WCF = T_WCF
live_cell.T_EVAL = T_WCF - 5.0
live_cell.WIN_END = T_WCF - 1.0
live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
live_cell.CALIB_NPZ = ROOT / f"scenario_{tag}_calibration.npz"

seed = 1012
d = load_seed(seed)
sigma_R_b_hat_m = float(np.load(live_cell.CALIB_NPZ, allow_pickle=True)["sigma_R_b_hat_m"])
sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)

seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
M = _load_tsv(main_p)
t_main = M["t"]
regb_lo = live_cell.T_EVAL - REGB_WIN_S
regb_hi = live_cell.T_EVAL
regb_m = (t_main >= regb_lo) & (t_main <= regb_hi)
tau_buffer = 1e3 * np.column_stack([M["Tx"][regb_m], M["Ty"][regb_m], M["Tz"][regb_m]])
dt = float(np.median(np.diff(t_main[regb_m])))
regb_fs_hz = 1.0 / dt if dt > 0 else 10.0

# Direct Regime-B
rb = estimate_regime_b_severity(
    tau_buffer=tau_buffer, fs_hz=regb_fs_hz,
    geometry=roll._REGB_GEOMETRY, surge_cap_N=roll._REGB_SURGE_CAP_N,
)
print(f"Regime-B on bf8_q10_w45 seed {seed}:")
print(f"  mu  [kN, kN, kNm] = {rb.mu / np.array([1e3,1e3,1e3])}")
print(f"  sig [kN, kN, kNm] = {rb.sigma / np.array([1e3,1e3,1e3])}")
print(f"  cap [kN, kN, kNm] = {rb.cap_residual / np.array([1e3,1e3,1e3])}")
print(f"  p_sat = {rb.p_sat}")
print(f"  severity = {rb.severity:.3e}  traffic = {rb.traffic}")

# Build aug (mirror the panel)
cfg = csov_default_config()
# Pull omega_n / zeta from the panel build_aug helper
from cqa.live_operator_view import _build_aug_for_live
aug = _build_aug_for_live(cfg, Tp_obs_s=10.0)
print()
print(f"  aug.A.shape = {aug.A.shape}, aug.B_lost.shape = {aug.B_lost.shape}")

# Try Option 2 at horizons 60 and 200
for T_h in (60.0, 200.0):
    excur = estimate_post_wcf_excursion_distribution(
        aug, mu=rb.mu, sigma=rb.sigma, cap_residual=rb.cap_residual,
        t_horizon_s=T_h,
    )
    print(f"\nT_horizon = {T_h:.0f} s")
    print(f"  mu_dtau [kN, kN, kNm] = {excur.mu_dtau / np.array([1e3,1e3,1e3])}")
    print(f"  sig_dtau [kN, kN, kNm] = {excur.sigma_dtau / np.array([1e3,1e3,1e3])}")
    print(f"  mu_eta [m, m, rad] = {excur.mu_eta}")
    print(f"  sig_eta [m, m, rad] = {excur.sigma_eta}")
    print(f"  nu0+ [Hz] = {excur.nu_0_plus}")
    print(f"  q_vm = {excur.q_vanmarcke}")
    print(f"  p50 [m, m, rad] = {excur.eta_p50}")
    print(f"  p95 [m, m, rad] = {excur.eta_p95}")
    print(f"  R_xy_p95_with_offset = {excur.eta_xy_p95_with_offset:.4f} m")
