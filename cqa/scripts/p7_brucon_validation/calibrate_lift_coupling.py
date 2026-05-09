"""Calibrate the lift-coupling scalar K = C_Lalpha / C_D0 for the live cell.

Background
----------
Per-axis diagnostic across the validation matrix (see
diagnose_pwo_surge_coupling.py) showed that the live cell's
constant-body-frame b_hat assumption produces systematic per-axis
under-prediction:
  - head/quartering cells under-predict the OFF-axis (sway)
  - beam-on (pwo) under-predicts the OFF-axis (surge)
The mechanism is post-WCF yaw drift dpsi(t) ~ 2-4 deg, which rotates
the wave-drift force vector in body frame. The live cell -- not
allowed to know sea state at runtime -- cannot directly query a QTF
to get dF/dpsi.

Equivalent-flow / slender-body simplification
---------------------------------------------
Treat the entire b_hat (waves + wind + current combined) as if it
came from an equivalent inflow at angle alpha to the bow. Slender-body
crossflow form (Soding-like):

  F_x(alpha) = -q*A * C_D0 * cos^2(alpha)
  F_y(alpha) = -q*A * (C_Y * sin(alpha)*cos(alpha) + C_DC * sin^2)

In the small-alpha regime relevant for W2W operations (head/quartering,
|alpha| <~ 30 deg) the crossflow-drag term is negligible, so:

  F_y / F_x = (C_Y / C_D0) * tan(alpha)
            = K * tan(alpha)
  -> K = (F_y/F_x) / tan(alpha)         [estimated from observed data]

At runtime, given measured b_hat = (F_x, F_y), we infer alpha:
  alpha_eff = atan(F_y / F_x / K)

The dF_y/dpsi coupling at small alpha follows from differentiating
F_y(alpha) wrt alpha (which equals -dpsi when the vessel yaws by +dpsi):

  dF_y/dalpha = -q*A*C_Y * cos(2*alpha)   ~ -q*A*C_Y at small alpha
  q*A*C_D0 = -F_x  (at small alpha, since cos^2 ~ 1)
  -> dF_y/dalpha = F_x * (C_Y / C_D0) = F_x * K
  -> dF_y/dpsi   = -F_x * K          [yaw rotates frame opposite to alpha]

So the live cell needs only ONE scalar K, calibrated offline from
known-direction cells.

Beam-on (alpha ~ 90 deg) is the lift-stall regime where this
linearisation breaks down. We do not target it; W2W operations are
head/quartering by design.

Usage
-----
    python calibrate_lift_coupling.py

Reads b_hat_mean from each scenario_<tag>_calibration.npz file in this
directory and fits K. Prints summary table and writes
lift_coupling_K.json with the fitted value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent

# Cells where alpha_eff is well-approximated by the wave/forcing direction
# (colinear or near-colinear, dominated by waves + colinear wind/current).
# Excludes pwo (beam-on, lift stall regime, not operational target).
# Excludes split-direction cells (alpha_eff != theta_wave because wind
# contributes off-axis force).
SMALL_ALPHA_CELLS: dict[str, float] = {
    "bf6_h0":   0.0,   # alpha=0 contributes no info to K (F_y -> 0 by symmetry)
    "bf6_q10":  10.0,
    "bf8_h0":   0.0,
    "bf8_q10":  10.0,
    "pwq30":    30.0,
}

# Reference cells included in summary table but not used in K fit.
REFERENCE_CELLS: dict[str, float] = {
    "pwo":         90.0,   # beam-on, lift stall (excluded)
    "bf6_h0_w45":  None,   # alpha_eff != theta_wave (mixed)
    "bf6_q10_w45": None,
    "bf8_h0_w45":  None,
    "bf8_q10_w45": None,
    "bf4_c1_h0":   None,   # current-dominated, alpha_eff != theta_wave
    "bf4_c1_q10":  None,
}


def _load_b_hat(tag: str) -> np.ndarray | None:
    npz_path = THIS / f"scenario_{tag}_calibration.npz"
    if not npz_path.exists():
        return None
    return np.load(npz_path)["b_hat_mean"]


def _fit_K(cells: dict[str, float]) -> tuple[float, list[tuple[str, float, float]]]:
    """Fit K = C_Y/C_D0 from F_y/F_x = K * tan(alpha)."""
    obs = []  # (tag, alpha_rad, K_hat_from_this_cell)
    for tag, a_deg in cells.items():
        b = _load_b_hat(tag)
        if b is None:
            continue
        Fx, Fy = float(b[0]), float(b[1])
        if abs(a_deg) < 1e-3:
            continue   # alpha=0 gives no info on slope
        a = np.radians(a_deg)
        if abs(np.tan(a)) < 1e-9 or abs(Fx) < 1e-3:
            continue
        K_i = (Fy / Fx) / np.tan(a)
        obs.append((tag, a, K_i))

    # Weighted least-squares minimisation of (F_y/F_x - K tan(alpha))^2 across cells:
    #   K_fit = sum(r_i tan(a_i)) / sum(tan^2(a_i))
    if not obs:
        raise RuntimeError("No usable cells for K fit.")
    rs = []
    tans = []
    for tag, a, _ in obs:
        b = _load_b_hat(tag)
        Fx, Fy = float(b[0]), float(b[1])
        rs.append(Fy / Fx)
        tans.append(np.tan(a))
    rs = np.array(rs)
    tans = np.array(tans)
    K_fit = float(np.sum(rs * tans) / np.sum(tans ** 2))
    return K_fit, obs


def main():
    print("=" * 72)
    print("Calibrate lift-coupling scalar K = C_Y / C_D0")
    print("=" * 72)
    print()

    K, obs = _fit_K(SMALL_ALPHA_CELLS)

    print(f"{'cell':<14} {'a_deg':>6} {'F_x[N]':>10} {'F_y[N]':>10} "
          f"{'r=Fy/Fx':>10} {'K_i':>8}")
    for tag, a_rad in [(t, np.radians(d)) for t, d in SMALL_ALPHA_CELLS.items()]:
        b = _load_b_hat(tag)
        if b is None:
            print(f"  {tag:<12} (npz missing)")
            continue
        Fx, Fy = float(b[0]), float(b[1])
        r = Fy / Fx if abs(Fx) > 1e-3 else np.nan
        K_i = (r / np.tan(a_rad)) if abs(np.tan(a_rad)) > 1e-9 else np.nan
        a_deg = np.degrees(a_rad)
        print(f"{tag:<14} {a_deg:>6.1f} {Fx:>10.0f} {Fy:>10.0f} "
              f"{r:>10.4f} {K_i:>8.3f}")

    print()
    print(f"  Fit K = {K:.3f} per rad   "
          f"(equivalently {K * np.pi / 180:.4f} per deg in r-space)")
    print()

    # Reference cells: print observed force-angle vs the K-prediction.
    print("Reference cells (NOT used in fit):")
    print(f"{'cell':<14} {'a_deg':>6} {'r=Fy/Fx':>10} {'angF_meas':>10}  notes")
    for tag, a_deg in REFERENCE_CELLS.items():
        b = _load_b_hat(tag)
        if b is None:
            continue
        Fx, Fy = float(b[0]), float(b[1])
        angF = np.degrees(np.arctan2(Fy, -Fx))
        r = Fy / Fx if abs(Fx) > 1e-3 else float("nan")
        a_str = f"{a_deg:>6.1f}" if a_deg is not None else "  ---"
        notes = ("beam-on (lift stall)" if tag == "pwo"
                 else "alpha_eff != theta_wave (split or current dominant)")
        print(f"{tag:<14} {a_str} {r:>10.4f} {angF:>10.2f}  {notes}")

    print()
    # Sanity: predicted dF_y/dpsi for each operational cell at small alpha.
    print("Predicted dF_y/dpsi = -F_x * K at each cell:")
    for tag in list(SMALL_ALPHA_CELLS.keys()) + ["bf8_h0_w45", "bf4_c1_h0"]:
        b = _load_b_hat(tag)
        if b is None:
            continue
        Fx = float(b[0])
        dFy_dpsi = -Fx * K   # N/rad
        print(f"  {tag:<14} F_x={Fx:>10.0f}  dF_y/dpsi = {dFy_dpsi:>10.0f} N/rad "
              f"({dFy_dpsi / 57.3:>7.0f} N/deg)")

    out = {
        "K_per_rad": K,
        "model": "F_y / F_x = K * tan(alpha)  (small-alpha slender-body approx)",
        "validity": "operational head/quartering, |alpha_eff| <~ 30 deg",
        "fitted_from": list(SMALL_ALPHA_CELLS.keys()),
    }
    out_path = THIS / "lift_coupling_K.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
