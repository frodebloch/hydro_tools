"""Validate cqa intact wave-frequency sigma_y at the vessel CG against
brucon pwq30 ensemble per-seed WF std.

Truth-side WF std definition (preferred): high-pass Butterworth at fc=0.05 Hz
applied to body-frame y (= project_ned_to_body(x, y, heading)). The 0.05 Hz
cutoff sits cleanly between LF (omega < 0.1 rad/s, controller bandwidth
omega_sway = 0.08 rad/s = 0.013 Hz) and WF (Tp = 10 s -> 0.1 Hz). Brucon's
HfPosY is the wave-filter's WF estimate and is biased low (the 2nd-order
notch filter has finite bandwidth and rolls off WF energy outside its pass
band) -- do NOT use HfPosY as truth.

Conventions discovered while writing this script (verified against seed 1000):
  - brucon `.out` `heading` column is in DEGREES, wrapped to (-180, +180].
    project_ned_to_body must call np.deg2rad first.
  - brucon `_estimator.out` `EstPosX/Y` are in NED, not body. The DP
    estimator outputs NED-frame LF position; convert via the same
    project_ned_to_body if you want the controller's body-frame view.
  - HfPosX/Y are body-frame (the wave-filter operates on body-frame y).

Comparison target:
    cqa.wave_response.sigma_pos_wave_at_body_point( body_point=(0,0,0) )
    with Hs=4.196, Tp=10.224, theta_wave_rel=30 deg.
The brucon-matched spreading is cos^n with n=2; brucon's wave spectrum
default is bretschneider. With those settings cqa matches brucon truth-side
WF sigma_y to within 3% (0.350 m vs 0.339 m).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cqa.rao import load_pdstrip_rao  # noqa: E402
from cqa.wave_response import sigma_pos_wave_at_body_point  # noqa: E402
from cqa.sea_spreading import SeaSpreading  # noqa: E402


PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
WORK = Path(__file__).resolve().parent / "work"
SEEDS = list(range(1000, 1030))

HS = 4.19571865443425
TP = 10.22443464601827
THETA_WAVE_REL = np.deg2rad(30.0)
T_WCF = 560.0
WIN_PRE = (200.0, T_WCF - 5.0)        # large stationary window pre-WCF
WIN_POST = (T_WCF + 80.0, 740.0)      # well after recovery


def project_ned_to_body_y(x_ned, y_ned, heading_deg):
    """Brucon 'heading' column is in DEGREES, wrapped to (-180, +180]."""
    h = np.deg2rad(heading_deg)
    c = np.cos(h); s = np.sin(h)
    return -s * x_ned + c * y_ned


def parse_main_out(p: Path):
    with p.open("r") as fh:
        headers = fh.readline().strip().split("\t")
    data = np.loadtxt(p, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(headers)}


def parse_est_out(p: Path):
    with p.open("r") as fh:
        headers = fh.readline().strip().split("\t")
    data = np.loadtxt(p, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(headers)}, headers


def sliding_mean(arr, t, window_s):
    """Centred sliding-window mean for uniformly-sampled arr."""
    dt = t[1] - t[0]
    n = max(3, int(round(window_s / dt)))
    if n % 2 == 0:
        n += 1
    # Box filter via cumulative sum, with edge handling that pads by reflection
    pad = n // 2
    arr_pad = np.concatenate([arr[pad:0:-1], arr, arr[-2:-2 - pad:-1]])
    csum = np.cumsum(np.insert(arr_pad, 0, 0.0))
    out = (csum[n:] - csum[:-n]) / n
    return out[: len(arr)]


def per_seed_wf_std(seed: int):
    main = parse_main_out(WORK / f"pwq30_seed{seed}" / f"pwq30_seed{seed}.out")
    est, est_headers = parse_est_out(WORK / f"pwq30_seed{seed}" / f"pwq30_seed{seed}_estimator.out")

    t = main["t"]
    heading = main["heading"]
    x = main["x"]
    y = main["y"]
    y_body = project_ned_to_body_y(x, y, heading)

    hf_y_key = est_headers[17]
    y_hf = est[hf_y_key]
    t_est = est[est_headers[0]]
    y_hf_on_t = np.interp(t, t_est, y_hf)

    # Truth-side WF: high-pass with Butterworth at fc = 0.05 Hz (omega_c=0.314 rad/s)
    # which sits cleanly between LF (omega < 0.1 rad/s) and WF (omega ~ 0.6 rad/s).
    from scipy.signal import butter, filtfilt
    dt = float(t[1] - t[0])
    fs = 1.0 / dt
    fc_hp = 0.05  # Hz
    b, a = butter(4, fc_hp / (fs / 2), btype="highpass")
    y_wf_truth_hp = filtfilt(b, a, y_body)

    # Also: 20s sliding-mean residual for cross-check
    y_wf_truth_sm20 = y_body - sliding_mean(y_body, t, 20.0)

    def std_in_win(arr, win):
        m = (t >= win[0]) & (t < win[1])
        a = arr[m]
        return float(np.std(a, ddof=1))

    return {
        "sigma_y_hf_pre": std_in_win(y_hf_on_t, WIN_PRE),
        "sigma_y_hf_post": std_in_win(y_hf_on_t, WIN_POST),
        "sigma_y_truth_hp_pre": std_in_win(y_wf_truth_hp, WIN_PRE),
        "sigma_y_truth_hp_post": std_in_win(y_wf_truth_hp, WIN_POST),
        "sigma_y_truth_sm20_pre": std_in_win(y_wf_truth_sm20, WIN_PRE),
        "sigma_y_truth_sm20_post": std_in_win(y_wf_truth_sm20, WIN_POST),
        "sigma_y_hf_key": hf_y_key,
    }


def main():
    print(f"--- ensemble per-seed WF sway std (brucon truth side) ---")
    pre_hf = []; post_hf = []
    pre_hp = []; post_hp = []
    pre_sm = []; post_sm = []
    key_seen = None
    for seed in SEEDS:
        try:
            r = per_seed_wf_std(seed)
        except FileNotFoundError as e:
            print(f"  seed {seed}: missing {e}")
            continue
        pre_hf.append(r["sigma_y_hf_pre"]); post_hf.append(r["sigma_y_hf_post"])
        pre_hp.append(r["sigma_y_truth_hp_pre"]); post_hp.append(r["sigma_y_truth_hp_post"])
        pre_sm.append(r["sigma_y_truth_sm20_pre"]); post_sm.append(r["sigma_y_truth_sm20_post"])
        key_seen = r["sigma_y_hf_key"]
    pre_hf = np.array(pre_hf); post_hf = np.array(post_hf)
    pre_hp = np.array(pre_hp); post_hp = np.array(post_hp)
    pre_sm = np.array(pre_sm); post_sm = np.array(post_sm)
    print(f"  estimator HF column header: {key_seen!r}")
    print(f"  N seeds = {len(pre_hf)}")
    print(f"  brucon HfPosY (wave-filter estimate, biased low):")
    print(f"    pre-WCF  mean={pre_hf.mean():.4f}  std={pre_hf.std(ddof=1):.4f}")
    print(f"    post-rec mean={post_hf.mean():.4f}  std={post_hf.std(ddof=1):.4f}")
    print(f"  truth-side, Butterworth HP fc=0.05 Hz:")
    print(f"    pre-WCF  mean={pre_hp.mean():.4f}  std={pre_hp.std(ddof=1):.4f}")
    print(f"    post-rec mean={post_hp.mean():.4f}  std={post_hp.std(ddof=1):.4f}")
    print(f"  truth-side, 20s sliding-mean residual:")
    print(f"    pre-WCF  mean={pre_sm.mean():.4f}  std={pre_sm.std(ddof=1):.4f}")
    print(f"    post-rec mean={post_sm.mean():.4f}  std={post_sm.std(ddof=1):.4f}")

    print(f"\n--- cqa intact WF prediction at body CG (0,0,0) ---")
    print(f"  Hs={HS:.4f}, Tp={TP:.4f}, theta_rel={np.rad2deg(THETA_WAVE_REL):.1f} deg")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    # Variant A: defaults (bretschneider + cos_2s s=4)
    res_a = sigma_pos_wave_at_body_point(
        body_point=(0.0, 0.0, 0.0),
        rao_table=rao, Hs=HS, Tp=TP, theta_wave_rel=THETA_WAVE_REL,
    )
    # Variant B: long-crested
    res_b = sigma_pos_wave_at_body_point(
        body_point=(0.0, 0.0, 0.0),
        rao_table=rao, Hs=HS, Tp=TP, theta_wave_rel=THETA_WAVE_REL,
        spreading=SeaSpreading.long_crested(),
    )
    # Variant C: brucon-exact spreading cos^2 (n=2)
    res_c = sigma_pos_wave_at_body_point(
        body_point=(0.0, 0.0, 0.0),
        rao_table=rao, Hs=HS, Tp=TP, theta_wave_rel=THETA_WAVE_REL,
        spreading=SeaSpreading.cos_n(2),
    )
    # Variant D: JONSWAP + cos^2
    res_d = sigma_pos_wave_at_body_point(
        body_point=(0.0, 0.0, 0.0),
        rao_table=rao, Hs=HS, Tp=TP, theta_wave_rel=THETA_WAVE_REL,
        spectrum="jonswap", gamma=3.3, spreading=SeaSpreading.cos_n(2),
    )
    print(f"  A bretsch + cos_2s(s=4) :  sigma_y={res_a.sigma_y_wave_m:.4f} m  sigma_x={res_a.sigma_x_wave_m:.4f} m")
    print(f"  B bretsch + long-crested:  sigma_y={res_b.sigma_y_wave_m:.4f} m  sigma_x={res_b.sigma_x_wave_m:.4f} m")
    print(f"  C bretsch + cos^2 (brucon): sigma_y={res_c.sigma_y_wave_m:.4f} m  sigma_x={res_c.sigma_x_wave_m:.4f} m")
    print(f"  D JONSWAP + cos^2       :  sigma_y={res_d.sigma_y_wave_m:.4f} m  sigma_x={res_d.sigma_x_wave_m:.4f} m")

    print(f"\n--- comparison to brucon truth-side HP {pre_hp.mean():.4f} m ---")
    for name, r in [("A bretsch+cos_2s", res_a), ("B bretsch+LC", res_b),
                    ("C bretsch+cos^2", res_c), ("D JONSWAP+cos^2", res_d)]:
        print(f"  {name:18s} ratio cqa/brucon = {r.sigma_y_wave_m/pre_hp.mean():.3f}")


if __name__ == "__main__":
    main()
