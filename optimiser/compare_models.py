"""
Compare C++ vs Python propeller model interpolation approaches.

The C++ CdSeries uses:
  - Pitch axis: LINEAR interpolation between adjacent pitch entries
  - Design pitch axis: NEAREST-NEIGHBOR (snaps to closest tabulated design pitch)

The Python CSeriesPropeller uses:
  - Pitch axis: PCHIP (monotone cubic Hermite) across all entries
  - Design pitch axis: PCHIP interpolation across all entries

This script quantifies the difference at operating points relevant to vessel 206.
"""

import math
import numpy as np
from propeller_model import load_c_series_data, CSeriesPropeller, _SingleBarModel, PchipInterpolator


DATA_PATH_C440 = "/home/blofro/src/prop_model/c4_40.dat"


def cpp_style_ct_cq(data, design_pitch, pitch, beta):
    """Emulate C++ CdSeries CT/CQ calculation.

    Uses nearest-neighbor design pitch and linear pitch interpolation.
    """
    # Nearest-neighbor design pitch (C++ logic from c4_40.cpp)
    dp_table = data.design_pitches
    dp_idx = 0
    for i in range(1, len(dp_table)):
        if design_pitch < dp_table[i]:
            if abs(design_pitch - dp_table[i]) < abs(design_pitch - dp_table[i - 1]):
                dp_idx = i
            else:
                dp_idx = i - 1
            break
    else:
        dp_idx = len(dp_table) - 1

    # Linear pitch interpolation (C++ PitchDoubleIndex logic)
    p_table = data.pitches
    if pitch <= p_table[0]:
        pitch_index = 0.0
    elif pitch >= p_table[-1]:
        pitch_index = float(len(p_table) - 1)
    else:
        i = 0
        while i < len(p_table) - 1 and p_table[i] < pitch:
            i += 1
        pitch_index = float(i - 1) + (pitch - p_table[i - 1]) / (p_table[i] - p_table[i - 1])

    p1 = int(math.floor(pitch_index))
    p2 = int(math.ceil(pitch_index))
    frac = pitch_index - math.floor(pitch_index)
    weight = 1.0 - frac  # weight for p1

    def fourier_interp(A, B, beta):
        s = 0.0
        for k in range(len(A)):
            s += A[k] * math.sin(k * beta) + B[k] * math.cos(k * beta)
        return s

    ct1 = fourier_interp(data.ct_ak[dp_idx, p1], data.ct_bk[dp_idx, p1], beta)
    ct2 = fourier_interp(data.ct_ak[dp_idx, p2], data.ct_bk[dp_idx, p2], beta)
    ct = ct1 * weight + ct2 * (1.0 - weight)

    cq1 = fourier_interp(data.cq_ak[dp_idx, p1], data.cq_bk[dp_idx, p1], beta)
    cq2 = fourier_interp(data.cq_ak[dp_idx, p2], data.cq_bk[dp_idx, p2], beta)
    cq = cq1 * weight + cq2 * (1.0 - weight)

    return ct, cq


def python_style_ct_cq(data, design_pitch, pitch, beta):
    """Python CSeriesPropeller CT/CQ (PCHIP on both axes)."""
    model = _SingleBarModel(data, design_pitch)
    ct = model.CT(pitch, beta)
    cq = model.CQ(pitch, beta)
    return ct, cq


def compute_thrust_torque_power(ct, cq, Va, n, D, rho=1025.0):
    """From CT/CQ to thrust [kN], torque [kNm], power [kW]."""
    beta = math.atan2(Va, 0.7 * math.pi * n * D)
    vr = math.hypot(Va, 0.7 * math.pi * n * D)
    T = 0.001 * ct * 0.5 * rho * vr**2 * (math.pi / 4.0) * D**2  # kN
    Q = 0.001 * cq * 0.5 * rho * vr**2 * (math.pi / 4.0) * D**3  # kNm
    P = Q * 2.0 * math.pi * n  # kW
    eta0 = T * Va / (2.0 * math.pi * n * Q) if abs(Q) > 1e-12 and abs(n) > 1e-12 else 0.0
    return T, Q, P, eta0


def main():
    data = load_c_series_data(DATA_PATH_C440)

    print(f"Design pitches: {data.design_pitches}")
    print(f"Operating pitches: {data.pitches}")
    print(f"Number of Fourier coefficients: {data.n_coefficients}")
    print()

    D = 4.80  # m
    design_pd = 0.771
    rho = 1025.0

    # For design_pd=0.771, C++ snaps to dp_table[0]=0.8
    print(f"Design P/D = {design_pd}")
    print(f"C++ nearest design pitch: 0.8 (index 0)")
    print()

    # ================================================================
    # Test 1: CT/CQ comparison at a grid of pitch and beta values
    # ================================================================
    print("=" * 90)
    print("CT/CQ COMPARISON: C++ (linear pitch, nearest DP) vs Python (PCHIP pitch, PCHIP DP)")
    print("=" * 90)
    print(f"{'P/D':>6} {'beta':>8} {'CT_cpp':>10} {'CT_py':>10} {'CT_err%':>10} "
          f"{'CQ_cpp':>10} {'CQ_py':>10} {'CQ_err%':>10}")
    print("-" * 90)

    for pd in [0.5, 0.6, 0.694, 0.7, 0.771, 0.8, 0.9, 1.0, 1.1, 1.15]:
        for beta_deg in [15, 20, 25, 30]:
            beta = math.radians(beta_deg)
            ct_cpp, cq_cpp = cpp_style_ct_cq(data, design_pd, pd, beta)
            ct_py, cq_py = python_style_ct_cq(data, design_pd, pd, beta)
            ct_err = 100 * (ct_cpp - ct_py) / ct_py if abs(ct_py) > 1e-10 else 0
            cq_err = 100 * (cq_cpp - cq_py) / cq_py if abs(cq_py) > 1e-10 else 0
            print(f"{pd:6.3f} {beta_deg:6d}° {ct_cpp:10.4f} {ct_py:10.4f} {ct_err:9.2f}% "
                  f"{cq_cpp:10.4f} {cq_py:10.4f} {cq_err:9.2f}%")
        print()

    # ================================================================
    # Test 2: Thrust, torque, power, eta0 at vessel 206 conditions
    # ================================================================
    print()
    print("=" * 90)
    print("THRUST/POWER/ETA0 COMPARISON at vessel 206 operating points")
    print("=" * 90)

    w = 0.228  # wake fraction

    cases = [
        # (speed_kn, pd, rpm, label)
        (8.0, 0.694, 70.6, "8kn factory"),
        (8.0, 0.900, 59.0, "8kn optimiser (approx)"),
        (10.0, 0.850, 78.0, "10kn factory"),
        (10.0, 0.950, 73.0, "10kn optimiser (approx)"),
        (12.0, 1.000, 96.8, "12kn factory"),
        (12.0, 1.050, 92.0, "12kn optimiser (approx)"),
        (14.0, 1.100, 113.5, "14kn factory"),
    ]

    print(f"{'Case':>25} {'T_cpp':>8} {'T_py':>8} {'Q_cpp':>8} {'Q_py':>8} "
          f"{'PD_cpp':>8} {'PD_py':>8} {'eta_cpp':>8} {'eta_py':>8} {'eta_err%':>9}")
    print("-" * 110)

    for vs_kn, pd, rpm, label in cases:
        Va = vs_kn * 0.5144 * (1 - w)
        n = rpm / 60.0
        beta = math.atan2(Va, 0.7 * math.pi * n * D)

        ct_cpp, cq_cpp = cpp_style_ct_cq(data, design_pd, pd, beta)
        ct_py, cq_py = python_style_ct_cq(data, design_pd, pd, beta)

        T_cpp, Q_cpp, P_cpp, eta_cpp = compute_thrust_torque_power(ct_cpp, cq_cpp, Va, n, D, rho)
        T_py, Q_py, P_py, eta_py = compute_thrust_torque_power(ct_py, cq_py, Va, n, D, rho)

        eta_err = 100 * (eta_cpp - eta_py) / eta_py if abs(eta_py) > 1e-10 else 0

        print(f"{label:>25} {T_cpp:8.1f} {T_py:8.1f} {Q_cpp:8.1f} {Q_py:8.1f} "
              f"{P_cpp:8.1f} {P_py:8.1f} {eta_cpp:8.4f} {eta_py:8.4f} {eta_err:8.2f}%")

    # ================================================================
    # Test 3: Isolate the two effects
    # ================================================================
    print()
    print("=" * 90)
    print("ISOLATING EFFECTS: design pitch only vs pitch interpolation only")
    print("=" * 90)

    # Effect A: Design pitch difference only (both use linear pitch interpolation)
    # Effect B: Pitch interpolation difference only (both use same design pitch)
    print()
    print("Effect A: Design pitch (nearest=0.8 vs PCHIP at 0.771), with LINEAR pitch interp")
    print("Effect B: Pitch interp (linear vs PCHIP), with nearest design pitch=0.8")
    print()
    print(f"{'P/D':>6} {'beta':>6} {'CT_nn_lin':>10} {'CT_nn_pch':>10} {'CT_pch_lin':>10} {'CT_pch_pch':>10} "
          f"{'DP_eff%':>8} {'PI_eff%':>8}")
    print("-" * 80)

    for pd in [0.694, 0.771, 0.900, 1.000]:
        beta_deg = 20
        beta = math.radians(beta_deg)

        # Case 1: nearest DP + linear pitch (= C++)
        ct_nn_lin, _ = cpp_style_ct_cq(data, design_pd, pd, beta)

        # Case 2: nearest DP + PCHIP pitch
        # Use _SingleBarModel with dp=0.8 (only 1 dp used)
        model_nn = _SingleBarModel(data, 0.8)  # forces dp_lo=dp_hi for dp=0.8
        ct_nn_pch = model_nn.CT(pd, beta)

        # Case 3: PCHIP DP + linear pitch
        # Evaluate Fourier at each dp with linear pitch interp, then PCHIP across dp
        p_table = data.pitches
        if pd <= p_table[0]:
            pitch_index = 0.0
        elif pd >= p_table[-1]:
            pitch_index = float(len(p_table) - 1)
        else:
            i = 0
            while i < len(p_table) - 1 and p_table[i] < pd:
                i += 1
            pitch_index = float(i - 1) + (pd - p_table[i - 1]) / (p_table[i] - p_table[i - 1])
        p1 = int(math.floor(pitch_index))
        p2 = int(math.ceil(pitch_index))
        frac = pitch_index - math.floor(pitch_index)
        weight = 1.0 - frac

        def fourier_at_dp(dp_idx, p_idx):
            s = 0.0
            for k in range(data.n_coefficients):
                s += data.ct_ak[dp_idx, p_idx, k] * math.sin(k * beta) + \
                     data.ct_bk[dp_idx, p_idx, k] * math.cos(k * beta)
            return s

        dp_vals = []
        for di in range(len(data.design_pitches)):
            ct1 = fourier_at_dp(di, p1)
            ct2 = fourier_at_dp(di, p2)
            dp_vals.append(ct1 * weight + ct2 * (1.0 - weight))
        dp_vals = np.array(dp_vals)
        interp = PchipInterpolator(data.design_pitches, dp_vals)
        ct_pch_lin = interp(design_pd)

        # Case 4: PCHIP DP + PCHIP pitch (= Python)
        ct_pch_pch, _ = python_style_ct_cq(data, design_pd, pd, beta)

        dp_eff = 100 * (ct_nn_lin - ct_pch_lin) / ct_pch_pch if abs(ct_pch_pch) > 1e-10 else 0
        pi_eff = 100 * (ct_nn_lin - ct_nn_pch) / ct_pch_pch if abs(ct_pch_pch) > 1e-10 else 0

        print(f"{pd:6.3f} {beta_deg:4d}° {ct_nn_lin:10.4f} {ct_nn_pch:10.4f} "
              f"{ct_pch_lin:10.4f} {ct_pch_pch:10.4f} {dp_eff:7.2f}% {pi_eff:7.2f}%")


if __name__ == "__main__":
    main()
